import os
import re
import torch
import torch.nn as nn
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from docx import Document
from PyPDF2 import PdfReader
from PIL import Image, ImageTk


# =========================================================
# 1. CONFIGURATION
# =========================================================

ENGLISH_MODEL_PATH = "pii_bilstm_english.pth"
HINDI_MODEL_PATH = "pii_bilstm_hindi.pth"

MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. MODEL CLASS
# =========================================================

class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_labels, pad_idx, dropout, num_layers):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=pad_idx
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.0
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_labels)

    def forward(self, x, lengths):
        x = self.embedding(x)
        x = self.embedding_dropout(x)

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        packed_out, _ = self.lstm(packed)

        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=x.size(1)
        )

        out = self.dropout(out)
        logits = self.fc(out)

        return logits


# =========================================================
# 3. TOKENIZATION AND LANGUAGE DETECTION
# =========================================================

def tokenize_text(text):
    pattern = (
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        r"|https?://\S+"
        r"|www\.\S+"
        r"|[\u0900-\u097F]+"
        r"|[A-Za-z]+(?:'[A-Za-z]+)?"
        r"|\d+(?:[-.]\d+)*"
        r"|[^\w\s]"
    )
    return re.findall(pattern, text)


def detect_language(text):
    hindi_chars = re.findall(r"[\u0900-\u097F]", text)
    if len(hindi_chars) > 0:
        return "Hindi"
    return "English"


def pad_or_truncate(sequence, max_len, pad_value):
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [pad_value] * (max_len - len(sequence))


# =========================================================
# 4. RULE HELPERS
# =========================================================

def is_email(token):
    return re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", token) is not None


def is_phone(token):
    return re.match(r"^\+?\d[\d\-\.\s]{7,}\d$", token) is not None


def is_url(token):
    return re.match(r"^(https?://|www\.)", token.lower()) is not None


def apply_english_name_fallback(tokens, labels):
    final = labels[:]
    lower = [t.lower() for t in tokens]

    for i in range(len(tokens) - 3):
        if lower[i:i + 3] == ["my", "name", "is"]:
            if i + 3 < len(tokens) and final[i + 3] == "O":
                final[i + 3] = "B-NAME_FALLBACK"
            if i + 4 < len(tokens) and final[i + 4] == "O":
                final[i + 4] = "I-NAME_FALLBACK"

    for i in range(len(tokens) - 2):
        if lower[i:i + 2] in [["i", "am"], ["this", "is"]]:
            if i + 2 < len(tokens) and final[i + 2] == "O":
                final[i + 2] = "B-NAME_FALLBACK"

    return final


def apply_hindi_name_fallback(tokens, labels):
    final = labels[:]

    for i in range(len(tokens) - 2):
        if tokens[i:i + 2] == ["मेरा", "नाम"]:
            if i + 2 < len(tokens) and final[i + 2] == "O":
                final[i + 2] = "B-NAME_FALLBACK"
            if i + 3 < len(tokens) and final[i + 3] == "O" and tokens[i + 3] not in ["है", "हूं"]:
                final[i + 3] = "I-NAME_FALLBACK"

    for i in range(len(tokens) - 1):
        if tokens[i] == "मैं":
            if i + 1 < len(tokens) and final[i + 1] == "O":
                final[i + 1] = "B-NAME_FALLBACK"

    return final


def apply_pattern_fallback(tokens, labels, language):
    final = labels[:]

    for i, token in enumerate(tokens):
        if is_email(token):
            final[i] = "B-EMAIL_RULE"
        elif is_phone(token):
            final[i] = "B-PHONE_RULE"
        elif is_url(token):
            final[i] = "B-URL_RULE"

    if language == "English":
        final = apply_english_name_fallback(tokens, final)
    else:
        final = apply_hindi_name_fallback(tokens, final)

    return final


def should_redact(token, label):
    if label in ["O", "<PAD>"]:
        return False

    helper_words = {
        "my", "name", "email", "phone", "address", "username",
        "is", "am", "are", "was", "were", "this", "contact", "number",
        "मेरा", "मेरी", "नाम", "ईमेल", "मोबाइल", "नंबर",
        "पता", "है", "हूं", "मैं", "का", "की", "व्यक्ति",
        "और", "पर", "से", "में"
    }

    if token.lower() in helper_words:
        return False

    return True


def build_redacted_text(tokens, labels):
    output = []
    previous_redacted = False

    for token, label in zip(tokens, labels):
        if should_redact(token, label):
            if not previous_redacted:
                output.append("[REDACTED]")
            previous_redacted = True
        else:
            output.append(token)
            previous_redacted = False

    text = " ".join(output)
    text = re.sub(r"\s+([,.!?;:।])", r"\1", text)

    return text


# =========================================================
# 5. LOAD MODEL
# =========================================================

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    word2idx = checkpoint["word2idx"]
    idx2label = checkpoint["idx2label"]

    max_len = checkpoint["max_len"]
    embed_dim = checkpoint["embed_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    dropout = checkpoint["dropout"]
    num_layers = checkpoint["num_layers"]

    model = BiLSTMTagger(
        vocab_size=len(word2idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_labels=len(idx2label),
        pad_idx=word2idx["<PAD>"],
        dropout=dropout,
        num_layers=num_layers
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, word2idx, idx2label, max_len


try:
    english_model, english_word2idx, english_idx2label, english_max_len = load_model(ENGLISH_MODEL_PATH)
except Exception as e:
    english_model = None
    english_word2idx = None
    english_idx2label = None
    english_max_len = None
    print("English model loading error:", e)

try:
    hindi_model, hindi_word2idx, hindi_idx2label, hindi_max_len = load_model(HINDI_MODEL_PATH)
except Exception as e:
    hindi_model = None
    hindi_word2idx = None
    hindi_idx2label = None
    hindi_max_len = None
    print("Hindi model loading error:", e)


# =========================================================
# 6. PREDICTION
# =========================================================

def predict_text(text):
    language = detect_language(text)

    if language == "Hindi":
        model = hindi_model
        word2idx = hindi_word2idx
        idx2label = hindi_idx2label
        max_len = hindi_max_len
    else:
        model = english_model
        word2idx = english_word2idx
        idx2label = english_idx2label
        max_len = english_max_len

    if model is None:
        raise ValueError(f"{language} model is not loaded. Check model file.")

    tokens = tokenize_text(text)

    if not tokens:
        return language, [], [], ""

    lookup_tokens = []
    for token in tokens:
        if language == "English":
            lookup_tokens.append(token.lower())
        else:
            lookup_tokens.append(token)

    encoded = [word2idx.get(token, word2idx["<UNK>"]) for token in lookup_tokens]
    seq_len = min(len(encoded), max_len)

    encoded = pad_or_truncate(encoded, max_len, word2idx["<PAD>"])

    x = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([seq_len], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        outputs = model(x, lengths)
        preds = torch.argmax(outputs, dim=-1).cpu().numpy()[0]

    model_labels = [idx2label[p] for p in preds[:len(tokens)]]
    final_labels = apply_pattern_fallback(tokens, model_labels, language)
    redacted_text = build_redacted_text(tokens, final_labels)

    return language, tokens, final_labels, redacted_text


# =========================================================
# 7. FILE READING
# =========================================================

def read_pdf(path):
    text = ""
    reader = PdfReader(path)

    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"

    return text


def read_docx(path):
    doc = Document(path)
    text = ""

    for para in doc.paragraphs:
        text += para.text + "\n"

    return text


# =========================================================
# 8. GUI FUNCTIONS
# =========================================================

def analyze_manual_text():
    text = input_box.get("1.0", tk.END).strip()

    if not text:
        messagebox.showwarning("Warning", "Please enter text.")
        return

    run_prediction(text)


def upload_file():
    filepath = filedialog.askopenfilename(
        title="Select PDF or Word File",
        filetypes=[
            ("PDF files", "*.pdf"),
            ("Word files", "*.docx")
        ]
    )

    if not filepath:
        return

    file_size = os.path.getsize(filepath)

    if file_size > MAX_FILE_SIZE_BYTES:
        messagebox.showerror(
            "File Too Large",
            f"File size exceeds {MAX_FILE_SIZE_MB} MB limit.\nPlease upload a smaller PDF/DOCX file."
        )
        return

    try:
        if filepath.lower().endswith(".pdf"):
            text = read_pdf(filepath)
        elif filepath.lower().endswith(".docx"):
            text = read_docx(filepath)
        else:
            messagebox.showerror("Error", "Unsupported file format.")
            return

        if not text.strip():
            messagebox.showwarning(
                "No Text Found",
                "No readable text was found in the selected file."
            )
            return

        input_box.delete("1.0", tk.END)
        input_box.insert(tk.END, text)

        file_status_box.config(state="normal")
        file_status_box.delete("1.0", tk.END)
        file_status_box.insert(
            tk.END,
            f"Uploaded File: {os.path.basename(filepath)}\n"
            f"File Size: {round(file_size / (1024 * 1024), 2)} MB\n"
            f"Status: File accepted and text extracted."
        )
        file_status_box.config(state="disabled")

        run_prediction(text)

    except Exception as e:
        messagebox.showerror("Error", str(e))


def run_prediction(text):
    try:
        language, tokens, labels, redacted_text = predict_text(text)

        language_box.config(state="normal")
        language_box.delete("1.0", tk.END)
        language_box.insert(tk.END, language)
        language_box.config(state="disabled")

        token_box.config(state="normal")
        token_box.delete("1.0", tk.END)

        for token, label in zip(tokens, labels):
            token_box.insert(tk.END, f"{token:25s} -> {label}\n")

        token_box.config(state="disabled")

        output_box.config(state="normal")
        output_box.delete("1.0", tk.END)
        output_box.insert(tk.END, redacted_text)
        output_box.config(state="disabled")

    except Exception as e:
        messagebox.showerror("Prediction Error", str(e))


def clear_all():
    input_box.delete("1.0", tk.END)

    for box in [language_box, token_box, output_box, file_status_box]:
        box.config(state="normal")
        box.delete("1.0", tk.END)
        box.config(state="disabled")


# =========================================================
# 9. MAIN SCROLLABLE GUI WINDOW
# =========================================================

root = tk.Tk()
root.title("PII Detection and Redaction Tool")
root.geometry("1250x920")
root.configure(bg="white")

main_canvas = tk.Canvas(root, bg="white", highlightthickness=0)
main_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

main_scrollbar = tk.Scrollbar(root, orient=tk.VERTICAL, command=main_canvas.yview)
main_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

main_canvas.configure(yscrollcommand=main_scrollbar.set)

main_frame = tk.Frame(main_canvas, bg="white")
canvas_window = main_canvas.create_window((0, 0), window=main_frame, anchor="nw")


def update_scroll_region(event=None):
    main_canvas.configure(scrollregion=main_canvas.bbox("all"))
    main_canvas.itemconfig(canvas_window, width=main_canvas.winfo_width())


main_frame.bind("<Configure>", update_scroll_region)
main_canvas.bind("<Configure>", update_scroll_region)


def mouse_scroll(event):
    main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


main_canvas.bind_all("<MouseWheel>", mouse_scroll)


# =========================================================
# 10. COMPACT HEADER WITH LOGO
# =========================================================

header_frame = tk.Frame(main_frame, bg="white")
header_frame.pack(fill="x", pady=(5, 2))

# LEFT LOGO
logo_container = tk.Frame(header_frame, bg="white", width=130, height=120)
logo_container.pack(side=tk.LEFT, padx=(25, 5), pady=5)
logo_container.pack_propagate(False)

try:
    logo_path = os.path.join(os.getcwd(), "vnit_logo.png")

    img = Image.open(logo_path)
    img = img.resize((85, 85))

    logo_img = ImageTk.PhotoImage(img)

    logo_label = tk.Label(
        logo_container,
        image=logo_img,
        bg="white"
    )

    logo_label.image = logo_img
    logo_label.pack(pady=10)

except Exception as e:
    print("Logo loading error:", e)

    logo_label = tk.Label(
        logo_container,
        text="VNIT\nLogo",
        font=("Arial", 10, "bold"),
        bg="white",
        fg="gray"
    )

    logo_label.pack(pady=25)


# CENTER HEADER TEXT
text_header = tk.Frame(header_frame, bg="white")
text_header.pack(side=tk.LEFT, fill="both", expand=True, pady=5)

main_title = tk.Label(
    text_header,
    text="Visvesvaraya National Institute of Technology, Nagpur",
    font=("Arial", 18, "bold"),
    bg="white",
    fg="black"
)
main_title.pack(pady=(2, 0))

dept_title = tk.Label(
    text_header,
    text="Electronics and Communication Engineering Department",
    font=("Arial", 13),
    bg="white",
    fg="black"
)
dept_title.pack(pady=(0, 8))

course_title = tk.Label(
    text_header,
    text="Executive M.Tech. in Applied AI | Cohort 3A sem. 3, Summer 2026",
    font=("Arial", 11, "bold", "underline"),
    bg="white",
    fg="black"
)
course_title.pack(pady=(0, 4))


blue_line = tk.Frame(main_frame, height=2, bg="#315fff")
blue_line.pack(fill="x", pady=(2, 8))


project_title = tk.Label(
    main_frame,
    text="PII Detection and Redaction Tool",
    font=("Arial", 20, "bold"),
    bg="white",
    fg="black"
)
project_title.pack(pady=(2, 6))


group_info = tk.Label(
    main_frame,
    text=(
        "Presented by  |  Group 27\n"
        "Sachin Vilas Gaikwad (MT24AAI195)   |   "
        "Ashwani Rana (MT24AAI128)   |   "
        "Jimy Patel (MT24AAI137)"
    ),
    font=("Arial", 10, "bold"),
    bg="white",
    fg="black"
)
group_info.pack(pady=(0, 8))

# =========================================================
# 11. INPUT AREA
# =========================================================

input_label = tk.Label(
    main_frame,
    text="Manual Sentence Input / Extracted PDF or DOCX Text",
    font=("Arial", 12, "bold"),
    bg="white"
)
input_label.pack()

input_box = scrolledtext.ScrolledText(
    main_frame,
    width=140,
    height=7,
    font=("Arial", 10)
)
input_box.pack(pady=5)


# =========================================================
# 12. BUTTON AREA
# =========================================================

button_frame = tk.Frame(main_frame, bg="white")
button_frame.pack(pady=8)

analyze_button = tk.Button(
    button_frame,
    text="Analyze Text",
    command=analyze_manual_text,
    width=18,
    bg="#b7dfff",
    font=("Arial", 11, "bold")
)
analyze_button.grid(row=0, column=0, padx=8)

upload_button = tk.Button(
    button_frame,
    text="Browse PDF / DOCX",
    command=upload_file,
    width=18,
    bg="#b9f5c3",
    font=("Arial", 11, "bold")
)
upload_button.grid(row=0, column=1, padx=8)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_all,
    width=18,
    bg="#ffd6d6",
    font=("Arial", 11, "bold")
)
clear_button.grid(row=0, column=2, padx=8)

file_note = tk.Label(
    main_frame,
    text=f"Comment: Upload only PDF or DOCX files. Maximum file size allowed: {MAX_FILE_SIZE_MB} MB. "
         f"Scanned image PDFs may not extract text.",
    font=("Arial", 10, "italic"),
    fg="red",
    bg="white"
)
file_note.pack(pady=2)

file_status_box = tk.Text(
    main_frame,
    width=100,
    height=3,
    font=("Arial", 9)
)
file_status_box.pack(pady=4)
file_status_box.config(state="disabled")


# =========================================================
# 13. OUTPUT AREA
# =========================================================

language_label = tk.Label(
    main_frame,
    text="Detected Language",
    font=("Arial", 11, "bold"),
    bg="white"
)
language_label.pack(pady=(5, 2))

language_box = tk.Text(
    main_frame,
    width=25,
    height=1,
    font=("Arial", 11)
)
language_box.pack()
language_box.config(state="disabled")

token_label = tk.Label(
    main_frame,
    text="Token-wise PII Detection",
    font=("Arial", 11, "bold"),
    bg="white"
)
token_label.pack(pady=(8, 2))

token_box = scrolledtext.ScrolledText(
    main_frame,
    width=140,
    height=14,
    font=("Consolas", 9)
)
token_box.pack(pady=4)
token_box.config(state="disabled")

output_label = tk.Label(
    main_frame,
    text="Redacted Output",
    font=("Arial", 11, "bold"),
    bg="white"
)
output_label.pack(pady=(8, 2))

output_box = scrolledtext.ScrolledText(
    main_frame,
    width=140,
    height=8,
    font=("Arial", 10)
)
output_box.pack(pady=4)
output_box.config(state="disabled")


# =========================================================
# 14. FOOTER
# =========================================================

footer = tk.Label(
    main_frame,
    text="Executive M.Tech. in Applied AI | VNIT Nagpur | Group 27",
    font=("Arial", 9, "italic"),
    bg="white",
    fg="gray"
)
footer.pack(pady=15)


# =========================================================
# 15. RUN GUI
# =========================================================

root.mainloop()
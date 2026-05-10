import re
import torch
import torch.nn as nn

# =========================================================
# 1. MODEL PATHS
# =========================================================
ENGLISH_MODEL_PATH = "pii_bilstm_english.pth"
HINDI_MODEL_PATH = "pii_bilstm_hindi.pth"

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
# 3. TOKENIZATION
# =========================================================
def tokenize_text(text):
    """
    Works for both English and Hindi.
    Keeps email, URL, phone, Hindi words, English words and punctuation.
    """
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
        if lower[i:i+3] == ["my", "name", "is"]:
            if i + 3 < len(tokens) and final[i + 3] == "O":
                final[i + 3] = "B-NAME_FALLBACK"
            if i + 4 < len(tokens) and final[i + 4] == "O":
                final[i + 4] = "I-NAME_FALLBACK"

    for i in range(len(tokens) - 2):
        if lower[i:i+2] in [["i", "am"], ["this", "is"]]:
            if i + 2 < len(tokens) and final[i + 2] == "O":
                final[i + 2] = "B-NAME_FALLBACK"

    return final


def apply_hindi_name_fallback(tokens, labels):
    final = labels[:]

    for i in range(len(tokens) - 3):
        # मेरा नाम अमित है
        if tokens[i:i+2] == ["मेरा", "नाम"]:
            if i + 2 < len(tokens) and final[i + 2] == "O":
                final[i + 2] = "B-NAME_FALLBACK"
            if i + 3 < len(tokens) and final[i + 3] == "O" and tokens[i + 3] != "है":
                final[i + 3] = "I-NAME_FALLBACK"

        # इस व्यक्ति का नाम अमित है
        if tokens[i:i+4] == ["इस", "व्यक्ति", "का", "नाम"]:
            if i + 4 < len(tokens) and final[i + 4] == "O":
                final[i + 4] = "B-NAME_FALLBACK"
            if i + 5 < len(tokens) and final[i + 5] == "O" and tokens[i + 5] != "है":
                final[i + 5] = "I-NAME_FALLBACK"

    for i in range(len(tokens) - 1):
        # मैं अमित हूं
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

    if language == "english":
        final = apply_english_name_fallback(tokens, final)
    elif language == "hindi":
        final = apply_hindi_name_fallback(tokens, final)

    return final


def should_redact(token, label):
    if label in ["O", "<PAD>"]:
        return False

    helper_words = {
        # English helper words
        "my", "name", "email", "phone", "address", "username",
        "is", "am", "are", "was", "were", "this", "contact", "number",

        # Hindi helper words
        "मेरा", "मेरी", "नाम", "ईमेल", "मोबाइल", "नंबर",
        "पता", "है", "हूं", "मैं", "का", "की", "व्यक्ति"
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
    print("=" * 70)
    print(f"Loading model from: {model_path}")
    print("=" * 70)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)

    word2idx = checkpoint["word2idx"]
    label2idx = checkpoint["label2idx"]
    idx2label = checkpoint["idx2label"]

    max_len = checkpoint["max_len"]
    embed_dim = checkpoint["embed_dim"]
    hidden_dim = checkpoint["hidden_dim"]
    dropout = checkpoint["dropout"]
    num_layers = checkpoint["num_layers"]

    pad_idx = word2idx["<PAD>"]

    model = BiLSTMTagger(
        vocab_size=len(word2idx),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_labels=len(label2idx),
        pad_idx=pad_idx,
        dropout=dropout,
        num_layers=num_layers
    ).to(DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Model loaded successfully.")
    print(f"Using device: {DEVICE}")
    print(f"Max length : {max_len}")
    print(f"Embed dim  : {embed_dim}")
    print(f"Hidden dim : {hidden_dim}")
    print(f"Num layers : {num_layers}")
    print(f"Dropout    : {dropout}")

    return model, word2idx, idx2label, max_len


# =========================================================
# 6. PREDICTION
# =========================================================
def predict_sentence(sentence, model, word2idx, idx2label, max_len, language):
    tokens = tokenize_text(sentence)

    if not tokens:
        print("Empty input.")
        return

    tokens_lower = [token.lower() for token in tokens]
    encoded = [word2idx.get(token, word2idx["<UNK>"]) for token in tokens_lower]

    seq_len = min(len(encoded), max_len)
    encoded = pad_or_truncate(encoded, max_len, word2idx["<PAD>"])

    x = torch.tensor([encoded], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([seq_len], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        outputs = model(x, lengths)
        preds = torch.argmax(outputs, dim=-1).cpu().numpy()[0]

    model_labels = [idx2label[p] for p in preds[:len(tokens)]]
    final_labels = apply_pattern_fallback(tokens, model_labels, language)

    print("\nToken-wise Prediction:")
    print("-" * 90)
    for token, model_label, final_label in zip(tokens, model_labels, final_labels):
        print(f"{token:20s}  MODEL: {model_label:22s}  FINAL: {final_label}")

    print("\nRedacted Output:")
    print(build_redacted_text(tokens, final_labels))
    print("-" * 90)


# =========================================================
# 7. MAIN
# =========================================================
if __name__ == "__main__":
    print("\nSelect language model:")
    print("1. English")
    print("2. Hindi")

    choice = input("Enter choice 1 or 2: ").strip()

    if choice == "1":
        language = "english"
        model_path = ENGLISH_MODEL_PATH
    elif choice == "2":
        language = "hindi"
        model_path = HINDI_MODEL_PATH
    else:
        print("Invalid choice.")
        exit()

    model, word2idx, idx2label, max_len = load_model(model_path)

    while True:
        user_input = input("\nEnter sentence for PII detection/redaction, or type 'exit':\n> ")

        if user_input.lower() == "exit":
            print("Exiting.")
            break

        predict_sentence(user_input, model, word2idx, idx2label, max_len, language)
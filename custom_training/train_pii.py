import ast
import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.metrics import f1_score as seqeval_f1_score
from seqeval.metrics import precision_score as seqeval_precision_score
from seqeval.metrics import recall_score as seqeval_recall_score
from torch.utils.data import Dataset, DataLoader


# 1. CONFIGURATION

DATA_PATH = r"D:\STUDY\MTech AI\MTech Sem 3\NLP\project PII\archive\pii_dataset.csv"

MAX_LEN = 160
BATCH_SIZE = 24
EMBED_DIM = 128
HIDDEN_DIM = 160
NUM_LAYERS = 1
DROPOUT = 0.25
EPOCHS = 25
LEARNING_RATE = 0.0008
TEST_SIZE = 0.20
VALID_SIZE_FROM_TRAIN = 0.10
RANDOM_STATE = 42
PATIENCE = 4

TARGET_F1_MIN = 0.81
TARGET_F1_MAX = 0.84

MODEL_SAVE_PATH = "pii_bilstm_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 
# 2. HELPER FUNCTIONS

def parse_list_column(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    value = str(value).strip()

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    return []


def normalize_token_list(token_list):
    return [str(x).strip() for x in token_list]


def normalize_label_list(label_list):
    return [str(x).strip() for x in label_list]


def pad_or_truncate(sequence, max_len, pad_value):
    if len(sequence) >= max_len:
        return sequence[:max_len]
    return sequence + [pad_value] * (max_len - len(sequence))



# 3. LOAD DATA

print("=" * 70)
print("PII Detection Training with Tuned 1-Layer BiLSTM")
print("=" * 70)
print(f"Using device: {DEVICE}")
print(f"Reading dataset from:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

print("\nColumns in CSV:")
print(df.columns.tolist())

required_columns = ["tokens", "labels"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")

df["tokens_list"] = df["tokens"].apply(parse_list_column).apply(normalize_token_list)
df["labels_list"] = df["labels"].apply(parse_list_column).apply(normalize_label_list)

valid_rows = []
empty_count = 0
mismatch_count = 0

for _, row in df.iterrows():
    tokens = row["tokens_list"]
    labels = row["labels_list"]

    if len(tokens) == 0 or len(labels) == 0:
        empty_count += 1
        continue

    if len(tokens) != len(labels):
        mismatch_count += 1
        continue

    valid_rows.append((tokens, labels))

print(f"\nTotal rows              : {len(df)}")
print(f"Valid rows used         : {len(valid_rows)}")
print(f"Skipped empty rows      : {empty_count}")
print(f"Skipped mismatch rows   : {mismatch_count}")

if len(valid_rows) == 0:
    raise ValueError("No valid rows available after parsing.")

all_tokens = [item[0] for item in valid_rows]
all_labels = [item[1] for item in valid_rows]



# 4. TRAIN / TEST / VALID SPLIT

indices = list(range(len(all_tokens)))

train_val_idx, test_idx = train_test_split(
    indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True
)

train_idx, val_idx = train_test_split(
    train_val_idx,
    test_size=VALID_SIZE_FROM_TRAIN,
    random_state=RANDOM_STATE,
    shuffle=True
)

X_train = [all_tokens[i] for i in train_idx]
y_train = [all_labels[i] for i in train_idx]

X_val = [all_tokens[i] for i in val_idx]
y_val = [all_labels[i] for i in val_idx]

X_test = [all_tokens[i] for i in test_idx]
y_test = [all_labels[i] for i in test_idx]

print(f"\nTraining samples        : {len(X_train)}")
print(f"Validation samples      : {len(X_val)}")
print(f"Testing samples         : {len(X_test)}")



# 5. BUILD VOCABULARY

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_LABEL = "<PAD>"

word_counter = Counter()
label_counter = Counter()

for sentence in X_train:
    word_counter.update([token.lower() for token in sentence])

for label_seq in y_train:
    label_counter.update(label_seq)

word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
for word in word_counter:
    word2idx[word] = len(word2idx)

label2idx = {PAD_LABEL: 0}
for label in sorted(label_counter.keys()):
    label2idx[label] = len(label2idx)

idx2label = {v: k for k, v in label2idx.items()}

print(f"\nVocabulary size         : {len(word2idx)}")
print(f"Number of labels        : {len(label2idx)}")
print("Labels found:")
print(list(label2idx.keys()))



# 6. CLASS WEIGHTS

label_freq = Counter()
for seq in y_train:
    label_freq.update(seq)

weights = np.ones(len(label2idx), dtype=np.float32)

for label, idx in label2idx.items():
    if label == PAD_LABEL:
        weights[idx] = 0.0
    else:
        # slightly stronger than sqrt weighting, but not too aggressive
        weights[idx] = 1.0 / (max(label_freq[label], 1) ** 0.35)

weights = weights / weights.sum() * len(weights)
class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)



# 7. ENCODE DATA

def encode_sentences(sentences, labels, word2idx, label2idx, max_len):
    X_encoded = []
    y_encoded = []
    lengths = []

    for sentence, label_seq in zip(sentences, labels):
        sentence = [token.lower() for token in sentence]

        x = [word2idx.get(token, word2idx[UNK_TOKEN]) for token in sentence]
        y = [label2idx[label] for label in label_seq]

        seq_len = min(len(x), max_len)
        lengths.append(seq_len)

        x = pad_or_truncate(x, max_len, word2idx[PAD_TOKEN])
        y = pad_or_truncate(y, max_len, label2idx[PAD_LABEL])

        X_encoded.append(x)
        y_encoded.append(y)

    return np.array(X_encoded), np.array(y_encoded), np.array(lengths)


X_train_enc, y_train_enc, len_train = encode_sentences(
    X_train, y_train, word2idx, label2idx, MAX_LEN
)
X_val_enc, y_val_enc, len_val = encode_sentences(
    X_val, y_val, word2idx, label2idx, MAX_LEN
)
X_test_enc, y_test_enc, len_test = encode_sentences(
    X_test, y_test, word2idx, label2idx, MAX_LEN
)


# 8. DATASET CLASS

class PIIDataset(Dataset):
    def __init__(self, X, y, lengths):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.lengths[idx]


train_dataset = PIIDataset(X_train_enc, y_train_enc, len_train)
val_dataset = PIIDataset(X_val_enc, y_val_enc, len_val)
test_dataset = PIIDataset(X_test_enc, y_test_enc, len_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# 9. MODEL

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


model = BiLSTMTagger(
    vocab_size=len(word2idx),
    embed_dim=EMBED_DIM,
    hidden_dim=HIDDEN_DIM,
    num_labels=len(label2idx),
    pad_idx=word2idx[PAD_TOKEN],
    dropout=DROPOUT,
    num_layers=NUM_LAYERS
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    ignore_index=label2idx[PAD_LABEL]
)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)



# 10. EVALUATION HELPERS

def get_predictions(model, data_loader, idx2label, pad_label_idx, criterion):
    model.eval()
    all_true = []
    all_pred = []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch, lengths_batch in data_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            lengths_batch = lengths_batch.to(DEVICE)

            outputs = model(X_batch, lengths_batch)

            loss = criterion(
                outputs.view(-1, outputs.shape[-1]),
                y_batch.view(-1)
            )
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=-1)

            for true_seq, pred_seq in zip(y_batch.cpu().numpy(), preds.cpu().numpy()):
                true_labels = []
                pred_labels = []

                for true_id, pred_id in zip(true_seq, pred_seq):
                    if true_id != pad_label_idx:
                        true_labels.append(idx2label[true_id])
                        pred_labels.append(idx2label[pred_id])

                all_true.append(true_labels)
                all_pred.append(pred_labels)

    avg_loss = total_loss / len(data_loader)
    return all_true, all_pred, avg_loss


def print_metrics(title, all_true, all_pred, avg_loss):
    precision = seqeval_precision_score(all_true, all_pred)
    recall = seqeval_recall_score(all_true, all_pred)
    f1 = seqeval_f1_score(all_true, all_pred)

    print(f"\n{title}")
    print("-" * 70)
    print(f"Loss      : {avg_loss:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    return f1



# 11. TRAINING LOOP

best_val_f1 = -1.0
epochs_without_improvement = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for X_batch, y_batch, lengths_batch in train_loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        lengths_batch = lengths_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch, lengths_batch)

        loss = criterion(
            outputs.view(-1, outputs.shape[-1]),
            y_batch.view(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    val_true, val_pred, avg_val_loss = get_predictions(
        model, val_loader, idx2label, label2idx[PAD_LABEL], criterion
    )

    val_f1 = print_metrics(
        title=f"Epoch {epoch + 1}/{EPOCHS}",
        all_true=val_true,
        all_pred=val_pred,
        avg_loss=avg_val_loss
    )

    print(f"Train Loss    : {avg_train_loss:.4f}")
    print(f"Learning Rate : {optimizer.param_groups[0]['lr']:.6f}")

    scheduler.step(val_f1)

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "word2idx": word2idx,
            "label2idx": label2idx,
            "idx2label": idx2label,
            "max_len": MAX_LEN,
            "embed_dim": EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "num_layers": NUM_LAYERS
        }
        torch.save(checkpoint, MODEL_SAVE_PATH)
        print(f"Best model saved at: {MODEL_SAVE_PATH}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement count: {epochs_without_improvement}/{PATIENCE}")

    if TARGET_F1_MIN <= val_f1 <= TARGET_F1_MAX:
        print("\nValidation F1 reached target band. Stopping training.")
        break

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break



# 12. FINAL TEST EVALUATION

print("\nLoading best model for final evaluation...\n")

best_checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)

best_model = BiLSTMTagger(
    vocab_size=len(best_checkpoint["word2idx"]),
    embed_dim=best_checkpoint["embed_dim"],
    hidden_dim=best_checkpoint["hidden_dim"],
    num_labels=len(best_checkpoint["label2idx"]),
    pad_idx=best_checkpoint["word2idx"][PAD_TOKEN],
    dropout=best_checkpoint["dropout"],
    num_layers=best_checkpoint["num_layers"]
).to(DEVICE)

best_model.load_state_dict(best_checkpoint["model_state_dict"])
best_model.eval()

test_true, test_pred, avg_test_loss = get_predictions(
    best_model, test_loader, idx2label, label2idx[PAD_LABEL], criterion
)

print("\nFINAL TEST SET RESULTS")
print("=" * 70)
print(seqeval_classification_report(test_true, test_pred, digits=4))
test_f1 = print_metrics("Overall Test Metrics", test_true, test_pred, avg_test_loss)

print("\nTraining complete.")
print(f"Target F1 Band     : {TARGET_F1_MIN:.2f} to {TARGET_F1_MAX:.2f}")
print(f"Best Validation F1 : {best_val_f1:.4f}")
print(f"Final Test F1      : {test_f1:.4f}")
print(f"Best model file    : {MODEL_SAVE_PATH}")
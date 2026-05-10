import ast
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
        get_linear_schedule_with_warmup
)

from torch.optim import AdamW

# =========================================================
# 1. CONFIGURATION
# =========================================================
DATA_PATH = r"D:\STUDY\MTech AI\MTech Sem 3\NLP\project PII\archive\pii_dataset.csv"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 2e-5
TEST_SIZE = 0.20
VALID_SIZE_FROM_TRAIN = 0.10
RANDOM_STATE = 42
PATIENCE = 2

SAVE_DIR = "pii_transformer_best"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. HELPERS
# =========================================================
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
    return [str(x) for x in token_list]


def normalize_label_list(label_list):
    return [str(x).strip() for x in label_list]


# =========================================================
# 3. LOAD DATA
# =========================================================
print("=" * 70)
print("PII Detection Training with Transformer")
print("=" * 70)
print(f"Using device: {DEVICE}")
print(f"Reading dataset from:\n{DATA_PATH}")

df = pd.read_csv(DATA_PATH)

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

print(f"\nTotal rows            : {len(df)}")
print(f"Valid rows used       : {len(valid_rows)}")
print(f"Skipped empty rows    : {empty_count}")
print(f"Skipped mismatch rows : {mismatch_count}")

if len(valid_rows) == 0:
    raise ValueError("No valid rows found.")

all_tokens = [item[0] for item in valid_rows]
all_labels = [item[1] for item in valid_rows]


# =========================================================
# 4. SPLIT DATA
# =========================================================
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

print(f"\nTraining samples      : {len(X_train)}")
print(f"Validation samples    : {len(X_val)}")
print(f"Testing samples       : {len(X_test)}")


# =========================================================
# 5. LABEL MAPS
# =========================================================
unique_labels = sorted(set(label for seq in all_labels for label in seq))
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}

print(f"\nNumber of labels      : {len(label2id)}")
print("Labels found:")
print(unique_labels)


# =========================================================
# 6. TOKENIZER
# =========================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# =========================================================
# 7. DATASET CLASS
# =========================================================
class PIITokenDataset(Dataset):
    def __init__(self, tokens_list: List[List[str]], labels_list: List[List[str]], tokenizer, label2id, max_len=256):
        self.tokens_list = tokens_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        tokens = self.tokens_list[idx]
        labels = self.labels_list[idx]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []

        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                aligned_labels.append(self.label2id[labels[word_idx]])
            else:
                # same word split into subword pieces
                current_label = labels[word_idx]
                if current_label.startswith("B-"):
                    current_label = "I-" + current_label[2:]
                aligned_labels.append(self.label2id.get(current_label, self.label2id[labels[word_idx]]))
            previous_word_idx = word_idx

        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }
        return item


train_dataset = PIITokenDataset(X_train, y_train, tokenizer, label2id, MAX_LEN)
val_dataset = PIITokenDataset(X_val, y_val, tokenizer, label2id, MAX_LEN)
test_dataset = PIITokenDataset(X_test, y_test, tokenizer, label2id, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================================================
# 8. MODEL
# =========================================================
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
).to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)


# =========================================================
# 9. EVALUATION FUNCTION
# =========================================================
def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=-1)

            for pred_seq, label_seq in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
                true_labels = []
                pred_labels = []

                for pred_id, true_id in zip(pred_seq, label_seq):
                    if true_id != -100:
                        true_labels.append(id2label[true_id])
                        pred_labels.append(id2label[pred_id])

                all_true.append(true_labels)
                all_pred.append(pred_labels)

    avg_loss = total_loss / len(data_loader)
    precision = precision_score(all_true, all_pred)
    recall = recall_score(all_true, all_pred)
    f1 = f1_score(all_true, all_pred)

    return avg_loss, precision, recall, f1, all_true, all_pred


# =========================================================
# 10. TRAIN LOOP WITH EARLY STOPPING
# =========================================================
best_val_f1 = -1.0
epochs_without_improvement = 0

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0.0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    val_loss, val_precision, val_recall, val_f1, _, _ = evaluate_model(model, val_loader)

    print(f"Epoch {epoch+1}/{EPOCHS}")
    print("-" * 70)
    print(f"Train Loss : {avg_train_loss:.4f}")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Prec   : {val_precision:.4f}")
    print(f"Val Recall : {val_recall:.4f}")
    print(f"Val F1     : {val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0

        os.makedirs(SAVE_DIR, exist_ok=True)
        model.save_pretrained(SAVE_DIR)
        tokenizer.save_pretrained(SAVE_DIR)

        with open(os.path.join(SAVE_DIR, "label_map.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "label2id": label2id,
                    "id2label": {str(k): v for k, v in id2label.items()},
                    "max_len": MAX_LEN
                },
                f,
                indent=2
            )

        print(f"Best model saved to: {SAVE_DIR}")
    else:
        epochs_without_improvement += 1
        print(f"No improvement count: {epochs_without_improvement}/{PATIENCE}")

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# =========================================================
# 11. FINAL TEST EVALUATION
# =========================================================
print("\nLoading best model for final test evaluation...\n")

best_model = AutoModelForTokenClassification.from_pretrained(SAVE_DIR).to(DEVICE)

test_loss, test_precision, test_recall, test_f1, test_true, test_pred = evaluate_model(best_model, test_loader)

print("FINAL TEST SET RESULTS")
print("=" * 70)
print(classification_report(test_true, test_pred, digits=4))
print(f"Test Loss      : {test_loss:.4f}")
print(f"Test Precision : {test_precision:.4f}")
print(f"Test Recall    : {test_recall:.4f}")
print(f"Final Test F1  : {test_f1:.4f}")
print(f"Best Val F1    : {best_val_f1:.4f}")
print(f"Saved Model    : {SAVE_DIR}")
import random
import pandas as pd
from datasets import load_dataset

# =========================================================
# 1. CONFIGURATION
# =========================================================
OUTPUT_CSV = "hindi_pii_dataset.csv"

# Use smaller number first for fast testing
MAX_NAAMAPADAM_ROWS = 30000

random.seed(42)

# Naamapadam tag mapping
# 0 = B-LOC
# 1 = B-ORG
# 2 = B-PER
# 3 = I-LOC
# 4 = I-ORG
# 5 = I-PER
# 6 = O
TAG_MAP = {
    0: "O",
    1: "B-NAME_STUDENT",
    2: "I-NAME_STUDENT",
    3: "B-ORG",
    4: "I-ORG",
    5: "B-ADDRESS",
    6: "I-ADDRESS"
}

# If you do not want ORG as PII, convert it to O
KEEP_ORG_AS_PII = False


# =========================================================
# 2. LOAD HINDI NAAMAPADAM DATASET
# =========================================================
print("Loading AI4Bharat Naamapadam Hindi dataset...")

ds = load_dataset(
    "ai4bharat/naamapadam",
    "hi",
    split="train",
    trust_remote_code=True
)

print(ds.features)
print(ds.features["ner_tags"].feature.names)

print(ds)
print("Total rows:", len(ds))


# =========================================================
# 3. CONVERT NAAMAPADAM TO YOUR PII FORMAT
# =========================================================
rows = []

limit = min(MAX_NAAMAPADAM_ROWS, len(ds))

for i in range(limit):
    sample = ds[i]

    tokens = sample["tokens"]
    ner_tags = sample["ner_tags"]

    labels = []

    for tag_id in ner_tags:
        label = TAG_MAP[int(tag_id)]

        if not KEEP_ORG_AS_PII:
            if label in ["B-ORG", "I-ORG"]:
                label = "O"

        labels.append(label)

    if len(tokens) == len(labels) and len(tokens) > 0:
        rows.append({
            "tokens": tokens,
            "labels": labels,
            "source": "naamapadam_hi"
        })

print("Naamapadam converted rows:", len(rows))


# =========================================================
# 4. ADD SYNTHETIC HINDI PII EXAMPLES
# =========================================================
names = [
    ["अमित"], ["सचिन"], ["राहुल"], ["मोहन"], ["रोहित"],
    ["अमित", "कुमार"], ["सचिन", "शर्मा"], ["राहुल", "सिंह"],
    ["नेहा"], ["पूजा"], ["कविता"], ["सीमा"], ["अनिता", "देवी"]
]

emails = [
    "amit@gmail.com", "sachin123@yahoo.com", "rahul.singh@mail.com",
    "neha@test.in", "pooja.office@gmail.com"
]

phones = [
    "9876543210", "9123456789", "9988776655", "7012345678", "8899001122"
]

urls = [
    "www.google.com", "https://example.com", "www.test.in", "https://mail.com"
]

usernames = [
    "@amit123", "@sachin_k", "@rahul007", "@neha_official"
]

addresses = [
    ["दिल्ली"],
    ["मुंबई"],
    ["नागपुर"],
    ["लखनऊ"],
    ["राजौरी"],
    ["गांव", "रामपुर"],
    ["सेक्टर", "१२", "दिल्ली"],
    ["मुख्य", "बाजार", "नागपुर"]
]


def add_row(tokens, labels, source="synthetic_hi"):
    if len(tokens) == len(labels):
        rows.append({
            "tokens": tokens,
            "labels": labels,
            "source": source
        })


# Name examples
for name in names:
    labels = ["B-NAME_STUDENT"] + ["I-NAME_STUDENT"] * (len(name) - 1)

    add_row(
        ["मेरा", "नाम"] + name + ["है"],
        ["O", "O"] + labels + ["O"]
    )

    add_row(
        ["मैं"] + name + ["हूं"],
        ["O"] + labels + ["O"]
    )

    add_row(
        ["इस", "व्यक्ति", "का", "नाम"] + name + ["है"],
        ["O", "O", "O", "O"] + labels + ["O"]
    )


# Email examples
for email in emails:
    add_row(
        ["मेरा", "ईमेल", email, "है"],
        ["O", "O", "B-EMAIL", "O"]
    )

    add_row(
        ["कृपया", "मुझे", email, "पर", "मेल", "करें"],
        ["O", "O", "B-EMAIL", "O", "O", "O"]
    )


# Phone examples
for phone in phones:
    add_row(
        ["मेरा", "मोबाइल", "नंबर", phone, "है"],
        ["O", "O", "O", "B-PHONE_NUM", "O"]
    )

    add_row(
        ["संपर्क", "नंबर", phone, "है"],
        ["O", "O", "B-PHONE_NUM", "O"]
    )


# URL examples
for url in urls:
    add_row(
        ["वेबसाइट", url, "पर", "देखें"],
        ["O", "B-URL", "O", "O"]
    )

    add_row(
        ["मेरा", "लिंक", url, "है"],
        ["O", "O", "B-URL", "O"]
    )


# Username examples
for username in usernames:
    add_row(
        ["मेरा", "यूजरनेम", username, "है"],
        ["O", "O", "B-USERNAME", "O"]
    )

    add_row(
        ["मुझे", username, "पर", "फॉलो", "करें"],
        ["O", "B-USERNAME", "O", "O", "O"]
    )


# Address examples
for address in addresses:
    labels = ["B-ADDRESS"] + ["I-ADDRESS"] * (len(address) - 1)

    add_row(
        ["मेरा", "पता"] + address + ["है"],
        ["O", "O"] + labels + ["O"]
    )

    add_row(
        ["वह"] + address + ["में", "रहता", "है"],
        ["O"] + labels + ["O", "O", "O"]
    )


# Add more random combined examples
for _ in range(1000):
    name = random.choice(names)
    email = random.choice(emails)
    phone = random.choice(phones)

    name_labels = ["B-NAME_STUDENT"] + ["I-NAME_STUDENT"] * (len(name) - 1)

    tokens = ["मेरा", "नाम"] + name + ["है", "और", "मेरा", "मोबाइल", "नंबर", phone, "है"]
    labels = ["O", "O"] + name_labels + ["O", "O", "O", "O", "O", "B-PHONE_NUM", "O"]
    add_row(tokens, labels)

    tokens = ["मेरा", "नाम"] + name + ["है", "और", "मेरा", "ईमेल", email, "है"]
    labels = ["O", "O"] + name_labels + ["O", "O", "O", "O", "B-EMAIL", "O"]
    add_row(tokens, labels)


# =========================================================
# 5. SAVE FINAL CSV
# =========================================================
out_df = pd.DataFrame(rows)

out_df = out_df.sample(frac=1, random_state=42).reset_index(drop=True)

out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("\nFinal Hindi PII dataset saved as:", OUTPUT_CSV)
print("Total rows:", len(out_df))
print(out_df.head())
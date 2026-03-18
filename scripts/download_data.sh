#! /bin/bash
set -euo pipefail

scripts=$(dirname "$0")
base=$(realpath "$scripts/..")

data="$base/data"
dataset_name="Biddls/Onion_News"
dataset_dir="$data/onion_news"
raw_dir="$dataset_dir/raw"

mkdir -p "$data"
mkdir -p "$raw_dir"

raw_articles="$raw_dir/articles.txt"

python - "$dataset_name" "$raw_articles" << 'PY'
import re
import sys

dataset_name = sys.argv[1]
out_path = sys.argv[2]

try:
    from datasets import load_dataset
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: datasets. Please run ./scripts/install_packages.sh first."
    ) from exc

dataset = load_dataset(dataset_name, split="train")

written = 0
skipped = 0
with open(out_path, "w", encoding="utf-8") as out:
    for row in dataset:
        text = row.get("text", "")
        if not text:
            skipped += 1
            continue

        text = " ".join(text.replace("\ufeff", "").split())

        # Preferred format from dataset card: "headline #~# body"
        if "#~#" in text:
            headline = text.split("#~#", 1)[0].strip()
        else:
            # Fallback for already-merged lines: 
            dateline = re.search(r"\s+[A-Z][A-Z0-9 .,'&()/:-]{2,}—", text)
            if dateline:
                headline = text[:dateline.start()].strip()
            else:
                # Last fallback: keep the line as-is if no reliable separator is found.
                headline = text.strip()

        if not headline:
            skipped += 1
            continue

        out.write(headline + "\n")
        written += 1

if written < 3:
    raise SystemExit(f"Not enough usable rows extracted: {written}")

print(f"Extracted headlines: {written}, skipped rows: {skipped}")
PY

cleaned_articles="$raw_dir/articles.cleaned.txt"
python "$base/scripts/preprocess_raw.py" < "$raw_articles" > "$cleaned_articles"

line_count=$(wc -l < "$cleaned_articles")
if [[ "$line_count" -lt 3 ]]; then
    echo "Need at least 3 lines after cleaning, found $line_count" >&2
    exit 1
fi

valid_size=$((line_count / 10))
test_size=$((line_count / 10))
train_size=$((line_count - valid_size - test_size))

if [[ "$valid_size" -lt 1 ]]; then valid_size=1; fi
if [[ "$test_size" -lt 1 ]]; then test_size=1; fi
train_size=$((line_count - valid_size - test_size))

if [[ "$train_size" -lt 1 ]]; then
    echo "Not enough data after split to create train set." >&2
    exit 1
fi

train_end=$train_size
valid_end=$((train_size + valid_size))

sed -n "1,${train_end}p" "$cleaned_articles" > "$dataset_dir/train.txt"
sed -n "$((train_end + 1)),${valid_end}p" "$cleaned_articles" > "$dataset_dir/valid.txt"
sed -n "$((valid_end + 1)),${line_count}p" "$cleaned_articles" > "$dataset_dir/test.txt"

echo "Prepared Onion News dataset in: $dataset_dir"

#!/usr/bin/env python3
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create training/validation perplexity line charts from dropout sweep TSV tables."
    )
    parser.add_argument(
        "--train-log",
        type=str,
        default="models/dropout_train_perplexities.tsv",
        help="Path to TSV with per-epoch training perplexities.",
    )
    parser.add_argument(
        "--val-log",
        type=str,
        default="models/dropout_val_perplexities.tsv",
        help="Path to TSV with per-epoch validation perplexities.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="plots",
        help="Directory for output charts.",
    )
    parser.add_argument(
        "--test-log",
        type=str,
        default="models/dropout_test_perplexities.tsv",
        help="Path to TSV with test perplexities by dropout.",
    )
    return parser.parse_args()


def load_long_table(path):
    df = pd.read_csv(path, sep="\t")
    if "epoch" not in df.columns:
        raise ValueError(f"Missing 'epoch' column in {path}")

    value_cols = [c for c in df.columns if c.startswith("dropout_")]
    if not value_cols:
        raise ValueError(f"No dropout columns found in {path}")

    long_df = df.melt(id_vars=["epoch"], value_vars=value_cols, var_name="dropout", value_name="perplexity")
    long_df = long_df.dropna(subset=["perplexity"])
    long_df["epoch"] = pd.to_numeric(long_df["epoch"], errors="coerce")
    long_df["perplexity"] = pd.to_numeric(long_df["perplexity"], errors="coerce")
    long_df = long_df.dropna(subset=["epoch", "perplexity"])

    long_df["dropout_value"] = long_df["dropout"].str.replace("dropout_", "", regex=False).str.replace("_", ".", regex=False)
    long_df["dropout_value"] = pd.to_numeric(long_df["dropout_value"], errors="coerce")
    long_df = long_df.dropna(subset=["dropout_value"])
    long_df["dropout_label"] = long_df["dropout_value"].map(lambda x: f"{x:g}")
    return long_df

def _dropout_to_float(column_name):
    token = column_name.replace("dropout_", "").replace("_", ".")
    try:
        return float(token)
    except ValueError:
        return float("inf")

def load_wide_table(path):
    df = pd.read_csv(path, sep="\t")
    if "epoch" not in df.columns:
        raise ValueError(f"Missing 'epoch' column in {path}")
    value_cols = [c for c in df.columns if c.startswith("dropout_")]
    if not value_cols:
        raise ValueError(f"No dropout columns found in {path}")

    value_cols = sorted(value_cols, key=_dropout_to_float)
    df = df[["epoch"] + value_cols].copy()
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce").astype("Int64")
    for col in value_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def write_markdown_table(df, output_path, title):
    headers = ["epoch"] + [
        f"dropout={_dropout_to_float(col):g}" if _dropout_to_float(col) != float("inf") else col
        for col in df.columns[1:]
    ]
    lines = [f"# {title}", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for _, row in df.iterrows():
        row_cells = []
        epoch_val = row.iloc[0]
        row_cells.append("" if pd.isna(epoch_val) else str(int(epoch_val)))
        for value in row.iloc[1:]:
            row_cells.append("" if pd.isna(value) else f"{value:.2f}")
        lines.append("| " + " | ".join(row_cells) + " |")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")

def load_test_long_table(path):
    df = pd.read_csv(path, sep="\t")
    if "metric" not in df.columns:
        raise ValueError(f"Missing 'metric' column in {path}")
    if "test_ppl" not in df["metric"].astype(str).values:
        raise ValueError(f"Missing 'test_ppl' row in {path}")

    row = df.loc[df["metric"].astype(str) == "test_ppl"].iloc[0]
    value_cols = [c for c in df.columns if c.startswith("dropout_")]
    if not value_cols:
        raise ValueError(f"No dropout columns found in {path}")

    rows = []
    for col in value_cols:
        val = pd.to_numeric(row[col], errors="coerce")
        if pd.isna(val):
            continue
        rows.append({"dropout": col, "dropout_value": _dropout_to_float(col), "perplexity": float(val)})

    out = pd.DataFrame(rows)
    out = out.sort_values("dropout_value")
    out["dropout_label"] = out["dropout_value"].map(lambda x: f"{x:g}")
    return out

def write_test_markdown_table(df, output_path, title):
    lines = [f"# {title}", ""]
    lines.append("| dropout | test_perplexity |")
    lines.append("| --- | --- |")
    for _, row in df.iterrows():
        lines.append(f"| {row['dropout_label']} | {row['perplexity']:.4f} |")

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(lines) + "\n")


def save_plot(df, title, output_path):
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=df, x="epoch", y="perplexity", hue="dropout_label", marker="o")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.legend(title="Dropout", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    train_df = load_long_table(args.train_log)
    val_df = load_long_table(args.val_log)
    test_df = load_test_long_table(args.test_log)
    train_wide_df = load_wide_table(args.train_log)
    val_wide_df = load_wide_table(args.val_log)

    train_out = os.path.join(args.out_dir, "train_perplexity.png")
    val_out = os.path.join(args.out_dir, "validation_perplexity.png")
    test_out = os.path.join(args.out_dir, "test_perplexity.png")
    train_table_out = os.path.join(args.out_dir, "train_perplexity_table.md")
    val_table_out = os.path.join(args.out_dir, "validation_perplexity_table.md")
    test_table_out = os.path.join(args.out_dir, "test_perplexity_table.md")

    save_plot(train_df, "Training Perplexity by Epoch and Dropout", train_out)
    save_plot(val_df, "Validation Perplexity by Epoch and Dropout", val_out)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=test_df, x="dropout_value", y="perplexity", marker="o")
    plt.title("Test Perplexity by Dropout")
    plt.xlabel("Dropout")
    plt.ylabel("Perplexity")
    plt.tight_layout()
    plt.savefig(test_out, dpi=150)
    plt.close()
    write_markdown_table(train_wide_df, train_table_out, "Training Perplexity Table")
    write_markdown_table(val_wide_df, val_table_out, "Validation Perplexity Table")
    write_test_markdown_table(test_df, test_table_out, "Test Perplexity Table")

    print(f"Saved: {train_out}")
    print(f"Saved: {val_out}")
    print(f"Saved: {test_out}")
    print(f"Saved: {train_table_out}")
    print(f"Saved: {val_table_out}")
    print(f"Saved: {test_table_out}")


if __name__ == "__main__":
    main()

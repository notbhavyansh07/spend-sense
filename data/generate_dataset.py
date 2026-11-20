"""Generates data/expenses_labeled.csv (700 rows) and class count chart."""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spendsense.synthetic import generate_labeled


def generate_dataset(output_path=None, plot_path=None):
    if output_path is None:
        output_path = ROOT / "data" / "expenses_labeled.csv"
    if plot_path is None:
        plot_path = ROOT / "docs" / "class_counts.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_labeled(per_category=100, seed=42)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "category"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_path}")

    # Plot class counts
    df = pd.DataFrame(rows, columns=["text", "category"])
    counts = df["category"].value_counts().sort_index()

    plt.figure(figsize=(8, 4.5))
    counts.plot(kind="bar", color="#2b5c8f", edgecolor="black")
    plt.title("Expense Count per Category (Synthetic Dataset)")
    plt.xlabel("Category")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved class counts plot to {plot_path}")

    return output_path


generate_csv = generate_dataset


if __name__ == "__main__":
    generate_dataset()

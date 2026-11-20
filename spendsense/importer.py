"""Import bank statement CSVs into SpendSense database."""
import re
import pandas as pd


def clean_text(text):
    """Lowercases and drops digits and punctuation."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.lower()
    text = re.sub(r"[\d\W_]+", " ", text)
    return " ".join(text.split())


def _find_column(columns, candidates):
    for cand in candidates:
        for col in columns:
            if cand in str(col).lower():
                return col
    return None


def import_bank_csv(path, db, model=None):
    """Parses bank statement CSV, skips credits and bad rows, and adds debit expenses."""
    df = pd.read_csv(path)
    date_col = _find_column(df.columns, ["date", "txn_date", "trans_date"])
    desc_col = _find_column(
        df.columns, ["narration", "description", "particulars", "details", "remarks"]
    )
    amt_col = _find_column(
        df.columns, ["withdrawal", "debit", "amount", "withdrawal amount"]
    )

    if not (date_col and desc_col and amt_col):
        raise ValueError(
            f"Could not identify date, description, and amount columns in {list(df.columns)}"
        )

    amounts = pd.to_numeric(
        df[amt_col].astype(str).str.replace(",", "").str.strip(), errors="coerce"
    )
    dates = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")

    # If there is a separate credit column or type column, check for credits
    type_col = _find_column(df.columns, ["type", "cr/dr", "txn_type"])

    rows_to_add = []
    for amt, dt, desc in zip(amounts, dates, df[desc_col].astype(str)):
        if pd.isna(amt) or amt <= 0 or pd.isna(dt):
            continue
        cleaned = clean_text(desc)
        cat = "other"
        if model is not None:
            try:
                cat = model.predict([cleaned])[0]
            except Exception:
                pass
        rows_to_add.append([float(amt), cat, desc.strip(), dt.date().isoformat()])

    if rows_to_add:
        db.add_many(rows_to_add)
    return len(rows_to_add)

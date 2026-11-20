"""Extracts amount, debit/credit type, merchant, and date from bank SMS text."""
import re
from datetime import date

# Regex matching Rs, Rs., INR, ₹ with optional commas and decimals
AMOUNT_REGEX = re.compile(
    r"(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE
)

# Debit and credit keywords
DEBIT_REGEX = re.compile(
    r"\b(debited|spent|sent|paid|payment\s+of|purchase|withdrawn)\b",
    re.IGNORECASE,
)
CREDIT_REGEX = re.compile(
    r"\b(credited|received|deposited|refund)\b", re.IGNORECASE
)

STOP_DELIMITERS = (
    r"(?=\s+on\s|\s+ref\b|\s+upi\b|\s+avl\b|\s+bal\b|\s+via\b|\.\s|\.$|,|\s*$)"
)
MERCHANT_PATTERNS = [
    re.compile(r"\bat\s+([A-Za-z0-9&@.' -]+?)" + STOP_DELIMITERS, re.IGNORECASE),
    re.compile(r"\bto\s+([A-Za-z0-9&@.' -]+?)" + STOP_DELIMITERS, re.IGNORECASE),
]

NUMERIC_DATE_REGEX = re.compile(r"\b(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b")
MONTH_NAME_DATE_REGEX = re.compile(
    r"\b(\d{1,2})[-\s]([A-Za-z]{3})[a-z]*[-\s,]*(\d{2,4})\b", re.IGNORECASE
)
MONTH_MAPPING = {
    m: i + 1
    for i, m in enumerate(
        "jan feb mar apr may jun jul aug sep oct nov dec".split()
    )
}


def _normalize_year(year_val):
    year_int = int(year_val)
    return 2000 + year_int if year_int < 100 else year_int


def extract_date(text):
    try:
        match_num = NUMERIC_DATE_REGEX.search(text)
        if match_num:
            day = int(match_num.group(1))
            month = int(match_num.group(2))
            year = _normalize_year(match_num.group(3))
            return date(year, month, day)

        match_name = MONTH_NAME_DATE_REGEX.search(text)
        if match_name:
            day = int(match_name.group(1))
            month_str = match_name.group(2).lower()
            year = _normalize_year(match_name.group(3))
            if month_str in MONTH_MAPPING:
                return date(year, MONTH_MAPPING[month_str], day)
    except ValueError:
        pass
    return None


def parse_sms(text):
    """Parses a bank transaction SMS. Returns dict or None if no amount is found."""
    amount_match = AMOUNT_REGEX.search(text)
    if not amount_match:
        return None

    # Priority: if message has 'debited' (even if it also contains 'credited to'), it is a debit
    if DEBIT_REGEX.search(text):
        transaction_type = "debit"
    elif CREDIT_REGEX.search(text):
        transaction_type = "credit"
    else:
        transaction_type = "unknown"

    merchant = ""
    for pattern in MERCHANT_PATTERNS:
        match = pattern.search(text)
        if match:
            raw_merchant = match.group(1).strip(" .-")
            # UPI IDs: extract part before '@'
            merchant = raw_merchant.split("@")[0].strip().lower()
            break

    parsed_date = extract_date(text) or date.today()
    amount_clean = float(amount_match.group(1).replace(",", ""))

    return {
        "amount": amount_clean,
        "type": transaction_type,
        "merchant": merchant,
        "date": parsed_date.isoformat(),
    }

"""Synthetic dataset and expense history generator for SpendSense."""
import random
import string
from datetime import date, timedelta

CATS = {
    "food": {
        "vendors": [
            "zomato", "swiggy", "dominos", "kfc", "mcdonalds", "starbucks", "chaayos",
            "canteen", "mess", "cafe", "restaurant", "dhaba", "bigbasket", "blinkit"
        ],
        "items": [
            "dinner", "lunch", "breakfast", "biryani", "pizza", "burger", "coffee", "chai",
            "snacks", "thali", "groceries", "fruits", "ice cream", "sandwich", "paneer meal"
        ],
        "range": (60, 600),
    },
    "travel": {
        "vendors": [
            "uber", "ola", "rapido", "irctc", "redbus", "indigo", "metro", "indian oil"
        ],
        "items": [
            "cab ride", "bus ticket", "train ticket", "petrol", "fuel refill", "auto rickshaw",
            "flight booking", "parking", "toll fee", "bike taxi", "airport transfer"
        ],
        "range": (50, 1500),
    },
    "education": {
        "vendors": [
            "udemy", "coursera", "amazon books", "stationery shop", "xerox shop", "nptel"
        ],
        "items": [
            "python book", "notebook", "pens", "online course", "printing", "exam fee",
            "lab manual", "dsa book", "certification", "project material", "library fine"
        ],
        "range": (100, 3000),
    },
    "entertainment": {
        "vendors": [
            "netflix", "spotify", "bookmyshow", "hotstar", "prime video", "steam", "youtube"
        ],
        "items": [
            "subscription", "movie tickets", "premium plan", "game purchase", "concert pass",
            "bowling", "gaming cafe", "ott recharge", "arcade"
        ],
        "range": (100, 1200),
    },
    "bills": {
        "vendors": [
            "jio", "airtel", "vi", "bsnl", "tata power", "act fibernet"
        ],
        "items": [
            "mobile recharge", "electricity bill", "wifi bill", "hostel rent", "water bill",
            "gas cylinder", "room rent", "dth recharge", "laundry", "internet bill", "pg rent"
        ],
        "range": (200, 2500),
    },
    "health": {
        "vendors": [
            "apollo", "medplus", "pharmeasy", "netmeds", "cult fit", "1mg"
        ],
        "items": [
            "medicines", "doctor consultation", "pharmacy", "gym membership", "health checkup",
            "vitamins", "dental visit", "eye test", "protein powder"
        ],
        "range": (100, 2000),
    },
    "shopping": {
        "vendors": [
            "amazon", "flipkart", "myntra", "ajio", "decathlon", "croma", "nykaa"
        ],
        "items": [
            "tshirt", "shoes", "headphones", "backpack", "jeans", "charger", "watch",
            "phone cover", "hoodie", "keyboard", "water bottle"
        ],
        "range": (300, 5000),
    },
}

TEMPLATES = [
    "{v} {i}",
    "{i} from {v}",
    "paid {v} for {i}",
    "{i}",
    "{v}",
    "{v} order",
    "{i} at {v}",
]


def _typo(text, rng):
    if len(text) > 5 and rng.random() < 0.1:
        idx = rng.randrange(1, len(text) - 1)
        sub = rng.choice(string.ascii_lowercase)
        return text[:idx] + sub + text[idx + 1:]
    return text


def _case(text, rng):
    if rng.random() < 0.05:
        return text.upper()
    return text


def generate_labeled(per_category=100, seed=42):
    rng = random.Random(seed)
    dataset = []
    for cat, cfg in CATS.items():
        seen = set()
        while len(seen) < per_category:
            tmpl = rng.choice(TEMPLATES)
            v = rng.choice(cfg["vendors"])
            i = rng.choice(cfg["items"])
            text = tmpl.format(v=v, i=i).strip()
            text = _typo(text, rng)
            text = _case(text, rng)
            if text and text not in seen:
                seen.add(text)
                dataset.append((text, cat))
    return dataset


def _month_dates(year, month):
    start = date(year, month, 1)
    if month == 12:
        end = date(year + 1, 1, 1)
    else:
        end = date(year, month + 1, 1)
    days = (end - start).days
    return [start + timedelta(days=d) for d in range(days)]


def generate_expenses(months=12, seed=42):
    rng = random.Random(seed)
    today = date.today()
    cur_year = today.year
    cur_month = today.month

    month_tuples = []
    for m in range(months, 0, -1):
        target_month = cur_month - m
        target_year = cur_year
        while target_month <= 0:
            target_month += 12
            target_year -= 1
        month_tuples.append((target_year, target_month))

    rows = []
    for step_idx, (yr, mo) in enumerate(month_tuples):
        trend_mult = (1.0 + 0.015) ** step_idx
        days_in_month = _month_dates(yr, mo)
        for _ in range(41):
            cat = rng.choice(list(CATS.keys()))
            cfg = CATS[cat]
            low, high = cfg["range"]
            base_amount = rng.uniform(low, high) * trend_mult
            amount = round(base_amount, 2)
            spent_date = rng.choice(days_in_month).isoformat()
            vendor = rng.choice(cfg["vendors"])
            item = rng.choice(cfg["items"])
            note = f"{vendor} {item}"
            rows.append([amount, cat, note, spent_date])

    # Plant 6 distinct high-value outliers (6x to 10x normal amount)
    outlier_indices = rng.sample(range(len(rows)), 6)
    for idx in outlier_indices:
        cat = rows[idx][1]
        _, high = CATS[cat]["range"]
        mult = rng.uniform(6.0, 10.0)
        rows[idx][0] = round(high * mult, 2)
        rows[idx][2] += " (outlier)"

    return rows

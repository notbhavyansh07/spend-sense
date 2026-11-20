"""Natural language query parsing for expense questions.

Questions are parsed into a validated spec and executed in pandas.
No arbitrary SQL is ever generated from user queries.
"""
import json
import os
import re
from datetime import date, timedelta

import pandas as pd

INTENTS = {"sum", "mean", "count", "top", "breakdown"}
SYNONYMS = {
    "food": [
        "food", "eat", "eating", "restaurant", "restaurants", "meal",
        "meals", "dining", "lunch", "dinner", "snacks",
    ],
    "travel": [
        "travel", "transport", "cab", "cabs", "commute", "fuel", "petrol",
        "trip", "trips",
    ],
    "education": ["education", "study", "books", "book", "course", "courses"],
    "entertainment": [
        "entertainment", "movie", "movies", "fun", "netflix", "games",
        "gaming", "subscriptions",
    ],
    "bills": ["bills", "bill", "recharge", "rent", "electricity"],
    "health": ["health", "medical", "medicine", "medicines", "doctor", "gym"],
    "shopping": ["shopping", "clothes", "gadgets"],
}
CATEGORIES = set(SYNONYMS) | {"other"}
MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]


def parse_period(q, today):
    """Returns (start, end, label) where start and end are date objects or None."""
    m = re.search(r"last (\d+) (day|week|month)s?", q)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        days = n * {"day": 1, "week": 7, "month": 30}[unit]
        return (
            today - timedelta(days=days),
            today,
            f"in the last {n} {unit}{'s' if n > 1 else ''}",
        )
    if "yesterday" in q:
        d = today - timedelta(days=1)
        return d, d, "yesterday"
    if "today" in q:
        return today, today, "today"
    if "last week" in q:
        start = today - timedelta(days=today.weekday() + 7)
        return start, start + timedelta(days=6), "last week"
    if "this week" in q:
        return today - timedelta(days=today.weekday()), today, "this week"
    if "last month" in q:
        end = today.replace(day=1) - timedelta(days=1)
        return end.replace(day=1), end, "last month"
    if "this month" in q:
        return today.replace(day=1), today, "this month"
    if "last year" in q:
        return date(today.year - 1, 1, 1), date(today.year - 1, 12, 31), "last year"
    if "this year" in q:
        return date(today.year, 1, 1), today, "this year"

    for i, name in enumerate(MONTHS, 1):
        if name == "may" and not re.search(r"\b(in|during|for|of) may\b", q):
            continue
        if re.search(rf"\b{name}\b", q):
            y = today.year if i <= today.month else today.year - 1
            if i == 12:
                next_month = date(y + 1, 1, 1)
            else:
                next_month = date(y, i + 1, 1)
            end = next_month - timedelta(days=1)
            return date(y, i, 1), end, f"in {name.capitalize()}"

    return None, None, ""


def parse_rules(q, today=None):
    today = today or date.today()
    q = q.lower()
    words = set(re.findall(r"[a-z]+", q))
    cat = next((c for c, syn in SYNONYMS.items() if words & set(syn)), None)

    if re.search(
        r"(spend|spent) the most|most on|where did|by category|breakdown|each category|categories",
        q,
    ):
        intent = "breakdown"
    elif re.search(r"biggest|largest|highest|most expensive|\btop\b", q):
        intent = "top"
    elif re.search(r"average|avg|\bmean\b", q):
        intent = "mean"
    elif re.search(r"how many|number of|\bcount\b", q):
        intent = "count"
    else:
        intent = "sum"

    n_match = re.search(r"top (\d+)", q)
    start, end, label = parse_period(q, today)
    spec = {
        "intent": intent,
        "category": cat,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "n": int(n_match.group(1)) if n_match else 5,
    }
    return spec, label


def validate_spec(spec):
    """Enforces whitelist validation. Raises ValueError on unexpected parameters."""
    if not isinstance(spec, dict):
        raise ValueError("Spec must be a dictionary.")
    intent = spec.get("intent")
    if intent not in INTENTS:
        raise ValueError(f"Invalid intent '{intent}'. Allowed: {sorted(INTENTS)}")

    cat = spec.get("category")
    if cat is not None and cat not in CATEGORIES:
        raise ValueError(f"Invalid category '{cat}'. Allowed: {sorted(CATEGORIES)}")

    for k in ("start", "end"):
        val = spec.get(k)
        if val:
            date.fromisoformat(val)

    n = spec.get("n")
    if n is not None:
        try:
            n = int(n)
        except (ValueError, TypeError):
            n = 5
    else:
        n = 5

    return {
        "intent": intent,
        "category": cat,
        "start": spec.get("start") or None,
        "end": spec.get("end") or None,
        "n": max(1, min(n, 20)),
    }


def run_spec(df, spec, label=""):
    """Runs validated query spec over pandas DataFrame."""
    d = df.copy()
    if not d.empty and not pd.api.types.is_datetime64_any_dtype(d["spent_on"]):
        d["spent_on"] = pd.to_datetime(d["spent_on"])

    if spec["category"]:
        d = d[d["category"] == spec["category"]]
    if spec["start"]:
        d = d[d["spent_on"] >= pd.Timestamp(spec["start"])]
    if spec["end"]:
        d = d[d["spent_on"] < pd.Timestamp(spec["end"]) + pd.Timedelta(days=1)]

    where = (
        (f" on {spec['category']}" if spec["category"] else "")
        + (f" {label}" if label else "")
    )

    if d.empty:
        return f"No expenses found{where}."

    intent = spec["intent"]
    if intent == "sum":
        return f"You spent Rs {d['amount'].sum():,.0f}{where} across {len(d)} expenses."
    if intent == "mean":
        return f"Your average expense{where} is Rs {d['amount'].mean():,.0f}."
    if intent == "count":
        return f"You have {len(d)} expenses{where}."
    if intent == "top":
        rows = d.nlargest(spec["n"], "amount")
        lines = [
            f"- Rs {r.amount:,.0f}: {r.note} ({r.category}, {r.spent_on.strftime('%d %b %Y')})"
            for r in rows.itertuples()
        ]
        return f"Your biggest expenses{where}:\n" + "\n".join(lines)

    by = d.groupby("category")["amount"].sum().sort_values(ascending=False)
    lines = [f"- {c}: Rs {v:,.0f}" for c, v in by.items()]
    return f"Spending by category{where}:\n" + "\n".join(lines)


def _llm_spec(question, today):
    """Optional LLM spec generator with Anthropic Claude."""
    import anthropic

    client = anthropic.Anthropic()
    prompt = (
        f"Turn this question about personal spending into JSON only. Keys: "
        f"intent (one of {sorted(INTENTS)}), category (one of {sorted(CATEGORIES)} or null), "
        f"start and end (YYYY-MM-DD or null), n (integer or null). Today is {today}. "
        f"Question: {question}"
    )
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-5-5"),
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    json_start = text.find("{")
    json_end = text.rfind("}") + 1
    raw_spec = json.loads(text[json_start:json_end])
    return validate_spec(raw_spec)


def ask(df, question, today=None, use_llm=None):
    """Answers a spending question using rules or optional LLM with validated spec execution."""
    today = today or date.today()
    if df.empty:
        return {"answer": "There are no expenses yet.", "spec": None, "source": "none"}

    if use_llm is None:
        use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))

    if use_llm:
        try:
            spec = _llm_spec(question, today)
            label = (
                f"between {spec['start']} and {spec['end']}"
                if spec["start"] and spec["end"]
                else ""
            )
            return {"answer": run_spec(df, spec, label), "spec": spec, "source": "llm"}
        except Exception:
            pass  # Fallback to rule parser

    spec, label = parse_rules(question, today)
    validated = validate_spec(spec)
    return {"answer": run_spec(df, validated, label), "spec": validated, "source": "rules"}

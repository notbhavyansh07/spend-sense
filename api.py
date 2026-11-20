"""FastAPI backend REST API for SpendSense.

Run with: uvicorn api:app --reload
Interactive documentation available at http://localhost:8000/docs
"""
from datetime import date
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from spendsense import categorizer, chat, service, sms
from spendsense.database import ExpenseDB
from spendsense.synthetic import generate_expenses

app = FastAPI(title="SpendSense API", version="1.0")
db = ExpenseDB()
_cached_model = None


def get_model():
    global _cached_model
    if _cached_model is None:
        try:
            _cached_model = categorizer.load()
        except FileNotFoundError:
            _cached_model = None
    return _cached_model


def guess_category(text):
    m = get_model()
    if m is not None:
        try:
            return categorizer.predict(text, m)
        except Exception:
            pass
    return "other", 0.0


class ExpenseInput(BaseModel):
    amount: float = Field(gt=0, description="Expense amount must be greater than zero")
    note: str = ""
    category: Optional[str] = None
    spent_on: Optional[date] = None


class CategoryUpdateInput(BaseModel):
    category: str


class TextInput(BaseModel):
    text: str


class SmsInput(BaseModel):
    text: str
    save: bool = False


class QuestionInput(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/expenses")
def add_expense_endpoint(expense: ExpenseInput):
    cat = expense.category
    if not cat:
        cat = guess_category(expense.note)[0]
    spent_str = expense.spent_on.isoformat() if expense.spent_on else None
    new_id = db.add(expense.amount, cat, expense.note, spent_str)
    return {"id": new_id, "category": cat}


@app.get("/expenses")
def get_expenses_endpoint(limit: Optional[int] = None):
    return service.df_records(db.all(), limit=limit)


@app.put("/expenses/{expense_id}/category")
def update_expense_category(expense_id: int, body: CategoryUpdateInput):
    df = db.all()
    matched = df[df["id"] == expense_id]
    if matched.empty:
        raise HTTPException(status_code=404, detail="Expense not found")
    new_category = body.category.strip().lower()
    db.update_category(expense_id, new_category)
    categorizer.add_feedback(matched.iloc[0]["note"], new_category)
    return {"ok": True}


@app.delete("/expenses/{expense_id}")
def delete_expense_endpoint(expense_id: int):
    db.delete(expense_id)
    return {"ok": True}


@app.post("/predict-category")
def predict_category_endpoint(body: TextInput):
    cat, conf = guess_category(body.text)
    return {"category": cat, "confidence": conf}


@app.post("/parse-sms")
def parse_sms_endpoint(body: SmsInput):
    parsed = sms.parse_sms(body.text)
    if not parsed:
        raise HTTPException(status_code=422, detail="No transaction amount found in the SMS text")
    guess_text = parsed["merchant"] or body.text
    parsed["category"], parsed["confidence"] = guess_category(guess_text)
    if body.save and parsed["type"] == "debit":
        new_id = db.add(
            parsed["amount"],
            parsed["category"],
            parsed["merchant"],
            parsed["date"],
        )
        parsed["id"] = new_id
    return parsed


@app.get("/forecast")
def get_forecast_endpoint():
    f = service.forecast_summary(db.all())
    if not f:
        raise HTTPException(status_code=422, detail="Need at least 6 full months of historical data")
    return f


@app.get("/anomalies")
def get_anomalies_endpoint():
    return service.anomaly_records(db.all())


@app.post("/ask")
def ask_question_endpoint(body: QuestionInput):
    return chat.ask(db.all(), body.question)


@app.post("/demo")
def load_demo_endpoint():
    db.clear()
    rows = generate_expenses(months=12, seed=42)
    db.add_many(rows)
    return {"loaded": len(rows)}

# Week 9 Viva Notes: REST API

### 1. How does FastAPI utilize Pydantic for request validation, and how are errors formatted?
FastAPI relies on Pydantic models (such as `ExpenseInput`) with type annotations and validation fields (like `Field(gt=0)`). When a client sends a payload with negative amounts or missing required fields, FastAPI automatically rejects the request prior to invoking the endpoint logic, returning an HTTP 422 Unprocessable Entity status code with a standardized JSON error explaining the exact invalid field.

### 2. How does the API layer decouple data models from internal storage representations?
The database layer (`ExpenseDB`) interacts directly with SQLite and returns raw pandas DataFrames. `spendsense/service.py` functions as an intermediary service layer that transforms DataFrames into native Python dictionaries with ISO-formatted date strings (`service.df_records`). This decouples database schemas and pandas internal types from JSON serialization, allowing the dashboard and REST API to consume identical data formats.

### 3. How does POST /parse-sms combine NLP extraction with conditional persistence?
The endpoint calls `sms.parse_sms` to extract amount, type, merchant, and date. If an amount is detected, it immediately queries `guess_category` to provide a category prediction and confidence score. If the request body specifies `save: true` and the transaction is a debit, it automatically saves the record to SQLite and returns the generated database `id` within the response.

### 4. Why does the PUT /expenses/{id}/category endpoint simultaneously update feedback.csv?
When an expense category is updated via the API, it indicates that the existing or predicted category was incorrect. In addition to updating the SQLite row with `db.update_category`, the endpoint invokes `categorizer.add_feedback(note, new_category)`. This guarantees that feedback collected via HTTP clients or mobile frontends is saved for subsequent retraining runs.

### 5. What are the advantages of automatic Swagger UI documentation at /docs?
FastAPI inspects route definitions, type hints, and Pydantic schemas using OpenAPI 3.0 standards. It automatically serves an interactive Swagger documentation UI at `/docs`. Developers can test every endpoint directly in the browser, review schema requirements, inspect sample payloads, and verify status codes without external API tools like Postman.

---

### One Thing That Could Break
If a client issues a `GET /forecast` request when fewer than 6 complete months of transactions exist in the database, the forecast service returns `None`. The endpoint catches this and returns an HTTP 422 error; frontends that do not handle 422 responses gracefully may fail with unhandled exceptions.

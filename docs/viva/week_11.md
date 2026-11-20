# Week 11 Viva Notes: Question Answering

### 1. Why do we convert natural language questions into an intermediate query specification rather than generating SQL directly?
Generating raw SQL directly from user text creates severe security vulnerabilities, including SQL injection. In contrast, SpendSense converts questions into a constrained intermediate JSON specification:
`{"intent": intent, "category": category, "start": date, "end": date, "n": limit}`
The spec is checked against a strict whitelist of 5 supported intents and 7 categories. It is executed purely via pandas DataFrame slicing and aggregations, completely preventing arbitrary or destructive database access.

### 2. How does the rule-based parser handle temporal expressions without relying on external cloud APIs?
`parse_period` processes regular expressions relative to a reference date (`date.today()` or fixed date). It maps expressions like `"yesterday"`, `"last week"`, `"this month"`, `"last month"`, `"last 14 days"`, and full month names (`"in October"`) directly to exact calendar boundaries `(start_date, end_date)`. This runs entirely offline with zero latency and requires no network connectivity.

### 3. How does the optional LLM parser work, and what security guardrails protect it?
When `ANTHROPIC_API_KEY` is present in the environment, `chat.ask` prompts an LLM with strict instructions to return only a JSON representation of `{intent, category, start, end, n}`. Before this output is executed, it passes through the exact same `validate_spec` validator as rule-parsed outputs. If the LLM generates hallucinated keys, malicious SQL snippets, or unsupported intents, `validate_spec` throws a `ValueError`, causing the system to seamlessly fall back to the deterministic rule parser.

### 4. What are the 5 supported intents in the chat system and what calculations do they trigger?
1. `sum`: Computes total spending (`df['amount'].sum()`) for a category and timeframe.
2. `mean`: Computes average transaction amount (`df['amount'].mean()`).
3. `count`: Returns total transaction volume (`len(df)`).
4. `top`: Returns top \(N\) largest expenses sorted by amount (`df.nlargest(n, 'amount')`).
5. `breakdown`: Aggregates spending across all categories (`df.groupby('category')['amount'].sum()`).

### 5. How are category synonyms resolved in rule parsing?
A synonym dictionary maps diverse colloquial words to canonical database categories (for example, `"eat"`, `"dining"`, `"biryani"`, `"lunch"` map to `"food"`; `"commute"`, `"cabs"`, `"petrol"` map to `"travel"`). The parser extracts token intersections using set operations, matching informal vocabulary to canonical categories.

---

### One Thing That Could Break
If a question references complex composite or relative date logic that is not explicitly handled in `parse_period` (such as `"third Thursday of last quarter"` or `"since Diwali"`), the regex rules will fail to identify a date range and return `start=None, end=None`, causing the system to calculate the answer across all historical data rather than the intended period.

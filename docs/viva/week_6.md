# Week 6 Viva Notes: Bank SMS Parsing

### 1. How do you resolve conflicts in messages containing both "debited" and "credited to"?
In standard Indian banking formats, an account debit notification frequently reads:
`"Your account XX1234 is debited for Rs.89.00 on 04-11-2026 and credited to SWIGGY"`.
While the string contains both `"debited"` and `"credited"`, the transaction from the account holder's perspective is an expense (debit). In `spendsense/sms.py`, we evaluate debit keywords first. Any message containing a debit indicator is categorized as a `"debit"` regardless of subsequent recipient crediting clauses.

### 2. How are UPI merchant handles parsed and cleaned?
UPI IDs typically follow the format `username@bankhandle` or `merchant@upi` (for example, `ramesh@upi` or `swiggy.orders@icici`). The regex extracts the full string following `"to"` or `"at"`, strips trailing punctuation, and splits on `"@"` to retain only the prefix (`"ramesh"`). This normalized name is then fed into the categorizer model.

### 3. Why are amount formats with commas (e.g. INR 1,299.00) handled specially?
Indian financial systems often use thousands separators (commas) in amounts. A naive regex matching digits `\d+` would truncate `"1,299.00"` to `"1"`. The regex `(?:rs\.?|inr|₹)\s*([\d,]+(?:\.\d+)?)` captures numbers with commas, and `.replace(",", "")` strips the separators prior to converting to a `float`.

### 4. What happens when an SMS contains non-financial information such as an OTP?
Messages such as `"Your OTP is 123456"` contain numbers but no currency markers (Rs, INR, ₹). `parse_sms` requires a currency prefix match before extracting an amount. If no currency prefix is present, it returns `None`, preventing authentication codes from being erroneously recorded as monetary amounts.

### 5. How are dates normalized across varying date formats?
Indian bank SMS messages use diverse date patterns including two-digit years (`05-11-26`), four-digit years (`04-11-2026`), slash delimiters (`05/11/2026`), and three-letter month abbreviations (`03-Nov-2026`). Two regex patterns parse numeric and alphabetic dates, normalizing two-digit years (e.g. 26 -> 2026) and mapping month abbreviations (`nov` -> 11) to produce standard ISO format `YYYY-MM-DD`.

---

### One Thing That Could Break
If an SMS uses an unusual non-standard currency prefix or missing space (such as `"USD 50"` or `"Rs.500debited"` where the word directly abuts the number without spacing), the regex may fail to capture the amount, causing `parse_sms` to return `None`.

# Week 2 Viva Notes: Data Collection and Preprocessing

### 1. Why do we generate synthetic data with templates and injected noise instead of web scraping?
Real consumer expense records contain confidential personal information, banking details, and transaction credentials that are rarely published in open datasets. Synthetic data generation allows precise control over class balance (exactly 100 samples per category) and vocabulary diversity. Injecting 10% typos and 5% uppercase variations simulates noisy real-world text from user typing errors or SMS formatting.

### 2. What is the role of data/hard_test.csv in the machine learning evaluation?
`data/hard_test.csv` consists of 42 hand-crafted real-world transactions featuring merchants (such as "yatri sathi", "inox", "mamaearth", "bescom") that are completely absent from the synthetic training vendor list. It evaluates out-of-vocabulary generalization, measuring how well the model handles unseen merchants based on contextual tokens (like "cab", "recharge", "clinic", "shoes").

### 3. How does clean_text normalize bank statements, and what trade-offs exist?
`clean_text` lowercases strings and strips numbers and punctuation, collapsing whitespace into single spaces. This reduces vocabulary sparsity by ensuring "Uber-1234" and "uber" map to the same token. However, dropping digits also discards potentially informative numbers (such as bus routes like "bus 500D" or bill account numbers).

### 4. How does the synthetic expense history simulate real spending patterns over 12 months?
The generator simulates 12 past months with approximately 41 transactions per month across realistic category-specific price ranges (for example, food ranges from Rs 60 to 600, while shopping ranges up to Rs 5000). A compound 1.5% monthly upward multiplier models inflation or lifestyle creep. Six planted high-value outliers (6x to 10x normal bounds) provide ground truth for testing anomaly detection.

### 5. Why must bank CSV importers parse dates with dayfirst=True?
Most Indian banks and UPI applications format statement dates using British/Commonwealth convention (DD-MM-YYYY or DD/MM/YYYY) rather than American convention (MM-DD-YYYY). Setting `dayfirst=True` prevents ambiguous dates such as "05-11-2026" from being mistakenly interpreted as May 11 instead of November 5.

---

### One Thing That Could Break
If a bank changes its statement CSV header names from "Narration" or "Description" to an unexpected column name like "Transaction Remarks - Details", the column discovery heuristic may fail to match, causing the importer to raise a `ValueError` and reject the file.

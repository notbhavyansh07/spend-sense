# Week 1 Viva Notes: Problem Analysis and Planning

### 1. What problem does SpendSense solve, and why does it need three distinct ML tasks?
Students often track expenses too late because manual logging is tedious. SpendSense solves this by automating entry and analysis. It uses three ML tasks because each addresses a different stage of financial tracking: text classification automates categorization at transaction time, time series forecasting predicts next month's total to prevent overspending in advance, and anomaly detection highlights individual transactions that deviate from historical patterns.

### 2. Why did you start with a simple TF-IDF and Logistic Regression baseline instead of a transformer?
A baseline sets an initial performance benchmark with minimal computational complexity. TF-IDF with Logistic Regression trains in milliseconds on CPU, requires no external GPU dependencies, and provides an interpretable starting point. Once the baseline is measured, any improvements from character n-grams or embeddings can be quantified against it.

### 3. Why is SQLite appropriate for the local database layer, and how is data integrity enforced?
SQLite requires no separate database daemon or network setup, storing records in a single local file. Data integrity is enforced using SQL column constraints such as `CHECK(amount > 0)` and `NOT NULL`, preventing zero or negative values and empty records at the database engine level.

### 4. What is the difference between word-level TF-IDF and character n-gram TF-IDF for short expense texts?
Word-level TF-IDF tokenizes on whitespace and punctuation. If an expense text contains a typo, abbreviation, or novel merchant name (such as "swigy" instead of "swiggy"), word-level tokenization produces an out-of-vocabulary token and yields zero similarity. Character n-grams decompose words into sub-word slices (e.g. 2-4 characters), capturing morphological roots and tolerating minor spelling variations common in merchant receipts.

### 5. Why do we avoid committing trained machine learning models to the git repository?
Trained model binary files (such as `.joblib` or `.pkl`) can be large, change with every training run, and cause repository bloat. In addition, model binaries are derived artifacts that depend on specific library versions and code. Committing the training script and dataset ensures reproducible builds while keeping the git repository lightweight.

---

### One Thing That Could Break
If an expense description contains only punctuation, emojis, or numbers (for example, "12345" or "!!!"), a word-level TF-IDF vectorizer with standard token patterns will extract zero valid features. When passed to the classifier, the resulting empty feature vector may cause unpredictable predictions or low-confidence arbitrary class assignments.

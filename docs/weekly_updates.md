# Weekly Updates

### Week 1 — Problem Analysis and Planning
Scaffolded repository structure, continuous integration workflow, and problem documentation. Built a command-line interface with a local SQLite database for logging expenses and a baseline category classifier using word-level TF-IDF and Logistic Regression on 75 hand-written entries. The baseline model achieved a test accuracy of 0.400 on the 15-sample test split. A key limitation is that with only ~11 samples per category, word-level features fail to generalize to novel phrases or minor spelling variations.

- **Files changed**: `.github/workflows/tests.yml`, `.gitignore`, `LICENSE`, `README.md`, `requirements.txt`, `data/generate_dataset.py`, `data/expenses_labeled.csv`, `docs/PROBLEM.md`, `docs/viva/week_1.md`, `docs/weekly_updates.md`, `main.py`, `tests/test_spendsense.py`
- **Commands run**:
  - `python data/generate_dataset.py`
  - `python main.py add 150 food "lunch at canteen"`
  - `python main.py view`
  - `python main.py total`
  - `python main.py train`
  - `python main.py predict "swiggy burger delivery"`
  - `python -m unittest discover -s tests -v`
  - `git push -u origin main`
  - `git push -u origin week-1`
- **Test count and result**: 5 tests run, 5 passed (`OK`)
- **Anything untested**: GitHub Actions cloud runner execution on GitHub infrastructure

### Week 2 — Data Collection and Preprocessing
Built a synthetic expense text generator with template randomization, 10% injected typos, and 5% uppercase variations to produce a balanced 700-row training dataset (100 per category). Generated 12 months of synthetic student spending history with a 1.5% monthly upward trend and 6 planted high-value outliers. Created a hand-labeled 42-row hard evaluation set of unseen merchants, implemented a bank CSV importer with column heuristics and date normalization, and saved the category distribution plot to docs/class_counts.png.

- **Files changed**: `spendsense/__init__.py`, `spendsense/database.py`, `spendsense/synthetic.py`, `spendsense/importer.py`, `data/generate_dataset.py`, `data/expenses_labeled.csv`, `data/hard_test.csv`, `docs/class_counts.png`, `docs/viva/week_2.md`, `tests/test_spendsense.py`
- **Commands run**:
  - `python data/generate_dataset.py`
  - `python -m unittest discover -s tests -v`
- **Test count and result**: 9 tests run, 9 passed (`OK`)
- **Anything untested**: None

### Week 3 — Category Classification
Implemented character n-gram TF-IDF feature extraction (char_wb, ngrams 2-4) across Multinomial Naive Bayes, Logistic Regression, and LinearSVC. Selected the best model using 5-fold cross-validation on the 80% training partition, where LinearSVC achieved the highest CV score (0.998 vs 0.996 for Logistic Regression and 0.988 for Naive Bayes). On the held-out test split, LinearSVC scored 1.000 accuracy, but dropped to 0.595 (59.5%) on the unseen-merchant hard evaluation set. Implemented softmax normalization over decision scores for confidence estimation and persisted artifacts to models/categorizer.joblib and models/metrics.json.

- **Files changed**: `spendsense/categorizer.py`, `main.py`, `docs/viva/week_3.md`, `tests/test_spendsense.py`, `docs/weekly_updates.md`
- **Commands run**:
  - `python main.py train` (Best: linear_svm, CV: 0.998, Test: 1.000, Hard: 0.595)
  - `python main.py predict "netflix subscription"` (entertainment, 58.6%)
  - `python main.py predict "uber ride"` (travel, 52.4%)
  - `python -m unittest discover -s tests -v` (11 tests in 9.463s, OK)
- **Test count and result**: 11 tests run, 11 passed (`OK`)
- **Anything untested**: None

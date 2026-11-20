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

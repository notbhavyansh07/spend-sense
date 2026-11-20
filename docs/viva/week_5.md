# Week 5 Viva Notes: Database and Feedback Loop

### 1. How does the user feedback loop work, and how does it prevent catastrophic forgetting?
When a user overrides or corrects an expense category via the CLI menu or API, `categorizer.add_feedback` writes the original text and corrected category into `data/feedback.csv`. When `python main.py train` is executed, `categorizer.load_dataset` concatenates the base synthetic dataset (`data/expenses_labeled.csv`) with `data/feedback.csv`. Because the model is retrained on the merged dataset rather than fine-tuned solely on new instances, base knowledge is preserved and catastrophic forgetting is avoided.

### 2. Why is SPENDSENSE_DB environment variable support critical for test isolation?
If unit tests shared the default database file (`spendsense.db` or `data/expenses.db`), running tests would overwrite or erase user records, and concurrent test runs would produce race conditions. With `SPENDSENSE_DB`, tests point to temporary database files in isolated directories (or `:memory:`), guaranteeing clean test isolation and predictable state.

### 3. What database constraints ensure data hygiene in SQLite?
The table definition uses:
- `amount REAL NOT NULL CHECK(amount > 0)`: Rejects zero and negative amounts at the database engine level.
- `category TEXT NOT NULL`: Prevents unassigned categories.
- `spent_on TEXT NOT NULL`: Enforces that every expense has an associated date stamp.

### 4. How does the CLI interactive add workflow streamline transaction entry?
When the user types an expense description like `"swiggy biryani"`, `suggest_category` invokes the trained model and computes prediction confidence. It displays the suggestion and allows the user to press Enter to accept it instantly, reducing friction to a single keystroke. If the prediction is incorrect, typing a replacement automatically records training feedback.

### 5. Why is data/feedback.csv excluded from version control in .gitignore?
User feedback contains personal transaction notes, merchant names, and potentially private spending descriptions entered on a user's machine. Excluding `data/feedback.csv` from git protects user privacy and ensures that local user data is not committed into public code repositories.

---

### One Thing That Could Break
If a user enters a category with leading/trailing whitespace or unusual casing in the correction prompt (e.g. `" Food "`), and the input is not normalized with `.strip().lower()`, the model will treat it as a distinct class label (e.g. creating both `"food"` and `" Food "`), distorting stratified splits and cross-validation folds.

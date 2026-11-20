# SpendSense

SpendSense is an expense tracker with machine learning built in. I am building it for my AI/ML specialization project over 12 weekly milestones. It helps students track personal expenses without manual friction by categorizing expenses, parsing transaction SMS text, importing bank statements, forecasting monthly budgets, and flagging unusual spending.

## Current Status (Week 1)

In Week 1, I established the project foundations:
- Defined the problem, objectives, ML tasks, and risks in [PROBLEM.md](docs/PROBLEM.md).
- Built a command-line interface with a local SQLite database to add, view, and calculate total expenses.
- Built a first category classification baseline using word-level TF-IDF and Logistic Regression on 75 hand-written labeled expenses across 7 categories (`food`, `travel`, `education`, `entertainment`, `bills`, `health`, `shopping`).
- Measured baseline test accuracy: **0.400 (40.0%)** on an 80/20 train/test split (60 training samples, 15 testing samples). As expected, accuracy is low with such a small dataset; richer synthetic training data will be introduced in Week 2.
- Set up unit testing and continuous integration via GitHub Actions.

## Project Structure

```
main.py                      Command line app (SQLite storage and baseline model)
requirements.txt             Project dependencies
data/
  generate_dataset.py        Dataset generator (75 baseline labeled examples)
  expenses_labeled.csv       Generated labeled dataset
docs/
  PROBLEM.md                 Problem analysis and system design
  viva/
    week_1.md                Viva preparation questions and answers
tests/
  test_spendsense.py         Unit tests for database and baseline model
.github/workflows/
  tests.yml                  CI workflow running tests on push and pull requests
```

## Setup and Usage

### Prerequisites
- Python 3.12

### Installation

1. Clone the repository and enter the directory:
   ```bash
   git clone https://github.com/notbhavyansh07/spend-sense.git
   cd spend-sense
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Generate the baseline dataset:
   ```bash
   python data/generate_dataset.py
   ```

### Running the App

1. Train the baseline category model:
   ```bash
   python main.py train
   ```

2. Add an expense:
   ```bash
   python main.py add 150 food "lunch at canteen"
   ```

3. View recorded expenses:
   ```bash
   python main.py view
   ```

4. View total spending:
   ```bash
   python main.py total
   ```

5. Predict a category from description:
   ```bash
   python main.py predict "swiggy burger delivery"
   ```

### Running Tests

Run the test suite with unittest:
```bash
python -m unittest discover -s tests -v
```

## License

MIT

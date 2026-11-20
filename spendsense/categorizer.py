"""Category classification pipeline using character n-grams and linear models."""
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "expenses_labeled.csv"
FEEDBACK_PATH = ROOT / "data" / "feedback.csv"
HARD_PATH = ROOT / "data" / "hard_test.csv"
MODEL_PATH = ROOT / "models" / "categorizer.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"


def clean_text(text):
    """Lowercases text and strips non-alphabetic characters."""
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _make_pipeline(classifier):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                preprocessor=clean_text,
                analyzer="char_wb",
                ngram_range=(2, 4),
            ),
        ),
        ("clf", classifier),
    ])


CANDIDATES = {
    "naive_bayes": lambda: _make_pipeline(MultinomialNB()),
    "logistic_regression": lambda: _make_pipeline(
        LogisticRegression(max_iter=1000, random_state=42)
    ),
    "linear_svm": lambda: _make_pipeline(
        LinearSVC(random_state=42, dual="auto")
    ),
}


def get_candidates(use_embeddings=False):
    c = dict(CANDIDATES)
    if use_embeddings:
        from . import embed

        if not embed.available():
            raise RuntimeError("sentence-transformers is not installed.")
        c["embeddings_lr"] = embed.make_pipeline
    return c


def load_dataset(data_path=DATA_PATH):
    data_path = Path(data_path)
    if not data_path.exists():
        from data.generate_dataset import generate_dataset

        generate_dataset(data_path)
    df = pd.read_csv(data_path)
    if data_path == DATA_PATH and FEEDBACK_PATH.exists():
        fb_df = pd.read_csv(FEEDBACK_PATH)
        df = pd.concat([df, fb_df], ignore_index=True)
    return df.dropna().drop_duplicates().reset_index(drop=True)


def hard_accuracy(model):
    """Calculates accuracy on data/hard_test.csv containing unseen merchants."""
    if not HARD_PATH.exists():
        return None
    h_df = pd.read_csv(HARD_PATH)
    predictions = model.predict(h_df["text"])
    return float(accuracy_score(h_df["category"], predictions))


def train(data_path=DATA_PATH, save=True, use_embeddings=False):
    """Evaluates candidates using 5-fold CV, trains the best model, and scores on test set."""
    df = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"],
    )

    cands = get_candidates(use_embeddings=use_embeddings)
    cv_scores = {}
    for name, maker in cands.items():
        pipe = maker()
        scores = cross_val_score(pipe, x_train, y_train, cv=5)
        cv_scores[name] = float(scores.mean())

    best_name = max(cv_scores, key=cv_scores.get)
    best_model = cands[best_name]()
    best_model.fit(x_train, y_train)

    test_preds = best_model.predict(x_test)
    test_acc = float(accuracy_score(y_test, test_preds))
    hard_acc = hard_accuracy(best_model)
    report = classification_report(
        y_test, test_preds, zero_division=0, output_dict=True
    )

    result = {
        "dataset_size": len(df),
        "cv_scores": cv_scores,
        "best_model": best_name,
        "test_accuracy": test_acc,
        "hard_test_accuracy": hard_acc,
        "report": report,
    }

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
        try:
            plot_confusion_matrix(best_model, x_test, y_test)
        except Exception:
            pass

    return result


def plot_confusion_matrix(model=None, x_test=None, y_test=None, output_path=None):
    """Plots and saves the confusion matrix to docs/confusion_matrix.png."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

    if model is None:
        model = load()
    if x_test is None or y_test is None:
        df = load_dataset()
        _, x_test, _, y_test = train_test_split(
            df["text"],
            df["category"],
            test_size=0.2,
            random_state=42,
            stratify=df["category"],
        )
    if output_path is None:
        output_path = ROOT / "docs" / "confusion_matrix.png"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(x_test)
    labels = sorted(list(set(y_test)))
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation=35)
    plt.title("Category Confusion Matrix (Test Split)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path



def compare_hard(use_embeddings=False):
    """Fits all candidates on the full training set and reports accuracy on hard_test.csv."""
    df = load_dataset()
    results = {}
    for name, maker in get_candidates(use_embeddings=use_embeddings).items():
        pipe = maker()
        pipe.fit(df["text"], df["category"])
        results[name] = hard_accuracy(pipe)
    return results


def add_feedback(text, category):
    """Appends user corrections to data/feedback.csv for retraining."""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not FEEDBACK_PATH.exists()
    with open(FEEDBACK_PATH, "a", encoding="utf-8", newline="") as f:
        if is_new:
            f.write("text,category\n")
        clean_entry = str(text).replace('"', "")
        f.write(f'"{clean_entry}",{category}\n')


def load(model_path=MODEL_PATH):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train it first with: python main.py train"
        )
    return joblib.load(model_path)


def predict(text, model=None):
    """Predicts (category, confidence). Uses softmax over decision scores for LinearSVC."""
    model = model or load()
    classifier = model.named_steps["clf"]

    # LinearSVC does not provide predict_proba; calculate softmax of decision_function scores
    if hasattr(classifier, "predict_proba"):
        probabilities = model.predict_proba([text])[0]
    else:
        decision_scores = model.decision_function([text])[0]
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probabilities = exp_scores / exp_scores.sum()

    best_index = int(probabilities.argmax())
    category = model.classes_[best_index]
    confidence = float(probabilities[best_index])
    return category, confidence

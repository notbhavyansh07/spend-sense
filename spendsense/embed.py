"""Optional sentence embedding pipeline for expense classification."""
import importlib.util

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def available():
    """Checks whether sentence-transformers package is installed."""
    return importlib.util.find_spec("sentence_transformers") is not None


class EmbedTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        model = self._get_model()
        return model.encode(list(x), show_progress_bar=False)

    def __getstate__(self):
        # Do not serialize the heavy PyTorch transformer model
        state = self.__dict__.copy()
        state["_model"] = None
        return state


def make_pipeline():
    return Pipeline([
        ("embed", EmbedTransformer()),
        ("clf", LogisticRegression(max_iter=2000, random_state=42)),
    ])

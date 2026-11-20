# Week 4 Viva Notes: Model Evaluation and Embeddings

### 1. Why did the model achieve 100% accuracy on the test split but only ~60% on hard_test.csv?
The test split was drawn from the same synthetic template distribution as the training split (independent and identically distributed, or IID). Because both splits shared the same vendor names and item vocabulary, the model memorized these associations easily. In contrast, `data/hard_test.csv` represents out-of-distribution (OOD) data with unseen merchants (e.g. "yatri sathi", "inox", "puma"), demonstrating the real generalization gap between synthetic and real-world inputs.

### 2. What is keyword polysemy in expense classification, and how does it cause errors?
Polysemy occurs when a single word can belong to multiple categories depending on context. For example, the token `"membership"` in `"ieee membership"` belongs to education, but in synthetic training data, `"membership"` only appeared in `"gym membership"` (health). Without contextual sentence understanding, n-gram models weigh the word toward the single category it occurred with during training.

### 3. How does the confusion matrix help detect systematic classification errors?
A confusion matrix visualizes true labels versus predicted labels as a grid. Off-diagonal elements identify specific pairwise confusion patterns, such as whether entertainment is frequently confused with travel due to shared words like "ticket" or "booking". This directly guides targeted data augmentation or feature engineering.

### 4. What is the role of sentence-transformers in spendsense/embed.py, and why is it optional?
`spendsense/embed.py` uses pre-trained dense semantic embeddings (e.g. `all-MiniLM-L6-v2`) that capture semantic proximity rather than surface character n-grams. For instance, an embedding model understands that "sneakers" is semantically close to "shoes" even if the letters do not overlap. It is kept optional because PyTorch and HuggingFace dependencies are heavy (>1 GB) and may not be available in minimal or constrained production environments.

### 5. Why do we avoid pickling the PyTorch model object directly inside the scikit-learn pipeline?
Serializing deep learning transformer weights directly with `joblib.dump` bloats the serialized artifact to hundreds of megabytes and tightly couples the artifact to specific PyTorch internal class structures. In `spendsense/embed.py`, `__getstate__` sets `_model = None` before serialization, ensuring only configuration is saved and weights are loaded dynamically at runtime.

---

### One Thing That Could Break
If a user passes `--embeddings` when `sentence-transformers` is not installed, attempting to run `python main.py train --embeddings` will raise a `RuntimeError` stating that `sentence-transformers` must be installed first.

# Week 3 Viva Notes: Category Classification

### 1. Why do character n-grams with analyzer="char_wb" perform better than word-level tokenization for expenses?
The setting `char_wb` creates character n-grams strictly within word boundaries (padded with whitespace). For short transaction strings containing brand names or typos (e.g. "zomatoo" or "swigy"), word tokenizers miss the match entirely due to out-of-vocabulary limitations. In contrast, character n-grams (ranges 2 to 4) extract overlapping sub-word fragments like `"zom"`, `"omat"`, and `"ato"`, allowing the model to match misspelled or compound merchant names.

### 2. How did you compare Multinomial Naive Bayes, Logistic Regression, and LinearSVC fairly?
All three models were wrapped in identical TF-IDF pipelines and compared using 5-fold stratified cross-validation on the 80% training split only. The test split (20%) remained completely unseen during model selection. The candidate with the highest mean cross-validation score was chosen and subsequently evaluated on the held-out test split.

### 3. How do you extract confidence probabilities from LinearSVC when it does not support predict_proba?
Linear Support Vector Classifiers optimize a hinge loss without native probabilistic formulation, so they lack a `predict_proba` method. However, `LinearSVC` exposes `decision_function`, which computes signed geometric distances from the separating hyperplanes. We apply a numerically stable softmax transformation over these raw decision scores:
\[ P(y = c) = \frac{e^{z_c - \max(z)}}{\sum_j e^{z_j - \max(z)}} \]
This produces normalized pseudo-probabilities summing to 1.0.

### 4. What is the purpose of stratified splitting when dividing 700 expense records?
The dataset contains 100 entries per category across 7 categories. Standard random splitting could produce uneven class distributions in the test set. Stratified splitting guarantees that both the 80% training partition (80 samples per category) and 20% test partition (20 samples per category) maintain identical class proportions.

### 5. What are the contents of models/metrics.json and why save them alongside the model artifact?
`models/metrics.json` records the exact metadata of the training run: total dataset size, cross-validation scores across all candidates, the winning model name, test split accuracy, hard test set accuracy, and per-class precision, recall, and F1 scores. Saving these metrics provides an audit trail for experiment tracking and documentation.

---

### One Thing That Could Break
If an input consists exclusively of non-alphabetic characters or numbers (such as an SMS containing only a phone number or reference code like "9876543210"), `clean_text` will strip all characters, leaving an empty string. The TF-IDF vectorizer will produce an all-zero vector, causing `decision_function` to output zero for all classes and returning arbitrary predictions.

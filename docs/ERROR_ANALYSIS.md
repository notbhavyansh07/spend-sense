# Week 4 Error Analysis: Out-of-Distribution Hard Test Evaluation

## Overview

The category classification model achieved **1.000 (100.0%)** test split accuracy on the synthetic dataset, but dropped to **0.595 (59.5%)** on `data/hard_test.csv` (25 correct out of 42 samples). When trained on 100% of the training dataset using `python main.py compare`, candidate accuracy on the hard test set was:
- **Naive Bayes**: 0.643 (64.3%)
- **LinearSVC**: 0.619 (61.9%)
- **Logistic Regression**: 0.548 (54.8%)

This performance gap occurs because synthetic data shares identical template patterns and vendor names across train and test splits, whereas `data/hard_test.csv` introduces authentic merchants, regional phrasing, and context words that were never seen during training.

---

## Detailed Failure Analysis

Below are the 17 misclassified transactions from `data/hard_test.csv` along with root cause explanations:

### 1. Keyword Polysemy and Shared Context Tokens
Several transaction texts contain keywords that appeared exclusively in another category during synthetic generation:
- `ieee membership` (True: **education** | Predicted: **health**)
  - *Reason*: The token `"membership"` only occurred in `"gym membership"` in the health training set.
- `leetcode premium` (True: **education** | Predicted: **entertainment**)
  - *Reason*: `"premium"` was heavily weighted toward `"spotify premium"` and `"youtube premium"` in entertainment.
- `inox tickets` (True: **entertainment** | Predicted: **travel**)
  - *Reason*: `"tickets"` was strongly associated with travel (`"bus ticket"`, `"train ticket"`, `"flight booking"`).
- `gas booking` (True: **bills** | Predicted: **travel**)
  - *Reason*: `"booking"` was an active feature in travel bookings (`"flight booking"`, `"ticket booking"`).
- `tablet strip purchase` (True: **health** | Predicted: **entertainment**)
  - *Reason*: `"purchase"` was learned primarily from `"game purchase"` in entertainment.
- `ikea study lamp` (True: **shopping** | Predicted: **education**)
  - *Reason*: `"study"` strongly activated education features (`"python book"`, `"study materials"`).

### 2. Unseen Regional Vendors and Slang
Synthetic training data contained major global and national brands (e.g. Zomato, Swiggy, Uber, Decathlon). Unseen regional or independent merchants have zero exact n-gram overlap:
- `paytm tea stall` (True: **food** | Predicted: **entertainment**)
  - *Reason*: `"paytm"` is not a food vendor; character sub-ngrams had accidental overlap with entertainment terms.
- `porter tempo shifting` (True: **travel** | Predicted: **education**)
  - *Reason*: Neither `"porter"`, `"tempo"`, nor `"shifting"` appeared in travel training data.
- `yatri sathi cab` (True: **travel** | Predicted: **food**)
  - *Reason*: `"yatri sathi"` is a regional cab cooperative absent from training data.
- `pvr weekend` (True: **entertainment** | Predicted: **bills**)
  - *Reason*: `"pvr"` and `"weekend"` were not present in synthetic entertainment data.
- `zee5 renewal` (True: **entertainment** | Predicted: **bills**)
  - *Reason*: `"renewal"` activated bill recharge n-grams.
- `society maintenance` (True: **bills** | Predicted: **food**)
  - *Reason*: Neither `"society"` nor `"maintenance"` appeared in bills training templates.
- `physio session` (True: **health** | Predicted: **shopping**)
  - *Reason*: Medical terminology like `"physio"` was absent from the synthetic health items list.
- `dolo and cough syrup` (True: **health** | Predicted: **food**)
  - *Reason*: `"dolo"` was absent; sub-ngrams from `"syrup"` overlapped with dessert and food items.
- `boat earbuds` (True: **shopping** | Predicted: **food**)
  - *Reason*: `"boat"` audio brand was not included in the synthetic shopping vendor list.
- `puma sneakers` (True: **shopping** | Predicted: **food**)
  - *Reason*: Footwear brand `"puma"` and item `"sneakers"` were missing from shopping lists.
- `arduino kit for project` (True: **education** | Predicted: **food**)
  - *Reason*: `"arduino"` was absent, leading to accidental tie-breaking into food.

---

## Conclusion and Next Steps

Character n-grams are effective at handling spelling mistakes and small morphological shifts, but cannot understand semantic intent when an entire vendor name and item are unseen. In Week 5, we introduce the user feedback loop (`data/feedback.csv`) so that when users correct these errors, the model incorporates real merchant data into subsequent training runs.

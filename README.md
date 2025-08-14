# Breast Cancer Classification using Support Vector Machines (SVM)

## Overview

This project utilises Support Vector Machines to predict diagnosis on breast Cancer subjects classified as follows

- 1 : Malignant
- 0 : Benign

---

## Tools & Libraries Used

- VScode, Python, Pandas
- Scikit-learn (StandardScaler, SVM, accuracy scores.classification_report, cross_val_score, GridSearchCV)
- Matplotlib (for plotting Decision Boundary)

### Dataset used

[Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## Steps Performed

###  Data Preparation

- Loaded dataset using pandas & scikit-learn.
- Split features (X) and labels (y).
- Standardized features using StandardScaler for SVM compatibility.
- Train-test split (80% train, 20% test).

---

###  Hyperparameter Tuning

- Used GridSearchCV with 5-fold cross-validation only on training data.
- Searched over:
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.01, 0.1, 1, 'scale'],
    'kernel': ['linear', 'rbf']
}
```

#### Best parameters

- 'C': 0.1,
- 'gamma': 0.01,
- 'kernel': 'linear'
- Best CV Score: 0.9758

---
### Model Training

- Linear SVM: Trained with best parameters.
- RBF SVM: Trained with best parameters.

---

### Model Evaluation
#### Class based Metrics

| Metric    | Linear SVM (Class 0) | Linear SVM (Class 1) | RBF SVM (Class 0) | RBF SVM (Class 1) |
| ------------- | ---------------------------- | ---------------------------- | ------------------------- | ------------------------- |
| Precision | 0.98                         | 0.98                         | 0.98                      | 0.94                      |
| Recall    | 0.98                         | 0.98                         | 0.96                      | 0.98                      |
| F1-score  | 0.98                         | 0.98                         | 0.97                      | 0.96                      |
| Support   | 65                           | 49                           | 67                        | 47                        |
#### Overall Metrics

| Metric                 | Linear SVM | RBF SVM |
| -------------------------- | -------------- | ----------- |
| Accuracy               | 0.9825     | 0.9649      |
| Macro Avg Precision    | 0.98           | 0.96        |
| Macro Avg Recall       | 0.98           | 0.97        |
| Macro Avg F1-score     | 0.98           | 0.96        |
| Weighted Avg Precision | 0.98           | 0.97        |
| Weighted Avg Recall    | 0.98           | 0.96        |
| Weighted Avg F1-score  | 0.98           | 0.97        |

---
### Decision Boundary
- Plotted decision boundaries for 2D feature projections.

<img width="565" height="455" alt="image" src="https://github.com/user-attachments/assets/5dd11704-ae46-4d8a-8aa7-ccf5b9ec60b4" />

---
### Overall conclusion
- Linear SVM slightly outperforms RBF SVM in this dataset.
- High precision and recall for both models → balanced performance.
- Minimal difference between CV and test accuracy → low overfitting risk.

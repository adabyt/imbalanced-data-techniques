# Imbalanced Classification: Comparing Resampling and Ensemble Strategies

## 📘 Overview

This project explores how to handle imbalanced datasets in binary classification tasks using Python's `scikit-learn` and `imbalanced-learn` libraries. It provides a comparative analysis of several **oversampling**, **undersampling**, **hybrid**, and **ensemble** techniques, evaluating their performance on a synthetic dataset.

📌 **Goal**: Improve classification performance on the minority class without sacrificing overall model accuracy.

---

## 🧪 Dataset

We generate a synthetic binary classification dataset using `make_classification` with the following properties:
- 1,000 samples
- 10 features (5 informative)
- Imbalanced label distribution: **95% class 0**, **5% class 1**

This mimics real-world class imbalance challenges seen in:
- Fraud detection
- Medical diagnosis (e.g., rare disease prediction)
- Customer churn

---

## ⚙️ Dependencies

Install required libraries using pip:

```bash
pip install scikit-learn imbalanced-learn numpy pandas
```

---

## 📊 Techniques Explained

Below are the main approaches to handling class imbalance:
- **SMOTE** (Synthetic Minority Over-sampling Technique):
  - Creates synthetic samples by interpolating between minority class instances and their k-nearest neighbors.
- **ADASYN** (Adaptive Synthetic Sampling):
  - Generates synthetic samples for minority class instances that are harder to learn (i.e., those near the decision boundary or misclassified). It adaptively shifts the decision boundary.
- **Tomek Links**:
  - Identifies pairs of opposite-class samples that are close together (Tomek links) and removes the majority class sample of the pair.
- **SMOTEENN** (SMOTE + Edited Nearest Neighbours):
  - SMOTE for oversampling, followed by Edited Nearest Neighbours for cleaning up noisy or ambiguous samples.
- **SMOTETomek** (SMOTE + Tomek Links):
  - SMOTE for oversampling, followed by Tomek Links for removing noisy samples and clarifying decision boundaries.
- **Borderline-SMOTE**:
  - A variant that only oversamples minority examples that are on the "borderline" of the decision region, making them more impactful.
- **NearMiss**:
  - Selects majority class samples that are "near" to minority class samples, trying to preserve the decision boundary.
- **Bagging Classifiers** (e.g., BalancedBaggingClassifier):
  - Builds multiple base estimators (e.g., Decision Trees) on different random subsets of the training data, often with inherent re-weighting or resampling.
- **Boosting Classifiers** (e.g., EasyEnsembleClassifier, BalanceCascadeClassifier):
  - Combines multiple weak learners sequentially, focusing on misclassified samples. These are often built with resampling at each iteration.

## 🧪 Techniques Used in This Project

| Category            | Methods Used |
|---------------------|---------------|
| **Baseline**        | Class-weighted Logistic Regression |
| **Oversampling**    | SMOTE, ADASYN, BorderlineSMOTE |
| **Undersampling**   | NearMiss (v1, v2, v3), TomekLinks |
| **Hybrid Methods**  | SMOTETomek, SMOTEENN |
| **Ensemble Models** | Balanced Bagging, Easy Ensemble, Balanced Random Forest |

All models are evaluated using:
- **Classification report**
- **F1 Score (minority class)**
- **ROC AUC**

---

## 🧪 Sample Output Summary

| Classifier                    | ROC AUC | F1 Score (minority) |
|------------------------------|---------|----------------------|
| Baseline                     | 0.9677  | 0.7857 |
| SMOTE                        | 0.9673  | 0.7407 |
| ADASYN                       | **0.9864**  | **0.8148** ⭐ |
| SMOTETomek                   | 0.9673  | 0.7407 |
| Balanced Bagging             | 0.9022  | 0.5714 |
| BorderlineSMOTE (v1)         | 0.9673  | 0.7333 |
| BorderlineSMOTE (v2)         | 0.9846  | 0.7742 |
| NearMiss (v1)                | 0.9027  | 0.6061 |
| NearMiss (v2 & v3)           | 0.9041  | 0.3509 |
| EasyEnsemble                 | 0.9357  | 0.5217 |
| Balanced Random Forest       | 0.9421  | 0.6250 |

---

## 📈 Key Findings

- **ADASYN** performed best in this example, focusing on harder-to-classify samples near decision boundaries.
- **SMOTE** and **SMOTETomek** gave identical results here, likely due to TomekLinks not removing many samples.
- **NearMiss** methods reduced performance significantly by aggressively undersampling the majority class.
- **EasyEnsembleClassifier** achieved high recall for class 1 but at the cost of many false positives.

---


## 📚 References

- [Imbalanced-learn Documentation](https://imbalanced-learn.org)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [SMOTE: Synthetic Minority Oversampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813)

---

## 🧠 Author's Note

This analysis was motivated by previous challenges in training accurate models on imbalanced datasets. It reflects an effort to deeply understand and benchmark the most effective strategies.

---

## 📌 License

MIT License

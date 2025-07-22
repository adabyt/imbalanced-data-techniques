import numpy as np
import pandas as pd
from collections import Counter
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, f1_score

# For imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, EasyEnsembleClassifier, BalancedRandomForestClassifier

# 1. Generate an imbalanced dataset (example)
X, y = make_classification(n_samples=1000, n_features=10, n_informative=5,
                           n_redundant=0, n_clusters_per_class=1, weights=[0.95, 0.05],
                           flip_y=0, random_state=42)

print(f"Original dataset shape: {Counter(y)}")      # Counter({np.int64(0): 950, np.int64(1): 50})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Train dataset shape: {Counter(y_train)}")   # Counter({np.int64(0): 665, np.int64(1): 35})
print(f"Test dataset shape: {Counter(y_test)}")     # Counter({np.int64(0): 285, np.int64(1): 15})

# Function to evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test, sampler=None, model_name="Logistic Regression"):
    if sampler:
        X_resampled, y_resampled = sampler.fit_resample(X_train, y_train)
        print(f"  Resampled train dataset shape ({sampler.__class__.__name__}): {Counter(y_resampled)}")
    else:
        X_resampled, y_resampled = X_train, y_train
        print(f"  No resampling (Original data)")  

    model = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced') # Start with balanced class weight
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n--- Results for {model_name} (Sampler: {sampler.__class__.__name__ if sampler else 'None'}) ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1 Score (minority class): {f1_score(y_test, y_pred, pos_label=1):.4f}")

    print("-" * 50)

# --- Experimenting with different techniques ---

# 1. Baseline (No resampling, but with class_weight='balanced')
evaluate_model(X_train, y_train, X_test, y_test, model_name="Baseline LR")

# --- Results for Baseline LR (Sampler: None) ---
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99       285
#            1       0.85      0.73      0.79        15

#     accuracy                           0.98       300
#    macro avg       0.92      0.86      0.89       300
# weighted avg       0.98      0.98      0.98       300

# ROC AUC: 0.9677
# F1 Score (minority class): 0.7857

# 2. Oversampling with SMOTE
smote_sampler = SMOTE(random_state=42)
evaluate_model(X_train, y_train, X_test, y_test, sampler=smote_sampler, model_name="LR with SMOTE")

#   Resampled train dataset shape (SMOTE): Counter({np.int64(0): 665, np.int64(1): 665})

# --- Results for LR with SMOTE (Sampler: SMOTE) ---
#               precision    recall  f1-score   support

#            0       0.98      0.99      0.99       285
#            1       0.83      0.67      0.74        15

#     accuracy                           0.98       300
#    macro avg       0.91      0.83      0.86       300
# weighted avg       0.98      0.98      0.98       300

# ROC AUC: 0.9673
# F1 Score (minority class): 0.7407

# 3. Oversampling with ADASYN
adasyn_sampler = ADASYN(random_state=42)
evaluate_model(X_train, y_train, X_test, y_test, sampler=adasyn_sampler, model_name="LR with ADASYN")

#   Resampled train dataset shape (ADASYN): Counter({np.int64(0): 665, np.int64(1): 662})

# --- Results for LR with ADASYN (Sampler: ADASYN) ---
#               precision    recall  f1-score   support

#            0       0.99      1.00      0.99       285
#            1       0.92      0.73      0.81        15

#     accuracy                           0.98       300
#    macro avg       0.95      0.86      0.90       300
# weighted avg       0.98      0.98      0.98       300

# ROC AUC: 0.9864
# F1 Score (minority class): 0.8148

# 4. Combination method: SMOTETomek
smotetomek_sampler = SMOTETomek(random_state=42)
evaluate_model(X_train, y_train, X_test, y_test, sampler=smotetomek_sampler, model_name="LR with SMOTETomek")

#   Resampled train dataset shape (SMOTETomek): Counter({np.int64(0): 665, np.int64(1): 665})

# --- Results for LR with SMOTETomek (Sampler: SMOTETomek) ---
#               precision    recall  f1-score   support

#            0       0.98      0.99      0.99       285
#            1       0.83      0.67      0.74        15

#     accuracy                           0.98       300
#    macro avg       0.91      0.83      0.86       300
# weighted avg       0.98      0.98      0.98       300

# ROC AUC: 0.9673
# F1 Score (minority class): 0.7407

# 5. Ensemble method: BalancedBaggingClassifier
# This integrates resampling within the ensemble process
print("\n--- BalancedBaggingClassifier (inherent resampling) ---")
balanced_bagging_model = BalancedBaggingClassifier(estimator=LogisticRegression(solver='liblinear', random_state=42),
                                                   sampling_strategy='auto',
                                                   random_state=42,
                                                   n_estimators=10) # 10 base estimators
balanced_bagging_model.fit(X_train, y_train)
y_pred_bb = balanced_bagging_model.predict(X_test)
y_proba_bb = balanced_bagging_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_bb))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_bb):.4f}")
print(f"F1 Score (minority class): {f1_score(y_test, y_pred_bb, pos_label=1):.4f}")
print("-" * 50)

# --- BalancedBaggingClassifier (inherent resampling) ---
#               precision    recall  f1-score   support

#            0       0.98      0.96      0.97       285
#            1       0.50      0.67      0.57        15

#     accuracy                           0.95       300
#    macro avg       0.74      0.82      0.77       300
# weighted avg       0.96      0.95      0.95       300

# ROC AUC: 0.9022
# F1 Score (minority class): 0.5714


# 6. Ensemble method: BorderlineSMOTE (kind='borderline-1')
borderlinesmote_sampler = BorderlineSMOTE(
    random_state=42,
    kind='borderline-1'
    )
evaluate_model(X_train, y_train, X_test, y_test, sampler=borderlinesmote_sampler, model_name="LR with BorderlineSMOTE (borderline-1)")

#   Resampled train dataset shape (BorderlineSMOTE): Counter({np.int64(0): 665, np.int64(1): 665})

# --- Results for LR with BorderlineSMOTE (Sampler: BorderlineSMOTE) ---
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99       285
#            1       0.73      0.73      0.73        15

#     accuracy                           0.97       300
#    macro avg       0.86      0.86      0.86       300
# weighted avg       0.97      0.97      0.97       300

# ROC AUC: 0.9673
# F1 Score (minority class): 0.7333


# 7. Ensemble method: BorderlineSMOTE (kind='borderline-1')
borderlinesmote_sampler = BorderlineSMOTE(
    random_state=42,
    kind='borderline-2'
    )
evaluate_model(X_train, y_train, X_test, y_test, sampler=borderlinesmote_sampler, model_name="LR with BorderlineSMOTE (borderline-2)")

#   Resampled train dataset shape (BorderlineSMOTE): Counter({np.int64(0): 665, np.int64(1): 665})

# --- Results for LR with BorderlineSMOTE (Sampler: BorderlineSMOTE) ---
#               precision    recall  f1-score   support

#            0       0.99      0.99      0.99       285
#            1       0.75      0.80      0.77        15

#     accuracy                           0.98       300
#    macro avg       0.87      0.89      0.88       300
# weighted avg       0.98      0.98      0.98       300

# ROC AUC: 0.9846
# F1 Score (minority class): 0.7742

# 8. Ensemble method: NearMiss (version=1)
nearmiss_sampler = NearMiss(version=1)
evaluate_model(X_train, y_train, X_test, y_test, sampler=nearmiss_sampler, model_name="LR with NearMiss (version=1)")

#   Resampled train dataset shape (NearMiss): Counter({np.int64(0): 35, np.int64(1): 35})

# --- Results for LR with NearMiss (version=1) (Sampler: NearMiss) ---
#               precision    recall  f1-score   support

#            0       0.98      0.97      0.98       285
#            1       0.56      0.67      0.61        15

#     accuracy                           0.96       300
#    macro avg       0.77      0.82      0.79       300
# weighted avg       0.96      0.96      0.96       300

# ROC AUC: 0.9027
# F1 Score (minority class): 0.6061

# 9. Ensemble method: NearMiss (version=2)
nearmiss_sampler = NearMiss(version=2)
evaluate_model(X_train, y_train, X_test, y_test, sampler=nearmiss_sampler, model_name="LR with NearMiss (version=2)")

#   Resampled train dataset shape (NearMiss): Counter({np.int64(0): 35, np.int64(1): 35})

# --- Results for LR with NearMiss (version=1) (Sampler: NearMiss) ---
#               precision    recall  f1-score   support

#            0       0.98      0.89      0.93       285
#            1       0.24      0.67      0.35        15

#     accuracy                           0.88       300
#    macro avg       0.61      0.78      0.64       300
# weighted avg       0.94      0.88      0.90       300

# ROC AUC: 0.9041
# F1 Score (minority class): 0.3509

# 10. Ensemble method: NearMiss (version=3)
nearmiss_sampler = NearMiss(version=2)
evaluate_model(X_train, y_train, X_test, y_test, sampler=nearmiss_sampler, model_name="LR with NearMiss (version=3)")

#   Resampled train dataset shape (NearMiss): Counter({np.int64(0): 35, np.int64(1): 35})

# --- Results for LR with NearMiss (version=1) (Sampler: NearMiss) ---
#               precision    recall  f1-score   support

#            0       0.98      0.89      0.93       285
#            1       0.24      0.67      0.35        15

#     accuracy                           0.88       300
#    macro avg       0.61      0.78      0.64       300
# weighted avg       0.94      0.88      0.90       300

# ROC AUC: 0.9041
# F1 Score (minority class): 0.3509

# 11. Ensemble method: EasyEnsembleClassifier
print("\n--- EasyEnsembleClassifier ---")
easyensemble_model = EasyEnsembleClassifier(
    n_estimators=10,
    random_state=42
    )
easyensemble_model.fit(X_train, y_train)
y_pred_ee = easyensemble_model.predict(X_test)
y_proba_ee = easyensemble_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_ee))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_ee):.4f}")
print(f"F1 Score (minority class): {f1_score(y_test, y_pred_ee, pos_label=1):.4f}")
print("-" * 50)

# --- EasyEnsembleClassifier ---
#               precision    recall  f1-score   support

#            0       0.99      0.93      0.96       285
#            1       0.39      0.80      0.52        15

#     accuracy                           0.93       300
#    macro avg       0.69      0.87      0.74       300
# weighted avg       0.96      0.93      0.94       300

# ROC AUC: 0.9357
# F1 Score (minority class): 0.5217


# 12. Ensemble method: BalancedRandomForestClassifier
print("\n--- BalancedRandomForestClassifier ---")
balancedrf_model = BalancedRandomForestClassifier(
    n_estimators=100,
    random_state=42
    )
balancedrf_model.fit(X_train, y_train)
y_pred_rf = balancedrf_model.predict(X_test)
y_proba_rf = balancedrf_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred_rf))
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_rf):.4f}")
print(f"F1 Score (minority class): {f1_score(y_test, y_pred_rf, pos_label=1):.4f}")
print("-" * 50)

# --- BalancedRandomForestClassifier ---
#               precision    recall  f1-score   support

#            0       0.98      0.98      0.98       285
#            1       0.59      0.67      0.62        15

#     accuracy                           0.96       300
#    macro avg       0.79      0.82      0.80       300
# weighted avg       0.96      0.96      0.96       300

# ROC AUC: 0.9421
# F1 Score (minority class): 0.6250

"""
Ways to handle imbalanced datasets:
- SMOTE (Synthetic Minority Over-sampling Technique): creates synthetic samples by interpolating between minority class instances and their k-nearest neighbors.
- ADASYN (Adaptive Synthetic Sampling): generates synthetic samples for minority class instances that are harder to learn (i.e., those near the decision boundary or misclassified). It adaptively shifts the decision boundary.
- Tomek Links: identifies pairs of opposite-class samples that are close together (Tomek links) and removes the majority class sample of the pair.
- SMOTEENN (SMOTE + Edited Nearest Neighbours): SMOTE for oversampling, followed by Edited Nearest Neighbours for cleaning up noisy or ambiguous samples.
- SMOTETomek (SMOTE + Tomek Links): SMOTE for oversampling, followed by Tomek Links for removing noisy samples and clarifying decision boundaries.
- Borderline-SMOTE: a variant that only oversamples minority examples that are on the "borderline" of the decision region, making them more impactful.
- NearMiss: selects majority class samples that are "near" to minority class samples, trying to preserve the decision boundary.
- Bagging Classifiers (e.g., BalancedBaggingClassifier): builds multiple base estimators (e.g., Decision Trees) on different random subsets of the training data, often with inherent re-weighting or resampling.
- Boosting Classifiers (e.g., EasyEnsembleClassifier, BalanceCascadeClassifier): combines multiple weak learners sequentially, focusing on misclassified samples. These are often built with resampling at each iteration.

Note: Exploring specific cost_weight
For a more nuanced cost-sensitive approach, we could set the weights manually if we knew the cost of false negatives vs. false positives.
e.g. If misclassifying the positive class (1) as negative (0) is 10x worse than misclassifying the negative class (0) as positive (1):

```
model_custom_weight = LogisticRegression(solver='liblinear', random_state=42, class_weight={0: 1, 1: 10})
model_custom_weight.fit(X_train, y_train)
```


Classifier              | ROC AUC     | F1 score
------------------------|-------------|----------
Baseline                |0.9677       |0.7857
SMOTE                   |0.9673       |0.7407
ADASYN                  |0.9864       |0.8148 ***
SMOTETomek              |0.9673       |0.7407
BalancedBagging         |0.9022       |0.5714
BorderlineSMOTE (1)     |0.9673       |0.7333
BorderlineSMOTE (2)     |0.9846       |0.7742
NearMiss (1)            |0.9027       |0.6061
NearMiss (2)            |0.9041       |0.3509
NearMiss (3)            |0.9041       |0.3509
EasyEnsemble            |0.9357       |0.5217
BalancedRandomForest    |0.9421       |0.6250

Conclusions:
In this example dataset, ADASYN performs the best with the maximum ROC AUC and maximum F1 score.
    ADASYN focuses on generating samples for minority class instances that are harder to learn (those near the decision boundary).
    This often leads to a more effective shift in the decision boundary, as it targets the "harder" examples.

SMOTE and SMOTETomek yielded identical results. 
    This can happen if the Tomek Links step doesn't remove any majority samples in this specific dataset configuration, or if the added SMOTE samples are not "cleaned up" enough to change the decision boundary significantly for the test set performance.

EasyEnsembleClassifier shows a much lower precision for class 1 (0.39) compared to recall (0.80). 
    This indicates it's predicting class 1 much more often, leading to many false positives. 
    While recall is high, the trade-off in precision makes the F1-score lower. 
    This is a common characteristic of some ensemble methods when trying to maximise recall.    
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

import mlflow
import mlflow.sklearn

rfm = pd.read_csv('data/rfm.csv')

######################### Split the Data ######################################

# Drop columns that shouldn't be part of the model
X = rfm.drop(columns=['CustomerId', 'Cluster', 'is_high_risk'])
y = rfm['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

######################### Train the Models ######################################


# Train Logistic Regression
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)

# Predictions
y_pred_log = log_model.predict(X_test)

# Evaluation
print("📊 Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_log))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log))
print("Classification Report:\n", classification_report(y_test, y_pred_log))


# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
print("\n🌲 Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))

######################### Hyperparameter Tunning ######################################

# Grid Search for Logistic Regression
param_grid_log = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear']
}

grid_log = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), 
                        param_grid_log, 
                        cv=5, 
                        scoring='f1', 
                        n_jobs=-1)
grid_log.fit(X_train, y_train)

print("🔍 Best Logistic Regression Params:", grid_log.best_params_)
print("F1 Score on Test Set:", grid_log.score(X_test, y_test))

# Randomized Search for Random Forest

param_dist_rf = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

random_rf = RandomizedSearchCV(RandomForestClassifier(random_state=42),
                               param_distributions=param_dist_rf,
                               n_iter=20,
                               cv=5,
                               scoring='f1',
                               random_state=42,
                               n_jobs=-1)
random_rf.fit(X_train, y_train)

print("🎯 Best Random Forest Params:", random_rf.best_params_)
print("F1 Score on Test Set:", random_rf.score(X_test, y_test))



######################### Model Evaluation ######################################

# Accuracy Model Evalution
# For the best Logistic Regression model
y_pred_log = grid_log.predict(X_test)
accuracy_log = accuracy_score(y_test, y_pred_log)
print("✅ Logistic Regression Accuracy:", accuracy_log)

# For the best Random Forest model
y_pred_rf = random_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("🌲 Random Forest Accuracy:", accuracy_rf)

# Precision Model Evaluation
# Logistic Regression
precision_log = precision_score(y_test, y_pred_log)
print("✅ Logistic Regression Precision:", precision_log)

# Random Forest
precision_rf = precision_score(y_test, y_pred_rf)
print("🌲 Random Forest Precision:", precision_rf)

# Recall Model Evaluation
# Logistic Regression
recall_log = recall_score(y_test, y_pred_log)
print("✅ Logistic Regression Recall:", recall_log)

# Random Forest
recall_rf = recall_score(y_test, y_pred_rf)
print("🌲 Random Forest Recall:", recall_rf)

# F1 Score Model Evalutaion
# Logistic Regression
f1_log = f1_score(y_test, y_pred_log)
print("✅ Logistic Regression F1 Score:", f1_log)

# Random Forest
f1_rf = f1_score(y_test, y_pred_rf)
print("🌲 Random Forest F1 Score:", f1_rf)

# ROC-AUC Model Evaluation
# Logistic Regression
y_prob_log = grid_log.predict_proba(X_test)[:, 1]  # get probability of class 1
roc_auc_log = roc_auc_score(y_test, y_prob_log)
print("✅ Logistic Regression ROC-AUC:", roc_auc_log)

# Random Forest
y_prob_rf = random_rf.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
print("🌲 Random Forest ROC-AUC:", roc_auc_rf)


# Plot ROC Curve (Optional but Recommended)
# ROC curve for both models
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 5))
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {roc_auc_log:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {roc_auc_rf:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Classifier (AUC = 0.5)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


######################### MLflow ######################################

with mlflow.start_run(run_name="LogisticRegression_CreditRisk") as run:
    # Log parameters
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("penalty", grid_log.best_params_['penalty'])
    mlflow.log_param("C", grid_log.best_params_['C'])
    mlflow.log_param("solver", grid_log.best_params_['solver'])
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_log)
    mlflow.log_metric("precision", precision_log)
    mlflow.log_metric("recall", recall_log)
    mlflow.log_metric("f1_score", f1_log)
    mlflow.log_metric("roc_auc", roc_auc_log)

    # Log model
    mlflow.sklearn.log_model(grid_log.best_estimator_, "model", registered_model_name="CreditRiskLogisticModel")
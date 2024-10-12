import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns 
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss, balanced_accuracy_score, roc_auc_score

# Replace the dataset with your validation dataset
merged_test = pd.read_csv("merged_test.csv")    # input the transformed test

X_test = merged_test.drop(['Unnamed: 0', 'target'], axis=1)
y_test = merged_test['target']

# Load the saved model
model = joblib.load("lgbm.pkl")

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    log_loss_value = log_loss(y_test, model.predict_proba(X_test)[:, 1])
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

    # AUC-ROC calculation if applicable
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Log Loss: {log_loss_value}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    print(f"AUC-ROC: {roc_auc}")
    print("\n" + "-" * 50 + "\n")
    
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues")
    plt.show()

    return accuracy, conf_matrix, precision, recall, f1, log_loss_value, balanced_accuracy, roc_auc

# Evaluate the loaded model
print("Evaluating LightGBM...")
results = evaluate_model(model, X_test, y_test)
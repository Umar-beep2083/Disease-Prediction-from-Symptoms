import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("dataset/training_data.csv")

# Fix NaN values
print(f"NaN values before cleaning: {df.isnull().sum().sum()}")
df = df.fillna(0)
print(f"NaN values after cleaning:  {df.isnull().sum().sum()}")

X = df.drop("prognosis", axis=1)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["prognosis"])

os.makedirs("model", exist_ok=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Classifier 1: Random Forest ===
print("\n=== Random Forest + GridSearchCV ===")
rf_params = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5]
}
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_params, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
rf_grid.fit(X, y)
print(f"Best RF Params : {rf_grid.best_params_}")
print(f"Best RF CV Acc : {rf_grid.best_score_:.4f}")

# === Classifier 2: SVM ===
print("\n=== SVM + GridSearchCV ===")
svm_params = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"]
}
svm_grid = GridSearchCV(SVC(probability=True, random_state=42),
                        svm_params, cv=cv, scoring="accuracy", n_jobs=-1, verbose=1)
svm_grid.fit(X, y)
print(f"Best SVM Params : {svm_grid.best_params_}")
print(f"Best SVM CV Acc : {svm_grid.best_score_:.4f}")

# === Comparison ===
print("\n=== Classifier Comparison ===")
print(f"  Random Forest CV Accuracy : {rf_grid.best_score_:.4f}")
print(f"  SVM           CV Accuracy : {svm_grid.best_score_:.4f}")

# === Pick winner ===
if rf_grid.best_score_ >= svm_grid.best_score_:
    best_model = rf_grid.best_estimator_
    print("\n✅ Winner: Random Forest")
else:
    best_model = svm_grid.best_estimator_
    print("\n✅ Winner: SVM")

# Save artifacts
pickle.dump(best_model,       open("model/best_model.pkl",      "wb"))
pickle.dump(label_encoder,    open("model/label_encoder.pkl",   "wb"))
pickle.dump(list(X.columns),  open("model/feature_columns.pkl", "wb"))
print("✅ All artifacts saved to model/")
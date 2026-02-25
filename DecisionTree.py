import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# ======================
# LOAD DATASET
# ======================

column_names = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

df = pd.read_csv("heart_disease.csv", names=column_names)

print("Dataset Loaded Successfully ✅")
print("\nFirst 5 rows:\n", df.head())

# ======================
# HANDLE MISSING VALUES
# ======================

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert columns to numeric
df = df.apply(pd.to_numeric)

# Fill missing values with median
df.fillna(df.median(), inplace=True)

print("\nMissing Values After Cleaning:\n", df.isnull().sum())

# Convert target to binary (0 = No Disease, 1 = Disease)
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

# ======================
# SPLIT DATA
# ======================

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nData Split Completed ✅")

# ======================
# TRAIN MODEL
# ======================

model = DecisionTreeClassifier(random_state=42, max_depth=4)
model.fit(X_train, y_train)

print("\nModel Trained Successfully ✅")

# ======================
# EVALUATION
# ======================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(report)

# ======================
# DISPLAY DECISION TREE
# ======================

plt.figure(figsize=(18,8))
plot_tree(model,
          feature_names=X.columns,
          class_names=["No Disease", "Disease"],
          filled=True)

plt.title("Decision Tree - Heart Disease Dataset")
plt.show()
import pandas as pd
import tkinter as tk
from tkinter import messagebox, ttk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# ======================
# LOAD DATA
# ======================

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv("diabetes.csv", names=columns)

# ======================
# SPLIT DATA
# ======================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# TRAIN MODEL
# ======================

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# ======================
# EVALUATION
# ======================

y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\nModel Trained Successfully ✅")
print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:\n")
print(report)

# ======================
# DISPLAY ONE TREE
# ======================

plt.figure(figsize=(18,8))
tree = rf.estimators_[0]

plot_tree(
    tree,
    feature_names=X.columns,
    class_names=["No Diabetes","Diabetes"],
    filled=True,
    max_depth=3
)

plt.title("Random Forest - Sample Tree")
plt.show()

# ======================
# GUI FUNCTION
# ======================

def predict_diabetes():
    try:
        values = []
        for entry in entries:
            values.append(float(entry.get()))

        input_data = pd.DataFrame([values], columns=X.columns)

        prediction = rf.predict(input_data)[0]

        if prediction == 1:
            result_label.config(
                text="Prediction: DIABETES ⚠",
                foreground="red"
            )
        else:
            result_label.config(
                text="Prediction: NO DIABETES 😊",
                foreground="green"
            )

    except:
        messagebox.showerror("Input Error", "Enter valid numeric values")

# ======================
# CREATE UI
# ======================

root = tk.Tk()
root.title("Diabetes Prediction System")
root.geometry("600x700")
root.configure(bg="#f4f6f9")

style = ttk.Style()
style.theme_use("clam")

main_frame = ttk.Frame(root, padding=20)
main_frame.pack(fill="both", expand=True)

title = ttk.Label(
    main_frame,
    text="🩺 Diabetes Prediction System",
    font=("Helvetica", 18, "bold")
)
title.grid(row=0, column=0, columnspan=2, pady=20)

labels = X.columns.tolist()

entries = []
for i, text in enumerate(labels):
    lbl = ttk.Label(main_frame, text=text, font=("Arial", 11))
    lbl.grid(row=i+1, column=0, sticky="w", pady=8)
    entry = ttk.Entry(main_frame, width=25)
    entry.grid(row=i+1, column=1, pady=8)
    entries.append(entry)

predict_btn = ttk.Button(
    main_frame,
    text="Predict",
    command=predict_diabetes
)
predict_btn.grid(row=len(labels)+1, column=0, columnspan=2, pady=20)

result_label = ttk.Label(
    main_frame,
    text="",
    font=("Arial", 14, "bold")
)
result_label.grid(row=len(labels)+2, column=0, columnspan=2, pady=15)

footer = ttk.Label(
    main_frame,
    text="Random Forest Model | Desktop ML Application",
    font=("Arial", 9)
)
footer.grid(row=len(labels)+3, column=0, columnspan=2, pady=10)

root.mainloop()
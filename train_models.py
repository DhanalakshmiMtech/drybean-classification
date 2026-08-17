import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings("ignore")


# Create folders for saved models and results
os.makedirs("model", exist_ok=True)
os.makedirs("results", exist_ok=True)


# --------------------------------------------------
# 1. Load the Dry Bean dataset
# --------------------------------------------------

file_path = "Dry_Bean_Dataset.xlsx"

data = pd.read_excel(file_path)

print("\nDataset loaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst 5 rows:")
print(data.head())

print("\nClass distribution:")
print(data["Class"].value_counts())


# --------------------------------------------------
# 2. Separate features and target
# --------------------------------------------------

X = data.drop(columns=["Class"])
y = data["Class"]


# --------------------------------------------------
# 3. Split data into training and testing sets
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=73,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 4. Define the five required ML models
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=2000,
                random_state=73
            )
        )
    ]),

    "Decision Tree": DecisionTreeClassifier(
        random_state=73
    ),

    "kNN": Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            KNeighborsClassifier(
                n_neighbors=7
            )
        )
    ]),

    "Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=73,
        n_jobs=-1
    )
}


# --------------------------------------------------
# 5. Train and evaluate every model
# --------------------------------------------------

results = []

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print("Training:", model_name)
    print("=" * 70)

    # Train model
    model.fit(X_train, y_train)

    # Generate predictions
    y_pred = model.predict(X_test)

    # Generate probability predictions
    y_proba = model.predict_proba(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    # Multiclass AUC
    auc = roc_auc_score(
        y_test,
        y_proba,
        multi_class="ovr",
        average="weighted"
    )

    # Weighted precision
    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Weighted recall
    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Weighted F1
    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    # Store results
    results.append({
        "ML Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "MCC": mcc
    })

    # Display metrics
    print(f"Accuracy : {accuracy:.4f}")
    print(f"AUC      : {auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"MCC      : {mcc:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=model.classes_
    )

    cm_df = pd.DataFrame(
        cm,
        index=model.classes_,
        columns=model.classes_
    )

    confusion_file = (
        "results/"
        + model_name.lower().replace(" ", "_")
        + "_confusion_matrix.csv"
    )

    cm_df.to_csv(confusion_file)

    # Save trained model
    model_file = (
        "model/"
        + model_name.lower().replace(" ", "_")
        + ".joblib"
    )

    joblib.dump(model, model_file)

    print("\nSaved model:", model_file)


# --------------------------------------------------
# 6. Create model comparison table
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\n\n")
print("=" * 80)
print("FINAL MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Save comparison table
results_df.to_csv(
    "results/model_comparison.csv",
    index=False
)


# --------------------------------------------------
# 7. Save test data
# --------------------------------------------------

test_data = X_test.copy()

test_data["Class"] = y_test.values

test_data.to_csv(
    "test_data.csv",
    index=False
)

print("\nTest data saved as: test_data.csv")


# --------------------------------------------------
# 8. Display overall winner
# --------------------------------------------------

winner = results_df.iloc[0]

print("\n" + "=" * 80)
print("OVERALL WINNER BASED ON F1 SCORE")
print("=" * 80)

print(
    "Model:",
    winner["ML Model"]
)

print(
    "F1 Score:",
    round(winner["F1"], 4)
)

print("\nTraining and evaluation completed successfully!")
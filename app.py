import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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


# ==========================================================
# Page configuration
# ==========================================================

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)


# ==========================================================
# Custom styling
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #1B5E20;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #555555;
        margin-bottom: 25px;
    }

    .metric-card {
        background-color: #F1F8E9;
        border-radius: 12px;
        padding: 15px;
        text-align: center;
        border: 1px solid #C5E1A5;
    }

    .section-title {
        color: #2E7D32;
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# Header
# ==========================================================

st.markdown(
    '<div class="main-title">🌱 Dry Bean Classification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Compare five machine learning classification models on the
    Dry Bean dataset using multiple evaluation metrics.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# Model paths
# ==========================================================

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}


# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("⚙️ Model Settings")

selected_model = st.sidebar.selectbox(
    "Select a classification model:",
    list(MODEL_FILES.keys())
)

st.sidebar.markdown("---")

st.sidebar.info(
    """
    **Dataset:** Dry Bean

    **Classes:** 7

    **Models:** 5

    **Test size:** 20%
    """
)


# ==========================================================
# File upload
# ==========================================================

st.markdown(
    '<div class="section-title">📂 Upload Test Dataset</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload your test_data.csv file",
    type=["csv"]
)


# ==========================================================
# Load uploaded data
# ==========================================================

if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to begin model evaluation."
    )

    st.markdown(
        """
        ### How to use this application

        1. Upload the test CSV generated during model training.
        2. Select one of the five machine learning models.
        3. View the six evaluation metrics.
        4. Examine the confusion matrix.
        5. Review the classification report.
        """
    )

    st.stop()


# ==========================================================
# Read CSV
# ==========================================================

try:

    test_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"Unable to read the uploaded CSV file: {error}"
    )

    st.stop()


# ==========================================================
# Validate target column
# ==========================================================

if "Class" not in test_data.columns:

    st.error(
        "The uploaded CSV must contain a 'Class' column."
    )

    st.stop()


# ==========================================================
# Display uploaded data
# ==========================================================

st.success(
    f"Dataset uploaded successfully: "
    f"{test_data.shape[0]} rows × {test_data.shape[1]} columns"
)

with st.expander("Preview uploaded test data"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )


# ==========================================================
# Separate features and target
# ==========================================================

X_test = test_data.drop(columns=["Class"])
y_test = test_data["Class"]


# ==========================================================
# Load selected model
# ==========================================================

model_path = MODEL_FILES[selected_model]


if not os.path.exists(model_path):

    st.error(
        f"Model file not found: {model_path}"
    )

    st.stop()


model = joblib.load(model_path)


# ==========================================================
# Generate predictions
# ==========================================================

try:

    y_pred = model.predict(X_test)

    y_proba = model.predict_proba(X_test)

except Exception as error:

    st.error(
        f"Prediction failed: {error}"
    )

    st.stop()


# ==========================================================
# Calculate evaluation metrics
# ==========================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class="ovr",
    average="weighted"
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ==========================================================
# Display selected model
# ==========================================================

st.markdown(
    f'<div class="section-title">🤖 {selected_model} Results</div>',
    unsafe_allow_html=True
)


# ==========================================================
# Metric cards
# ==========================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", f"{accuracy:.4f}")

with col2:
    st.metric("AUC", f"{auc:.4f}")

with col3:
    st.metric("Precision", f"{precision:.4f}")


col4, col5, col6 = st.columns(3)

with col4:
    st.metric("Recall", f"{recall:.4f}")

with col5:
    st.metric("F1 Score", f"{f1:.4f}")

with col6:
    st.metric("MCC", f"{mcc:.4f}")


# ==========================================================
# Confusion Matrix
# ==========================================================

st.markdown(
    '<div class="section-title">📊 Confusion Matrix</div>',
    unsafe_allow_html=True
)

class_labels = sorted(y_test.unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=class_labels
)

cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
)

fig, ax = plt.subplots(figsize=(9, 7))

sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Greens",
    linewidths=0.5,
    ax=ax
)

ax.set_xlabel("Predicted Class")
ax.set_ylabel("Actual Class")
ax.set_title(
    f"Confusion Matrix - {selected_model}"
)

st.pyplot(fig)

plt.close(fig)


# ==========================================================
# Classification Report
# ==========================================================

st.markdown(
    '<div class="section-title">📋 Classification Report</div>',
    unsafe_allow_html=True
)

report = classification_report(
    y_test,
    y_pred,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ==========================================================
# Model Comparison
# ==========================================================

st.markdown(
    '<div class="section-title">🏆 Model Comparison</div>',
    unsafe_allow_html=True
)

comparison_file = "results/model_comparison.csv"

if os.path.exists(comparison_file):

    comparison_df = pd.read_csv(
        comparison_file
    )

    numeric_columns = [
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC"
    ]

    for column in numeric_columns:
        comparison_df[column] = comparison_df[column].round(4)

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.warning(
        "Model comparison file is not available."
    )


# ==========================================================
# Footer
# ==========================================================

st.markdown("---")

st.caption(
    "Dry Bean Classification | Machine Learning Assignment"
)
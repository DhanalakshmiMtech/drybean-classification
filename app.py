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
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Dry Bean Classification",
    page_icon="🌱",
    layout="wide"
)


# ==========================================================
# CUSTOM CSS
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

    .section-title {
        color: #2E7D32;
        font-size: 25px;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<div class="main-title">🌱 Dry Bean Classification System</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Compare five machine learning classification models on the
    Dry Bean dataset using Accuracy, AUC, Precision, Recall,
    F1 Score and MCC.
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================================
# MODEL FILES
# ==========================================================

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "kNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Random Forest": "model/random_forest.joblib"
}


# ==========================================================
# SIDEBAR
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
# FILE UPLOAD
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
# NO FILE UPLOADED
# ==========================================================

if uploaded_file is None:

    st.info(
        "Please upload test_data.csv to begin model evaluation."
    )

    st.markdown(
        """
        ### How to use this application

        1. Upload the `test_data.csv` file.
        2. Select a machine learning model.
        3. View the six evaluation metrics.
        4. Examine the confusion matrix.
        5. Review the classification report.
        6. Compare all models in the comparison table.
        """
    )

    st.stop()


# ==========================================================
# READ CSV
# ==========================================================

try:

    test_data = pd.read_csv(uploaded_file)

except Exception as error:

    st.error(
        f"Unable to read the uploaded CSV file: {error}"
    )

    st.stop()


# ==========================================================
# VALIDATE TARGET COLUMN
# ==========================================================

if "Class" not in test_data.columns:

    st.error(
        "The uploaded CSV must contain a 'Class' column."
    )

    st.stop()


# ==========================================================
# DISPLAY DATA INFORMATION
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
# SEPARATE FEATURES AND TARGET
# ==========================================================

X_test = test_data.drop(
    columns=["Class"]
)

# Keep original target for display
y_test_original = test_data["Class"]


# ==========================================================
# LOAD SELECTED MODEL
# ==========================================================

model_path = MODEL_FILES[selected_model]


if not os.path.exists(model_path):

    st.error(
        f"Model file not found: {model_path}"
    )

    st.stop()


try:

    model = joblib.load(model_path)

except Exception as error:

    st.error(
        f"Unable to load model: {error}"
    )

    st.stop()


# ==========================================================
# GENERATE PREDICTIONS
# ==========================================================

try:

    y_pred_original = model.predict(X_test)

    y_proba = model.predict_proba(X_test)

except Exception as error:

    st.error(
        f"Prediction failed: {error}"
    )

    st.stop()


# ==========================================================
# NORMALIZE CLASS LABELS
#
# IMPORTANT:
# Streamlit Cloud was receiving mixed float/string labels.
# We convert all labels to strings and then map them to
# integer IDs for metric calculation.
# ==========================================================

model_classes = list(model.classes_)

model_class_strings = [
    str(value) for value in model_classes
]

y_true_strings = [
    str(value) for value in y_test_original
]

y_pred_strings = [
    str(value) for value in y_pred_original
]


# Create a stable mapping from model class -> integer
class_to_id = {
    class_name: index
    for index, class_name in enumerate(model_class_strings)
}


# ==========================================================
# CHECK FOR UNKNOWN TARGET CLASSES
# ==========================================================

unknown_classes = sorted(
    set(y_true_strings) - set(class_to_id.keys())
)


if unknown_classes:

    st.error(
        "The uploaded test dataset contains class labels "
        "that were not present in the selected trained model: "
        + ", ".join(unknown_classes)
    )

    st.stop()


# ==========================================================
# CONVERT LABELS TO INTEGER IDs
# ==========================================================

y_true = np.array(
    [class_to_id[value] for value in y_true_strings],
    dtype=int
)

y_pred = np.array(
    [class_to_id[value] for value in y_pred_strings],
    dtype=int
)

# Integer class labels corresponding exactly to columns
# of model.predict_proba()
class_ids = np.arange(
    len(model_class_strings)
)


# ==========================================================
# CALCULATE METRICS
# ==========================================================

try:

    # Accuracy
    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    # Precision
    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # Recall
    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # F1
    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # MCC
    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )

    # ------------------------------------------------------
    # Multiclass AUC
    # ------------------------------------------------------

    # Dry Bean is a multiclass problem.
    # predict_proba columns follow model.classes_.
    #
    # We use integer class IDs so sklearn never has to
    # compare strings and floats.
    auc = roc_auc_score(
        y_true,
        y_proba,
        multi_class="ovr",
        average="weighted",
        labels=class_ids
    )

except Exception as error:

    st.error(
        f"Metric calculation failed: {error}"
    )

    st.stop()


# ==========================================================
# SELECTED MODEL RESULT
# ==========================================================

st.markdown(
    f'<div class="section-title">🤖 {selected_model} Results</div>',
    unsafe_allow_html=True
)


# ==========================================================
# METRIC CARDS - ROW 1
# ==========================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )


with col2:

    st.metric(
        "AUC",
        f"{auc:.4f}"
    )


with col3:

    st.metric(
        "Precision",
        f"{precision:.4f}"
    )


# ==========================================================
# METRIC CARDS - ROW 2
# ==========================================================

col4, col5, col6 = st.columns(3)


with col4:

    st.metric(
        "Recall",
        f"{recall:.4f}"
    )


with col5:

    st.metric(
        "F1 Score",
        f"{f1:.4f}"
    )


with col6:

    st.metric(
        "MCC",
        f"{mcc:.4f}"
    )


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

st.markdown(
    '<div class="section-title">📊 Confusion Matrix</div>',
    unsafe_allow_html=True
)


# Use model class names so all seven Dry Bean classes
# are represented consistently.
class_labels = model_class_strings


cm = confusion_matrix(
    y_true,
    y_pred,
    labels=class_ids
)


cm_df = pd.DataFrame(
    cm,
    index=class_labels,
    columns=class_labels
)


fig, ax = plt.subplots(
    figsize=(10, 7)
)


sns.heatmap(
    cm_df,
    annot=True,
    fmt="d",
    cmap="Greens",
    linewidths=0.5,
    linecolor="white",
    ax=ax
)


ax.set_xlabel(
    "Predicted Class"
)

ax.set_ylabel(
    "Actual Class"
)

ax.set_title(
    f"Confusion Matrix - {selected_model}"
)


plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)


st.pyplot(
    fig,
    use_container_width=False
)

plt.close(fig)


# ==========================================================
# CLASSIFICATION REPORT
# ==========================================================

st.markdown(
    '<div class="section-title">📋 Classification Report</div>',
    unsafe_allow_html=True
)


report = classification_report(
    y_true,
    y_pred,
    labels=class_ids,
    target_names=class_labels,
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


st.dataframe(
    report_df.round(4),
    use_container_width=True
)


# ==========================================================
# MODEL COMPARISON
# ==========================================================

st.markdown(
    '<div class="section-title">🏆 Model Comparison</div>',
    unsafe_allow_html=True
)


comparison_file = (
    "results/model_comparison.csv"
)


if os.path.exists(comparison_file):

    try:

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

            if column in comparison_df.columns:

                comparison_df[column] = (
                    pd.to_numeric(
                        comparison_df[column],
                        errors="coerce"
                    ).round(4)
                )


        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )


        # --------------------------------------------------
        # Winner
        # --------------------------------------------------

        if "F1" in comparison_df.columns:

            winner_row = comparison_df.loc[
                comparison_df["F1"].idxmax()
            ]

            winner_name = winner_row[
                "ML Model"
            ]

            winner_f1 = winner_row[
                "F1"
            ]

            st.success(
                f"🏆 Overall Winner based on F1 Score: "
                f"{winner_name} "
                f"({winner_f1:.4f})"
            )


    except Exception as error:

        st.warning(
            f"Unable to display model comparison: {error}"
        )


else:

    st.warning(
        "Model comparison file is not available."
    )


# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.caption(
    "Dry Bean Classification | Machine Learning Assignment"
)
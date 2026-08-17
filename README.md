# 🌱 Dry Bean Classification using Machine Learning

## 1. Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models for identifying different varieties of dry beans based on their measured physical characteristics.

The project implements five classification algorithms and evaluates their performance using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also developed to allow users to upload test data, select a machine learning model, and view its evaluation results.

---

## 2. Dataset Description

The dataset used for this project is the **Dry Bean Dataset**, a multiclass classification dataset containing measurements of dry bean samples.

The dataset contains:

- **Instances:** 13,611
- **Features:** 16 numerical features
- **Target:** Class
- **Number of classes:** 7

The seven bean classes are:

1. Seker
2. Barbunya
3. Bombay
4. Cali
5. Dermosan
6. Horoz
7. Sira

The features describe physical properties of the bean samples such as area, perimeter, major axis length, minor axis length, aspect ratio, eccentricity, convex area, equivalent diameter, extent, solidity, roundness, compactness, shape factors, and related measurements.

The dataset was obtained from the publicly available Dry Bean Dataset.

---

## 3. GitHub Repository Link

**GitHub Repository:**

https://github.com/DhanalakshmiMtech/drybean-classification

The repository contains:

- `app.py`
- `train_models.py`
- `requirements.txt`
- `test_data.csv`
- Saved model files
- Evaluation results
- `.gitignore`

---

## 4. Models Used

The following five classification algorithms were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

### Evaluation Metrics

The following metrics were calculated for each model:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9291 | 0.9948 | 0.9291 | 0.9291 | 0.9290 | 0.9143 |
| Decision Tree | 0.9027 | 0.9405 | 0.9023 | 0.9027 | 0.9024 | 0.8824 |
| kNN | 0.9262 | 0.9879 | 0.9264 | 0.9262 | 0.9260 | 0.9108 |
| Naive Bayes | 0.9012 | 0.9917 | 0.9016 | 0.9012 | 0.9010 | 0.8809 |
| Random Forest (Ensemble) | 0.9269 | 0.9940 | 0.9268 | 0.9269 | 0.9266 | 0.9117 |

---

## 5. Observations on Model Performance

### Logistic Regression

Logistic Regression achieved the highest overall performance among the implemented models. It obtained an accuracy of 0.9291 and an F1 score of 0.9290. Its AUC of 0.9948 also indicates strong class discrimination.

### Decision Tree

The Decision Tree achieved an accuracy of 0.9027 and an F1 score of 0.9024. Its performance was lower than the other stronger models, although it still provided reasonable classification results.

### kNN

The kNN classifier performed strongly, achieving an accuracy of 0.9262 and an F1 score of 0.9260. Its AUC of 0.9879 indicates good discrimination between the bean classes.

### Naive Bayes

Naive Bayes achieved an accuracy of 0.9012 and an F1 score of 0.9010. Although its AUC was high at 0.9917, its accuracy, precision, recall, and F1 score were lower than the leading models.

### Random Forest

Random Forest achieved an accuracy of 0.9269 and an F1 score of 0.9266. It performed very close to Logistic Regression and kNN and demonstrated strong overall classification performance.

---

## 6. Overall Winner

### 🏆 Logistic Regression

Logistic Regression was selected as the overall winner based on the F1 Score.

**Best F1 Score: 0.9290**

It also achieved the highest accuracy among the five implemented models at **0.9291**.

Therefore, Logistic Regression provided the best overall balance between precision and recall for this Dry Bean classification experiment.

---

## 7. Streamlit Application

An interactive Streamlit web application was developed for model evaluation.

The application provides:

- CSV test-data upload
- Machine learning model selection
- Accuracy display
- AUC display
- Precision display
- Recall display
- F1 Score display
- MCC display
- Confusion matrix heatmap
- Classification report
- Model comparison table

The application uses the saved trained models from the `model` directory.

---

## 8. Project Structure

```text
drybean-classification/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── .gitignore
│
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── random_forest.joblib
│
└── results/
    ├── model_comparison.csv
    ├── logistic_regression_confusion_matrix.csv
    ├── decision_tree_confusion_matrix.csv
    ├── knn_confusion_matrix.csv
    ├── naive_bayes_confusion_matrix.csv
    └── random_forest_confusion_matrix.csv
    9. Technologies Used
Python
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn
Joblib
Streamlit
10. How to Run the Application Locally

Install the required packages:

pip install -r requirements.txt

Run the Streamlit application:

streamlit run app.py

Then open the local Streamlit URL in a web browser.

Upload test_data.csv and select a model from the sidebar to view its performance.

11. Deployment

The application is intended to be deployed using Streamlit Community Cloud.

Live application link:

To be updated after deployment.

12. Conclusion

The experiment shows that all five models were able to classify the Dry Bean classes with good performance.

Among the tested models, Logistic Regression produced the strongest overall results based on accuracy and F1 score. Random Forest and kNN also performed competitively, while Decision Tree and Naive Bayes produced comparatively lower accuracy and F1 scores.

The Streamlit application provides an interactive interface for evaluating the trained models using the test dataset.
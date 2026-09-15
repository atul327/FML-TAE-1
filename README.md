# Loan Approval Prediction using Machine Learning

## 📌 Project Overview

This project focuses on predicting whether a loan application will be **Approved or Rejected** using Machine Learning classification algorithms.

The project follows a complete machine learning workflow starting from **Exploratory Data Analysis (EDA)** and data preprocessing to **model training, cross-validation, model comparison, and final evaluation**.

The main objective is to understand how different classification algorithms perform on the same dataset and select a suitable model based on cross-validation performance.

---

## 🎯 Problem Statement

Loan approval depends on several factors such as:

* Applicant income
* Coapplicant income
* Credit history
* Education
* Loan amount
* Loan term
* Property area
* Marital status
* Employment status

The goal of this project is to use these applicant details to predict:

```text
Loan Approved (Y) → 1
Loan Rejected (N) → 0
```

This is a **Supervised Learning → Classification** problem.

---

## 📊 Dataset

The dataset contains **614 loan applications** with information about applicants.

### Features

| Feature           | Description                        |
| ----------------- | ---------------------------------- |
| Gender            | Applicant gender                   |
| Married           | Marital status                     |
| Dependents        | Number of dependents               |
| Education         | Education level                    |
| Self_Employed     | Self-employment status             |
| ApplicantIncome   | Applicant's income                 |
| CoapplicantIncome | Coapplicant's income               |
| LoanAmount        | Requested loan amount              |
| Loan_Amount_Term  | Loan repayment term                |
| Credit_History    | Credit history status              |
| Property_Area     | Area where the property is located |

### Target

**Loan_Status**

* `Y` → Loan Approved
* `N` → Loan Rejected

---

## 🔎 Exploratory Data Analysis

Before building the models, the dataset was explored to understand its structure and distribution.

The following checks were performed:

* Dataset shape
* Column names
* Data types
* Missing values
* Duplicate records
* Statistical summary
* Numerical feature distributions
* Categorical feature distributions
* Target distribution
* Potential outliers

### Visualizations

The project uses:

* Histograms
* Boxplots
* Countplots
* Scatterplots
* Correlation heatmap

These visualizations help understand the data before applying machine learning algorithms.

---

## 🧹 Data Preprocessing

### 1. Handling Missing Values

Missing categorical values were filled using the **mode**.

Example:

```python
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
```

Missing numerical values were filled using the **median**.

Example:

```python
df["LoanAmount"] = df["LoanAmount"].fillna(
    df["LoanAmount"].median()
)
```

### Why?

Missing values can cause problems during model training. The filling strategy depends on the type and distribution of the feature.

---

### 2. Removing Unnecessary Column

`Loan_ID` was removed because it is an identifier and does not provide useful predictive information.

```python
df = df.drop("Loan_ID", axis=1)
```

---

### 3. Encoding Categorical Variables

Categorical variables were converted into numerical form.

For example:

```text
Male → 1
Female → 0
```

The target variable was converted as:

```text
Y → 1
N → 0
```

`Property_Area` was converted using **One-Hot Encoding**.

```python
df = pd.get_dummies(
    df,
    columns=["Property_Area"],
    drop_first=True,
    dtype=int
)
```

---

## 📐 Feature and Target Separation

The input features were separated from the target variable.

```python
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
```

Where:

* `X` = Input features
* `y` = Target variable

The resulting feature matrix contains **614 samples and 12 features**.

---

## ✂️ Train-Test Split

The dataset was divided into training and testing sets using an **80:20 split**.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Why?

The training data is used to learn patterns, while the test data is kept separate to evaluate how well the trained model performs on unseen data.

`stratify=y` helps maintain a similar class distribution in both training and testing sets.

---

## ⚖️ Feature Scaling

Standardization was used for models that are sensitive to feature scale.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

For cross-validation, scaling was implemented inside a **Pipeline** to avoid data leakage.

---

## 🤖 Machine Learning Models

Four classification algorithms were compared:

### 1. Logistic Regression

A linear classification algorithm used as a strong baseline for binary classification problems.

### 2. Decision Tree

Creates decision rules by splitting the data based on feature values.

### 3. Random Forest

An ensemble of multiple decision trees that combines their predictions.

### 4. Support Vector Machine (SVM)

Finds a decision boundary that separates different classes.

---

## 🔄 Cross Validation

**5-Fold Cross Validation** was used to compare the models.

Instead of relying on a single train-validation split, the training data was divided into five parts.

```text
Fold 1 → Validation
Fold 2 → Training
Fold 3 → Training
Fold 4 → Training
Fold 5 → Training
```

This process is repeated so that each fold gets a chance to act as validation data.

The five scores are then averaged.

### Why Cross Validation?

Cross-validation provides a more reliable estimate of model performance and helps compare models using multiple validation splits.

---

## 📈 Cross Validation Results

| Model               | Mean CV Accuracy |
| ------------------- | ---------------: |
| Logistic Regression |       **79.84%** |
| SVM                 |       **79.64%** |
| Random Forest       |       **77.40%** |
| Decision Tree       |       **70.88%** |

### Best Model

Based on the mean 5-fold cross-validation accuracy:

**Logistic Regression performed best with 79.84% mean CV accuracy.**

SVM was very close with 79.64%.

---

## 🏆 Final Model Evaluation

After comparing the models using cross-validation, Logistic Regression was selected for final evaluation on the unseen test set.

### Test Accuracy

**86.18%**

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score

---

## 🔲 Confusion Matrix

The confusion matrix obtained on the test set was:

```text
[[22 16]
 [ 1 84]]
```

Interpretation:

|                 | Predicted Rejected | Predicted Approved |
| --------------- | -----------------: | -----------------: |
| Actual Rejected |                 22 |                 16 |
| Actual Approved |                  1 |                 84 |

Where:

* **TN = 22**
* **FP = 16**
* **FN = 1**
* **TP = 84**

The model correctly identified most approved loan applications, while some rejected applications were predicted as approved.

---

## 📋 Classification Report

```text
              precision    recall  f1-score   support

           0       0.96      0.58      0.72        38
           1       0.84      0.99      0.91        85

    accuracy                           0.86       123
   macro avg       0.90      0.78      0.81       123
weighted avg       0.88      0.86      0.85       123
```

The model achieved strong recall for approved loans, but recall for rejected loans was comparatively lower.

This also shows why **accuracy alone should not always be the only metric used for evaluating a classification model**.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 📁 Project Structure

```text
Loan-Approval-ML/
│
├── loan application.csv
├── FML_TAE_1.ipynb
└── README.md
```

> If the dataset license does not allow redistribution, do not upload `loan.csv`. Instead, provide the original dataset source in the project documentation.

---

## 📦 Installation

Clone the repository:

```bash
git clone <your-repository-url>
```

Move into the project directory:

```bash
cd Loan-Approval-ML
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```text
loan_approval_prediction.ipynb
```

---

## 📄 requirements.txt

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

---

## 🚀 ML Workflow

```text
Dataset
   ↓
Data Understanding
   ↓
Exploratory Data Analysis
   ↓
Data Cleaning
   ↓
Feature Encoding
   ↓
Feature / Target Separation
   ↓
Train-Test Split
   ↓
Feature Scaling
   ↓
Model Training
   ↓
5-Fold Cross Validation
   ↓
Model Comparison
   ↓
Best Model Selection
   ↓
Test Set Evaluation
```

---

## 🔮 Future Improvements

The project can be extended further with:

* Hyperparameter tuning
* GridSearchCV
* RandomizedSearchCV
* ROC-AUC evaluation
* Precision-Recall analysis
* Handling class imbalance
* Feature engineering
* Feature selection
* Model serialization using Joblib
* FastAPI prediction API
* Dockerization
* MLflow experiment tracking
* Cloud deployment
* Model monitoring

---

## 📌 Conclusion

This project demonstrates a complete basic machine learning classification workflow for loan approval prediction.

Four classification models were compared using **5-fold cross-validation**. Logistic Regression achieved the highest mean cross-validation accuracy of **79.84%** and achieved **86.18% accuracy on the unseen test set**.

The project also demonstrates the importance of EDA, data preprocessing, cross-validation, and multiple evaluation metrics when building and comparing machine learning models.

---

## 👨‍💻 Author

**Atul N. Patle**

B.Tech – Computer Science & Engineering (AI & ML)

GitHub: `https://github.com/atul327`

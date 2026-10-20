# Customer Purchase Prediction Using Machine Learning

An end-to-end Machine Learning project that predicts whether an online shopping session will result in a purchase based on browsing behavior, engagement metrics, and visitor information.

The project includes exploratory data analysis, data preprocessing, model training, hyperparameter tuning, feature importance analysis, and deployment using Streamlit.

**Live Application:** https://customerpurchaseprediction.streamlit.app/

## Project Overview

Understanding online shopping behavior can help e-commerce businesses identify sessions with higher purchase potential and improve their understanding of customer engagement.

This project aims to:

- Predict whether an online shopping session will result in a purchase.
- Analyze relationships between browsing activity and purchase outcomes.
- Compare machine learning classification algorithms.
- Improve model performance using cross-validation and hyperparameter tuning.
- Deploy an interactive web application for generating predictions.

## Dataset

**Dataset:** Online Shoppers Purchasing Intention Dataset

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

| Property | Value |
|---|---|
| Total Records | 12,330 |
| Input Features | 17 |
| Target Variable | Revenue |
| Problem Type | Binary Classification |
| Purchase Sessions | 1,908 (15.5%) |
| Non-Purchase Sessions | 10,422 (84.5%) |
| Missing Values | None |

### Features

**Browsing Activity**
- Administrative and Administrative_Duration
- Informational and Informational_Duration
- ProductRelated and ProductRelated_Duration

**Engagement Metrics**
- BounceRates
- ExitRates
- PageValues
- SpecialDay

**Visitor Information**
- Month
- OperatingSystems
- Browser
- Region
- TrafficType
- VisitorType
- Weekend

**Target Variable:** `Revenue`

- `True`: Purchase completed
- `False`: No purchase

## Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Optimization | GridSearchCV, StratifiedKFold |
| Model Serialization | Joblib |
| Web Framework | Streamlit |
| Development Tools | Google Colab, VS Code, GitHub |

## Project Workflow

### 1. Exploratory Data Analysis

- Examined dataset structure, distributions, and data types.
- Checked for missing values and duplicate records.
- Analyzed the distribution of purchasing and non-purchasing sessions.
- Investigated correlations between numerical features and purchase outcomes.
- Examined customer browsing patterns and visitor characteristics.

### 2. Data Preprocessing

- Separated input features and target variable.
- Performed an 80:20 stratified train-test split.
- Applied One-Hot Encoding to categorical variables.
- Standardized numerical features using StandardScaler.
- Used ColumnTransformer and Pipeline for consistent preprocessing.
- Integrated preprocessing into cross-validation to prevent data leakage.

### 3. Model Training

Implemented and compared:

- Logistic Regression
- Logistic Regression with balanced class weights
- Random Forest Classifier

Used five-fold stratified cross-validation with F1-score as the primary model-selection metric.

### 4. Hyperparameter Tuning

Used GridSearchCV to optimize the Random Forest classifier.

**Best Hyperparameters:**

| Hyperparameter | Value |
|---|---|
| n_estimators | 150 |
| max_depth | None |
| min_samples_split | 2 |

**Best Cross-Validation F1-score:** 0.6281

## Model Performance

### Tuned Random Forest — Test Results

| Metric | Score |
|---|---:|
| Accuracy | 89.50% |
| Precision | 74.70% |
| Recall | 48.69% |
| F1-score | 58.95% |

### Model Comparison

| Metric | Balanced Logistic Regression | Tuned Random Forest |
|---|---:|---:|
| Accuracy | 84.10% | 89.50% |
| Precision | 49.13% | 74.70% |
| Recall | 74.35% | 48.69% |
| F1-score | 59.17% | 58.95% |

The tuned Random Forest was selected for deployment based on its higher accuracy and precision.

Balanced Logistic Regression achieved higher recall, demonstrating the trade-off between identifying more purchasing sessions and minimizing false positives.

### Confusion Matrix — Tuned Random Forest

| | Predicted No Purchase | Predicted Purchase |
|---|---:|---:|
| Actual No Purchase | 2021 | 63 |
| Actual Purchase | 196 | 186 |

The model correctly identified 186 purchasing sessions while missing 196 actual purchases.

## Feature Importance

Random Forest feature importance analysis identified the following influential features:

| Feature | Importance |
|---|---:|
| PageValues | 32.27% |
| ExitRates | 8.22% |
| ProductRelated_Duration | 7.93% |
| ProductRelated | 7.20% |
| Administrative_Duration | 5.36% |

`PageValues` had the highest importance score in the trained Random Forest model.

Feature importance reflects contributions to model predictions and does not establish causal relationships.

## Streamlit Web Application

The trained model is deployed as an interactive Streamlit application.

**Live Demo:** [Customer Purchase Prediction](https://customerpurchaseprediction.streamlit.app/)

### Application Features

- Interactive inputs for all 17 dataset features.
- Separate sections for browsing activity, engagement metrics, and visitor information.
- Dropdown menus for categorical inputs.
- Predictions generated using the saved Random Forest pipeline.
- Clear purchase or no-purchase prediction results.

## Project Structure

```text
customer-purchase-prediction/
│
├── Customer_Purchase_Prediction.ipynb
├── app.py
├── customer_purchase_prediction.pkl
├── online_shoppers_intention.csv
├── requirements.txt
├── README.md
└── LICENSE
```

## Installation and Usage

### 1. Clone the Repository

```bash
git clone https://github.com/Radhika-238/customer-purchase-prediction.git
cd customer-purchase-prediction
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment.

**macOS/Linux:**

```bash
source .venv/bin/activate
```

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open locally at:

`http://localhost:8501`

Alternatively, access the deployed application:

https://customerpurchaseprediction.streamlit.app/

## Key Learnings

This project provided practical experience in:

- Exploratory data analysis and feature preprocessing.
- Working with numerical and categorical data.
- Handling imbalanced classification problems.
- Building reproducible preprocessing pipelines.
- Training and comparing classification algorithms.
- Evaluating models using precision, recall, F1-score, and confusion matrices.
- Applying stratified cross-validation.
- Performing hyperparameter tuning using GridSearchCV.
- Interpreting Random Forest feature importance.
- Saving and loading trained models using Joblib.
- Developing and deploying an interactive ML application using Streamlit.

## Limitations

- The dataset is imbalanced, which makes identifying purchasing sessions more challenging.
- The deployed model has relatively high precision but moderate recall.
- Certain features, such as PageValues, may not be available until later in a shopping session.
- Predictions depend on the availability and quality of session-level data.
- Model predictions identify statistical patterns rather than causal relationships.

## Future Improvements

- Experiment with XGBoost and other ensemble algorithms.
- Optimize classification thresholds to improve the precision-recall trade-off.
- Explore permutation importance and additional explainability techniques.
- Investigate alternative feature subsets for earlier purchase prediction.
- Improve the Streamlit interface with prediction probabilities and visualizations.
- Evaluate model performance on additional datasets.

## Author

**Radhika Agarwal**

B.Tech — Computer Science and Engineering

**GitHub:** [Radhika-238](https://github.com/Radhika-238)

**Live Project:** [Customer Purchase Prediction](https://customerpurchaseprediction.streamlit.app/)

---

This project demonstrates an end-to-end machine learning workflow, from dataset exploration and model development to deployment as an interactive web application.

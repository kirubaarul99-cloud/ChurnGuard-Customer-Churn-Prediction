# ChurnGuard — Customer Churn Prediction

## 📌 Project Overview

ChurnGuard is a beginner-friendly machine learning project that explores customer churn data, cleans common data-quality issues, trains a Logistic Regression model, and provides an interactive script for predicting whether a customer is likely to churn.

This project was completed as part of an AI/ML learning project.

## 🎯 Problem Statement

A telecom company wants to identify customers who may cancel their subscription. Before building a machine learning model, the raw customer data needs to be explored and cleaned.

The project covers:

1. Loading and exploring the raw dataset
2. Cleaning inconsistent and invalid data
3. Encoding categorical variables
4. Training a Logistic Regression model
5. Evaluating the model
6. Building an interactive prediction script

## 📊 Dataset

The dataset contains customer information such as:

- Gender
- Senior citizen status
- Tenure
- Phone service
- Internet service
- Contract type
- Paperless billing
- Payment method
- Monthly charges
- Total charges
- Churn

The raw dataset intentionally contains data-quality issues such as inconsistent casing, spelling variations, missing values, duplicate rows, and invalid numeric values.

## 🧹 Data Cleaning

The project applies the following cleaning steps:

- Removed `customerID`
- Removed duplicate rows
- Stripped unnecessary whitespace
- Standardised text casing
- Standardised `Contract` values
- Standardised `InternetService` values
- Converted `TotalCharges` to numeric
- Removed invalid tenure values
- Removed invalid MonthlyCharges values
- Filled missing numeric values using mean/median as required
- Filled missing InternetService values

After cleaning, the dataset contained **867 rows and 11 columns**, with no remaining missing values.

## 🤖 Machine Learning Model

The model used is:

**Logistic Regression**

### Target Encoding

- `Yes` → `1`
- `No` → `0`

### Categorical Encoding

Categorical features were converted using:

```python
pd.get_dummies(..., drop_first=True)
```

### Train/Test Split

- Training data: 80%
- Testing data: 20%
- `random_state=42`

## 📈 Model Evaluation

The Logistic Regression model achieved approximately:

**Accuracy: 70.11%**

The test-set classification report was also generated for the `Stay` and `Churn` classes.

The result is a learning-project baseline rather than a production-ready churn system.

## 🔮 Interactive Prediction

The final script retrains Logistic Regression using the full cleaned dataset and accepts five customer inputs:

- Tenure
- Monthly Charges
- Total Charges
- Senior Citizen status
- Contract type

The contract input is represented as:

- `0` = Month-to-month
- `1` = One year
- `2` = Two year

The script then prints whether the customer is predicted to churn or stay.

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression
- Google Colab

## 📁 Project Structure

```text
ChurnGuard/
│
├── data/
│   └── churnguard_data.csv
│
├── src/
│   ├── task1_load_explore.py
│   ├── task2_clean_data.py
│   ├── task3_train_model.py
│   └── task4_predict.py
│
├── README.md
└── requirements.txt
```

## ▶️ How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Make sure the dataset is available as:

```text
data/churnguard_data.csv
```

Run the scripts from the project directory.

For the prediction script:

```bash
python src/task4_predict.py
```

The script will ask for customer details and return a churn/stay prediction.

## 💡 Key Learning Outcomes

Through this project, I practised:

- Exploratory data analysis
- Identifying data-quality issues
- Data cleaning with Pandas
- Handling missing values
- Encoding categorical data
- Train/test splitting
- Logistic Regression
- Model evaluation
- Building an interactive ML prediction script

## 👩‍💻 Author

**Kiru**

This project was created as part of my AI/ML learning journey.

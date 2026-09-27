import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("churnguard_data.csv")

df = df.drop("customerID", axis=1)
df = df.drop_duplicates()
df["gender"] = df["gender"].str.strip()
df["PaymentMethod"] = df["PaymentMethod"].str.strip()
df["Churn"] = df["Churn"].str.strip().str.title()
df["PhoneService"] = df["PhoneService"].str.strip().str.title()
df["PaperlessBilling"] = df["PaperlessBilling"].str.strip().str.title()

df["Contract"] = (
    df["Contract"].str.strip().str.lower().replace({
        "month-to-month": "Month-to-month",
        "month to month": "Month-to-month",
        "monthly": "Month-to-month",
        "one year": "One year",
        "1 year": "One year",
        "two year": "Two year",
        "2 year": "Two year"
    })
)

df["InternetService"] = (
    df["InternetService"].str.strip().str.lower().replace({
        "dsl": "DSL",
        "fiber optic": "Fiber optic",
        "fibre optic": "Fiber optic",
        "fiberoptic": "Fiber optic",
        "no": "No"
    })
)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
df = df[df["tenure"] > 0]
df = df[(df["MonthlyCharges"] >= 10) & (df["MonthlyCharges"] <= 200)]
df["MonthlyCharges"] = df["MonthlyCharges"].fillna(df["MonthlyCharges"].mean())
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean())
df["tenure"] = df["tenure"].fillna(round(df["tenure"].median()))
df["InternetService"] = df["InternetService"].fillna("No")

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

categorical_columns = [
    "gender", "PhoneService", "InternetService",
    "Contract", "PaperlessBilling", "PaymentMethod"
]

df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, target_names=["Stay", "Churn"]
))

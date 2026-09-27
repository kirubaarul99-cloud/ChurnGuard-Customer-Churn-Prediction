import pandas as pd
from sklearn.linear_model import LogisticRegression

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
df["Contract"] = df["Contract"].map({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
})

features = [
    "tenure", "MonthlyCharges", "TotalCharges",
    "SeniorCitizen", "Contract"
]

X = df[features]
y = df["Churn"]

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

tenure = int(input("Enter tenure (months): "))
monthly_charges = float(input("Enter Monthly Charges: "))
total_charges = float(input("Enter Total Charges: "))
senior_citizen = int(input("Senior Citizen? (1 = Yes, 0 = No): "))
contract = int(input(
    "Contract type (0 = Month-to-month, 1 = One year, 2 = Two year): "
))

customer = pd.DataFrame({
    "tenure": [tenure],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges],
    "SeniorCitizen": [senior_citizen],
    "Contract": [contract]
})

prediction = model.predict(customer)[0]

if prediction == 1:
    print("Prediction: This customer is likely to CHURN.")
else:
    print("Prediction: This customer is likely to STAY.")

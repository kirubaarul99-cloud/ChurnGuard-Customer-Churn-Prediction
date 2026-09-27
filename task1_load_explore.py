import pandas as pd

df = pd.read_csv("churnguard_data.csv")

print("Dataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Information:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nNumber of Duplicate Rows:")
print(df.duplicated().sum())

print("\nChurn Value Counts:")
print(df["Churn"].value_counts())

print("\nUnique Contract Values:")
print(df["Contract"].unique())

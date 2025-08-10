import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("employee_attrition_dataset.csv")

# Encode categorical variables
le_role = LabelEncoder()
le_salary = LabelEncoder()
le_attrition = LabelEncoder()

df["JobRole"] = le_role.fit_transform(df["JobRole"])
df["SalaryLevel"] = le_salary.fit_transform(df["SalaryLevel"])
df["Attrition"] = le_attrition.fit_transform(df["Attrition"])

# Features and target
X = df.drop("Attrition", axis=1)
y = df["Attrition"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print(f"✅ Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# ---- Test with manual JSON ----
sample_employee = {
    "YearsAtCompany": 15,
    "JobRole": "HR",
    "SalaryLevel": "High",
    "OvertimeHours": 2,
    "WorkLifeBalance": 5
}

sample_df = pd.DataFrame([sample_employee])
sample_df["JobRole"] = le_role.transform(sample_df["JobRole"])
sample_df["SalaryLevel"] = le_salary.transform(sample_df["SalaryLevel"])

prediction = model.predict(sample_df)[0]
print(f"Prediction: {le_attrition.inverse_transform([prediction])[0]}")

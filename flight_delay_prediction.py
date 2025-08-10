import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
df = pd.read_csv("flight_delays_strong.csv")

# Encode categorical features
le_airline = LabelEncoder()
le_weather = LabelEncoder()

df["Airline"] = le_airline.fit_transform(df["Airline"])
df["Weather"] = le_weather.fit_transform(df["Weather"])

# Split data
X = df[["Airline", "Distance", "DepartureHour", "Weather", "DayOfWeek"]]
y = df["Delayed"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train strong model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Accuracy
print(f"✅ Model Accuracy: {model.score(X_test, y_test):.2f}")

# ---- Manual test ----
sample_flight = {
    "Airline": "AirlineD",
    "Distance": 1800,
    "DepartureHour": 15,
    "Weather": "Clear",
    "DayOfWeek": 4
}

sample_df = pd.DataFrame([sample_flight])
sample_df["Airline"] = le_airline.transform(sample_df["Airline"])
sample_df["Weather"] = le_weather.transform(sample_df["Weather"])

prediction = model.predict(sample_df)[0]
print(f"Sample Flight: {sample_flight}")
print(f"Prediction: {'Delayed' if prediction == 1 else 'On Time'}")

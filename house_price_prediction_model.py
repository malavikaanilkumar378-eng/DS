import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("house_prices_dataset.csv")

# Features & Target
X = df.drop("Price", axis=1)
y = df["Price"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

sample_house = {
    "Rooms": 2,
    "Bathrooms": 1,
    "LandSize": 50,
    "BuildingArea": 80,
    "YearBuilt": 2008,
    "DistanceToCity": 12
}

sample_df = pd.DataFrame([sample_house])

# Predict
predicted_price = model.predict(sample_df)[0]
print("\nSample House:", sample_house)
print(f"Predicted Price: ${predicted_price:,.2f}")


similar_houses = df[
    (df["Rooms"] == sample_house["Rooms"]) &
    (df["Bathrooms"] == sample_house["Bathrooms"]) &
    (abs(df["BuildingArea"] - sample_house["BuildingArea"]) < 20)
]
print("Compared to real market trends for similar houses:")
print(similar_houses.to_string(index=False))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import shap



# Load data
df = pd.read_csv('heart_disease_data.csv')
X = df.drop('target', axis=1)
y = df['target']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model   
clf = RandomForestClassifier(class_weight='balanced', random_state=42)
clf.fit(X_train_scaled, y_train)

rf_explainer = shap.TreeExplainer(clf)
shap_values = rf_explainer.shap_values(X_train_scaled)


sample =[[60, 1, 2, 140, 185, 0, 0, 155, 0, 3.0, 1, 0, 2]]
sample_scaled = scaler.transform(sample)
prediction = clf.predict(sample_scaled)[0]
proba = clf.predict_proba(sample_scaled)

print("Prediction:", prediction)
print(f"Probability of No Risk (0): {proba[0][0]:.4f}")
print(f"Probability of Risk (1): {proba[0][1]:.4f}")


import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/weather.csv"
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "weather_model.pkl")

df = pd.read_csv(DATA_PATH)

features = [
    "Temperature_C",
    "Humidity_pct",
    "WindSpeed_kmh",
    "Pressure_hPa",
    "CloudCover_pct"
]
target = "Weather"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)
model.fit(X_train, y_train)

pred = model.predict(X_test)
accuracy = accuracy_score(y_test, pred)

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred))

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved to: {MODEL_PATH}")

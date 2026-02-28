import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("Students Social Media Addiction.csv")

# Keep only behavioral predictors
df = df[[
    "Avg_Daily_Usage_Hours",
    "Sleep_Hours_Per_Night",
    "Affects_Academic_Performance",
    "Conflicts_Over_Social_Media",
    "Addicted_Score"
]]

# Convert Yes/No to numeric
df["Affects_Academic_Performance"] = df["Affects_Academic_Performance"].map({"Yes": 1, "No": 0})
df["Conflicts_Over_Social_Media"] = df["Conflicts_Over_Social_Media"].map({"Yes": 1, "No": 0})

# Define High Risk properly (top 25%)
df["High_Risk"] = (df["Addicted_Score"] >= 8).astype(int)

df = df.drop(columns=["Addicted_Score"])

X = df.drop("High_Risk", axis=1)
y = df["High_Risk"]

# Train model
model = RandomForestClassifier(
    max_depth=5,
    random_state=42
)

# Split data for evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# If Streamlit is available, display the accuracy there as well
try:
    import streamlit as st
    st.write(f"Model Accuracy: {round(accuracy * 100, 2)}%")
except Exception:
    pass

# Save
joblib.dump(model, "model.pkl")

print("✅ Model retrained WITHOUT mental health feature.")
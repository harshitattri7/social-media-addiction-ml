import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

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

model.fit(X, y)

# Save
joblib.dump(model, "model.pkl")

print("✅ Model retrained WITHOUT mental health feature.")
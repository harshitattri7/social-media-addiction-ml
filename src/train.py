import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("data/raw/Students Social Media Addiction.csv")

# Create target variable
df["High_Addiction"] = (df["Addicted_Score"] >= 7).astype(int)

# Drop unnecessary columns
X = df.drop(["Student_ID", "Addicted_Score", "High_Addiction", "Country"], axis=1)
y = df["High_Addiction"]

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=10,
    random_state=42
)

rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)

print("Train Accuracy:", rf.score(X_train, y_train))
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(rf, "random_forest_model.pkl")

print("\nModel saved successfully.")

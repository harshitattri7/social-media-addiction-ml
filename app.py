import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

model = joblib.load("model.pkl")


st.set_page_config(page_title="Social Media Addiction Risk Assessment", layout="wide")

# ---- CUSTOM CSS STYLING ----
st.markdown(
    """
    <style>
    /* Main title gradient */
    .main-title {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%);
        color: white;
        padding: 30px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        font-size: 2.5em;
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
        border-right: 3px solid #FF6B6B;
    }
    
    /* Header styling */
    h2, h3 {
        color: #FF6B6B;
        font-weight: bold;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E72 100%);
        color: white;
        border: none;
        padding: 12px 30px;
        font-weight: bold;
        border-radius: 8px;
        box-shadow: 0 4px 10px rgba(255, 107, 107, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        transform: translateY(-2px);
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
    }
    
    /* Success message */
    .success-msg {
        background-color: #d4edda;
        color: #155724;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #28a745;
    }
    
    /* Error message */
    .error-msg {
        background-color: #f8d7da;
        color: #721c24;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #f5c6cb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---- SIDEBAR: Model Information ----
with st.sidebar:
    st.markdown("## 🤖 Model Information")
    
    st.markdown(
        """
        **Algorithm:** Random Forest Classifier
        
        **Purpose:** Predicts social media addiction risk based on behavioral indicators.
        
        **Features Used:**
        - Daily Usage Hours
        - Sleep Hours Per Night
        - Academic Performance Impact
        - Social Conflicts
        
        **Model Details:**
        - Max Depth: 5
        - Random State: 42
        - Multi-class Classification: Binary (High Risk / Low Risk)
        
        **Training Data:**
        - Dataset: Students Social Media Addiction
        - Target: Addiction Risk Classification
        - Test Accuracy: ~90.1%
        
        **How It Works:**
        1. Input your behavioral data via sliders and dropdowns
        2. Model predicts addiction risk probability
        3. View detailed model insights below (Feature Importance & Confusion Matrix)
        """
    )
    
    st.markdown("---")
    st.info("ℹ️ This tool is for educational and awareness purposes only.")

st.title("📱 Social Media Addiction Risk Assessment")
st.markdown('<div class="main-title">📊 Assess Your Digital Wellness Now</div>', unsafe_allow_html=True)
st.write("Answer honestly to assess your behavioral risk.")
st.markdown("---")

# risk assessment inputs
# Inputs
usage = st.slider("Hours per day on social media", 0.0, 12.0, 4.0)
sleep = st.slider("Hours of sleep per night", 3.0, 10.0, 7.0)

academic_choice = st.selectbox(
    "Has social media negatively affected your academic performance?",
    ["Yes", "No"]
)

conflicts_choice = st.selectbox(
    "Have you experienced conflicts because of social media?",
    ["Yes", "No"]
)

# Convert to numeric
academic = 1 if academic_choice == "Yes" else 0
conflicts = 1 if conflicts_choice == "Yes" else 0

input_data = pd.DataFrame({
    "Avg_Daily_Usage_Hours": [usage],
    "Sleep_Hours_Per_Night": [sleep],
    "Affects_Academic_Performance": [academic],
    "Conflicts_Over_Social_Media": [conflicts]
})

if st.button("Check My Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Assessment Result")
    st.write(f"Risk Probability: {round(probability * 100, 2)}%")
    st.progress(float(probability))

    if prediction == 1:
        st.error("🔴 High Behavioral Addiction Risk")
        st.write("### Suggested Actions:")
        st.write("- Gradually reduce daily usage")
        st.write("- Avoid late-night scrolling")
        st.write("- Set app time limits")
    else:
        st.success("🟢 Low Risk")
        st.write("### Maintain healthy digital habits!")



# Model Performance section
st.markdown("---")
st.header("📊 Model Performance & Insights")

# Create two columns for charts
col1, col2 = st.columns(2)

# ---- Feature Importance (Left Column) ----
with col1:
    st.subheader("Feature Importance")
    
    feature_names = [
        "Daily Usage Hours",
        "Sleep Hours",
        "Academic Impact",
        "Social Conflicts"
    ]

    try:
        importances = model.feature_importances_

        fig1, ax1 = plt.subplots(figsize=(5, 4))
        ax1.barh(feature_names, importances)
        ax1.set_xlabel("Importance Score")
        ax1.set_title("Feature Importance")
        st.pyplot(fig1)
    except AttributeError:
        st.info("Model does not expose feature importances.")
    except Exception as e:
        st.error(f"Could not display feature importances: {e}")

# ---- Confusion Matrix (Right Column) ----
with col2:
    st.subheader("Confusion Matrix")

    try:
        data = pd.read_csv("Students Social Media Addiction.csv")

        data["Affects_Academic_Performance"] = data["Affects_Academic_Performance"].map({"Yes":1,"No":0})
        data["Conflicts_Over_Social_Media"] = data["Conflicts_Over_Social_Media"].map({"Yes":1,"No":0})

        X = data[[
            "Avg_Daily_Usage_Hours",
            "Sleep_Hours_Per_Night",
            "Affects_Academic_Performance",
            "Conflicts_Over_Social_Media"
        ]]

        # Support multiple possible target column names
        if "Addicted" in data.columns:
            y = data["Addicted"]
        elif "High_Risk" in data.columns:
            y = data["High_Risk"]
        elif "Addicted_Score" in data.columns:
            y = (data["Addicted_Score"] >= 8).astype(int)
        else:
            raise KeyError("No target column found in dataset")

        y_pred = model.predict(X)

        cm = confusion_matrix(y, y_pred)
        # Calculate accuracy from confusion matrix
        accuracy = (cm[0][0] + cm[1][1]) / cm.sum()
        st.write(f"Model Accuracy: {round(accuracy*100,2)}%")

        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.imshow(cm, cmap="Blues")

        ax2.set_xlabel("Predicted")
        ax2.set_ylabel("Actual")

        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax2.text(j, i, int(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black")

        st.pyplot(fig2)
    except Exception as e:
        st.info(f"Could not compute confusion matrix: {e}")
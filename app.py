import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

model = joblib.load("model.pkl")

st.set_page_config(page_title="Social Media Addiction Risk Assessment")

st.title("📱 Social Media Addiction Risk Assessment")
st.write("Answer honestly to assess your behavioral risk.")

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
    
        # Show model feature importances if available
        try:
            st.subheader("📊 Model Insights")

            feature_names = [
                "Daily Usage Hours",
                "Sleep Hours",
                "Academic Impact",
                "Social Conflicts"
            ]

            importances = model.feature_importances_

            fig, ax = plt.subplots()
            ax.barh(feature_names, importances)
            ax.set_xlabel("Importance Score")
            ax.set_title("Feature Importance")

            st.pyplot(fig)
        except AttributeError:
            st.info("Model does not expose feature importances.")
        except Exception as e:
            st.error(f"Could not display feature importances: {e}")
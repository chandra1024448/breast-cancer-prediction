import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Cache model and data (fast load)
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("src/breast_cancer_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("src/breast_cancer")  # fixed filename

model = load_model()
df = load_data()

# -------------------------------
# App Title
# -------------------------------
st.title("🔍 Breast Cancer Prediction App")

# -------------------------------
# Input Section
# -------------------------------
# Input patient code instead of index
patient_id_input = st.text_input("Enter Patient Code", "")

if st.button("Get Details"):
    if patient_id_input.strip() == "":
        st.error("Please enter a Patient Code.")
    else:
        try:
            # Convert to int
            patient_id = int(patient_id_input)

            # Check if patient ID exists
            if patient_id not in df['id'].values:
                st.error("Invalid Patient Code! Please enter a valid code from the dataset.")
            else:
                # Fetch patient row
                patient_data = df[df['id'] == patient_id].iloc[0]
                
                # Show details
                st.write("### Patient Details:")
                st.dataframe(patient_data.to_frame().T)
                
                # Prepare features for prediction
                if 'target' in df.columns:
                    features = patient_data.drop(['target', 'id'])
                else:
                    features = patient_data.drop(['id'])
                
                # Make prediction
                prediction = model.predict([features])[0]
                result = "Malignant" if prediction == 0 else "Benign"
                st.success(f"Prediction: {result}")
        except ValueError:
            st.error("Please enter a valid numeric Patient Code.")
import pickle
import streamlit as st
import numpy as np

# Load models
model_logistic = pickle.load(open('model_diabetes.sav', 'rb'))

# App Title
st.title("Diabetes Prediction Application")
st.subheader("Predict the likelihood of diabetes using health parameters.")

# Sidebar for model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.radio(
    "Choose a Prediction Model",
    ['Logistic Regression', 'Random Forest', 'Gaussian']
)

# Input fields in collapsible sections
with st.expander("Enter Patient Details"):
    Pregnancies = st.number_input('Number of Pregnancies', min_value=0, step=1)
    Glucose = st.number_input('Glucose Level', min_value=0.0)
    BloodPressure = st.number_input('Blood Pressure Level', min_value=0.0)
    SkinThickness = st.number_input('Skin Thickness', min_value=0.0)
    Insulin = st.number_input('Insulin Level', min_value=0.0)
    BMI = st.number_input('BMI (Body Mass Index)', min_value=0.0)
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function', min_value=0.0)
    Age = st.number_input('Age', min_value=0, step=1)

# Prediction Button and Output
if st.button('Predict Diabetes'):
    # Select the model
    if model_choice == 'Logistic Regression':
        model = model_logistic
    # elif model_choice == 'Random Forest':
        # model = model_random_forest
    # else:
        # model = model_gaussian

    # Prepare input data
    input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
    
    try:
        diabetes_prediction = model.predict(input_data)
        if diabetes_prediction[0] == 1:
            st.error("The patient is likely to have diabetes.")
        else:
            st.success("The patient is unlikely to have diabetes.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

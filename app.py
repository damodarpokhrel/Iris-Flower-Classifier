import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Set page config
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('iris_model.joblib')

# Load iris dataset for feature names
iris = load_iris()
feature_names = iris.feature_names
target_names = iris.target_names

# App title and description
st.title("🌸 Iris Flower Classifier")
st.write("""
This app predicts the type of Iris flower based on the following measurements:
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)
""")

# Create input fields
st.subheader("Enter Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

# Create a button for prediction
if st.button("Predict Iris Type"):
    try:
        # Load the model
        model = load_model()
        
        # Make prediction
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(features)
        probability = model.predict_proba(features)
        
        # Display results
        st.subheader("Prediction Results")
        st.success(f"Predicted Iris Type: {target_names[prediction[0]]}")
        
        # Display probability distribution
        st.write("Prediction Probabilities:")
        prob_df = pd.DataFrame({
            'Iris Type': target_names,
            'Probability': probability[0]
        })
        st.bar_chart(prob_df.set_index('Iris Type'))
        
        # Display feature importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': abs(model.coef_[0])
        })
        st.bar_chart(feature_importance.set_index('Feature'))
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please make sure the model file 'iris_model.joblib' exists in the same directory.")

# Add some information about Iris flowers
with st.expander("About Iris Flowers"):
    st.write("""
    The Iris dataset contains three species of Iris flowers:
    1. Iris Setosa
    2. Iris Versicolor
    3. Iris Virginica
    
    Each flower is characterized by four features:
    - Sepal length and width
    - Petal length and width
    
    These measurements are used to distinguish between the different species of Iris flowers.
    """) 
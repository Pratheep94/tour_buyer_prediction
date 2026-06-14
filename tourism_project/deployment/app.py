
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import os # Import os to access environment variables
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set MLflow tracking URI for model loading
# Use environment variable to dynamically set the tracking URI
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

st.set_page_config(page_title="Tour Buyer Prediction App", layout="wide")

st.title("🌴 Tour Buyer Prediction App")
st.markdown("This app predicts whether a customer will purchase a Wellness Tourism Package.")

# --- Load Model and Preprocessor ---
@st.cache_resource
def load_model_and_preprocessor():
    try:
        # Load the latest version of the registered model from MLflow Model Registry
        # Ensure MLflow tracking server is accessible or model artifacts are locally available
        model_uri = "models:/TourBuyerPredictionRFC/latest"
        model = mlflow.sklearn.load_model(model_uri)

        # In a real scenario, the preprocessor would also be logged and loaded via MLflow
        # For now, we'll re-create the preprocessor components as they were defined in data_preparation
        # It's crucial that the preprocessor structure matches exactly what was used for training.

        # Dummy data for preprocessor columns (should be same as training)
        # It is highly recommended to save the fitted preprocessor object itself via MLflow
        # and load it back. Reconstructing it like this is error-prone.
        sample_data_for_preprocessor = pd.DataFrame({
            'CustomerID': [200000], 'Age': [30.0], 'CityTier': [1], 'DurationOfPitch': [10.0],
            'NumberOfPersonVisiting': [2], 'NumberOfFollowups': [3.0], 'PreferredPropertyStar': [3.0],
            'NumberOfTrips': [1.0], 'Passport': [0], 'PitchSatisfactionScore': [3], 'OwnCar': [1],
            'NumberOfChildrenVisiting': [0.0], 'MonthlyIncome': [20000.0],
            'TypeofContact': ['Self Enquiry'], 'Occupation': ['Salaried'], 'Gender': ['Male'],
            'ProductPitched': ['Basic'], 'MaritalStatus': ['Married'], 'Designation': ['Executive']
        })

        # Corrected: 'Unnamed: 0' removed, 'Designation' added to categorical_cols
        numerical_cols = ['CustomerID', 'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting',
                          'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 'Passport',
                          'PitchSatisfactionScore', 'OwnCar', 'NumberOfChildrenVisiting', 'MonthlyIncome']
        categorical_cols = ['TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 'MaritalStatus', 'Designation']

        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ],
            remainder='passthrough'
        )

        # Fit preprocessor on dummy data to get feature names in correct order
        # THIS IS A CRITICAL ASSUMPTION AND SHOULD BE REPLACED BY LOADING A SAVED PREPROCESSOR
        preprocessor.fit(sample_data_for_preprocessor)

        return model, preprocessor, numerical_cols, categorical_cols
    except Exception as e:
        st.error(f"Error loading model or preprocessor: {e}")
        return None, None, None, None # Return None for all to avoid TypeError on unpack

model, preprocessor, numerical_cols, categorical_cols = load_model_and_preprocessor()

# Only proceed if model and preprocessor are loaded successfully
if model is None or preprocessor is None:
    st.warning("Model or preprocessor failed to load. Please check MLflow server and model registration.")
    st.stop() # Stop Streamlit app if components are not loaded

# --- Input Features ---
st.sidebar.header("Customer Input Features")

def user_input_features():
    # Using st.slider for numerical inputs and st.selectbox for categorical
    age = st.sidebar.slider('Age', 18, 70, 35)
    typeofcontact = st.sidebar.selectbox('Type of Contact', ['Self Enquiry', 'Company Invited'])
    citytier = st.sidebar.slider('City Tier', 1, 3, 2)
    durationofpitch = st.sidebar.slider('Duration of Pitch (minutes)', 1, 60, 10)
    occupation = st.sidebar.selectbox('Occupation', ['Salaried', 'Small Business', 'Large Business', 'Freelancer'])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Fe Male']) # 'Fe Male' is present in dataset
    numberofpersonvisiting = st.sidebar.slider('NumberOf Persons Visiting', 1, 10, 2)
    numberoffollowups = st.sidebar.slider('NumberOf Follow-ups', 0, 10, 3)
    productpitched = st.sidebar.selectbox('Product Pitched', ['Basic', 'Deluxe', 'Standard', 'Super Deluxe', 'King'])
    preferredpropertystar = st.sidebar.slider('Preferred Property Star Rating', 1, 5, 3)
    maritalstatus = st.sidebar.selectbox('Marital Status', ['Married', 'Single', 'Divorced', 'Unmarried'])
    numberoftrips = st.sidebar.slider('NumberOf Trips Annually', 1, 20, 2)
    passport = st.sidebar.selectbox('Has Passport?', [0, 1])
    pitchsatisfactionscore = st.sidebar.slider('Pitch Satisfaction Score', 1, 5, 3)
    owncar = st.sidebar.selectbox('Owns Car?', [0, 1])
    numberofchildrenvisiting = st.sidebar.slider('NumberOf Children Visiting', 0, 5, 1)
    monthlyincome = st.sidebar.slider('Monthly Income', 10000, 100000, 30000)
    designation = st.sidebar.selectbox('Designation', ['Executive', 'Manager', 'Senior Manager', 'AVP', 'VP', 'Director', 'Junior Executive', 'CEO', 'CFO', 'Chairman'])

    # Dummy 'CustomerID' as it was just an index in the original dataset, set to 0 for new predictions
    customer_id = 200000 # Use a dummy ID, or remove from features if not used

    data = {
            'CustomerID': customer_id,
            'Age': age,
            'TypeofContact': typeofcontact,
            'CityTier': citytier,
            'DurationOfPitch': durationofpitch,
            'Occupation': occupation,
            'Gender': gender,
            'NumberOfPersonVisiting': numberofpersonvisiting,
            'NumberOfFollowups': numberoffollowups,
            'ProductPitched': productpitched,
            'PreferredPropertyStar': preferredpropertystar,
            'MaritalStatus': maritalstatus,
            'NumberOfTrips': numberoftrips,
            'Passport': passport,
            'PitchSatisfactionScore': pitchsatisfactionscore,
            'OwnCar': owncar,
            'NumberOfChildrenVisiting': numberofchildrenvisiting,
            'MonthlyIncome': monthlyincome,
            'Designation': designation
           }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# --- Prediction ---
if st.button('Predict'):
    # Preprocess the input features
    # The feature names from the preprocessor object need to be retrieved correctly.
    # This part assumes `preprocessor` can directly transform the input_df.
    try:
        # Ensure that the column order and names match those used during preprocessor.fit()
        # The order of columns in input_df should match numerical_cols + categorical_cols conceptual order
        # even though ColumnTransformer handles mapping by name.
        processed_input = preprocessor.transform(input_df)

        # Get feature names after one-hot encoding for categorical columns
        # The get_feature_names_out method requires knowing the transformer's name (e.g., 'cat')
        ohe_feature_names = preprocessor.named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_cols)

        # Construct the full list of processed feature names in the correct order
        full_processed_feature_names = numerical_cols + list(ohe_feature_names)

        processed_input_df = pd.DataFrame(processed_input, columns=full_processed_feature_names, index=input_df.index)

        prediction = model.predict(processed_input_df)
        prediction_proba = model.predict_proba(processed_input_df)

        st.subheader('Prediction')
        if prediction[0] == 1:
            st.success("The customer is likely to purchase the Wellness Tourism Package!")
        else:
            st.info("The customer is unlikely to purchase the Wellness Tourism Package.")

        st.subheader('Prediction Probability (0: No Purchase, 1: Purchase)')
        st.write(f"No Purchase: {prediction_proba[0][0]:.2f}")
        st.write(f"Purchase: {prediction_proba[0][1]:.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.warning("Please ensure the input features are valid and the model and preprocessor are loaded correctly.")

# Instructions for deployment
st.markdown("""
---
### Deployment Instructions:

1.  **Save `app.py` and `requirements.txt`**: Ensure this `app.py` and the `requirements.txt` (from the next cell) are saved in `tourism_project/deployment/`.
2.  **Docker**: The `Dockerfile` (already generated) will build an image based on these files.
3.  **Hugging Face Space**: Push these files to your Hugging Face Space for deployment.
""")

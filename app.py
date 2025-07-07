import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the columns used for training
rf_model = joblib.load('random_forest_model.pkl')
training_columns = joblib.load('columns_used.pkl')

# Load the dataset to match columns
df_train = pd.read_csv('df_train.csv')

# Columns to drop
columns_to_drop = [
    'Customer_ID', 
    'Month',                   
    'Annual_Income',           
    'Monthly_Balance',   
    'Changed_Credit_Limit',    
    'Credit_Utilization_Ratio', 
    'Total_EMI_per_month',     
    'Amount_invested_monthly', 
]

# Clean the training data
df_train_cleaned = df_train.drop(columns=columns_to_drop, errors='ignore')

# Perform one-hot encoding on the cleaned training data
df_train_encoded = pd.get_dummies(df_train_cleaned)

# Streamlit UI: Title and Instructions
st.markdown("<h1 style='text-align: center;'>💳 Credit Score Prediction 💳</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center;'>Enter your details below to predict your credit score.</h5>", unsafe_allow_html=True)

# Collect input data from the user
new_data = {}

# Input numeric data for all numeric columns except the dropped ones
for col in df_train_cleaned.select_dtypes(include=['float64', 'int64']).columns:
    if col not in columns_to_drop and col != 'Credit_Score':
        min_value = int(df_train_cleaned[col].min())
        max_value = int(df_train_cleaned[col].max())
        label = col.replace("_", " ").capitalize()  # Replacing underscores with spaces and capitalizing
        new_data[col] = st.number_input(f'{label}', min_value=min_value, max_value=max_value, value=int(df_train_cleaned[col].mean()), key=f"{col}_input")

# Input categorical data for all categorical columns
for col in df_train_cleaned.select_dtypes(include=['object']).columns:
    if col not in columns_to_drop and col != 'Credit_Score':
        label = col.replace("_", " ").capitalize()  # Replacing underscores with spaces and capitalizing
        new_data[col] = st.selectbox(f'{label}', options=df_train_cleaned[col].unique(), key=f"{col}_input")

# Build the new data for prediction
new_data_df = pd.DataFrame([new_data])

# Function to preprocess new data and ensure it matches the training data's feature columns
def preprocess_new_data(new_data_df, training_columns):
    # Perform one-hot encoding on the new data
    new_data_encoded = pd.get_dummies(new_data_df)
    # Ensure the new data has the same columns as the training data
    new_data_encoded = new_data_encoded.reindex(columns=training_columns, fill_value=0)
    return new_data_encoded

# Function to predict credit score
def predict_credit_score(model, new_data, training_columns):
    # Preprocess the new data to match the training data format
    new_data_processed = preprocess_new_data(new_data, training_columns)
    # Perform prediction
    prediction = model.predict(new_data_processed)
    return prediction[0]

# Display the prediction result when the button is clicked
if st.button("Predict Credit Score"):
    credit_score = predict_credit_score(rf_model, new_data_df, training_columns)
    if credit_score == 'Poor':
        st.markdown(f"<h3 style='text-align: center;'>🚨 Your credit score is **Poor**. Consider improving your financial habits.</h3>", unsafe_allow_html=True)
    elif credit_score == 'Standard':
        st.markdown(f"<h3 style='text-align: center;'>⚖️ Your credit score is **Standard**. You're doing well, but there's room for improvement!</h3>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h3 style='text-align: center;'>🌟 Your credit score is **Good**. Keep up the good financial practices!</h3>", unsafe_allow_html=True)
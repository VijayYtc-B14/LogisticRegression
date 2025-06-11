# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Customer Travel Classifier", layout="wide")
st.title("Customer Travel Logistic Regression Classifier")

# --- Navigation ---
menu = st.radio("Navigate to:", [
    "View Data", 
    "Add New Data", 
    "Data Exploration", 
    "Model Training & Results"
])

# --- Load Data ---
DATA_URL = "https://raw.githubusercontent.com/VijayYtc-B14/LogisticRegression/refs/heads/main/Customertravel.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

# Load base dataset and initialize session state to track added data
if "full_data" not in st.session_state:
    st.session_state.full_data = load_data().dropna()

df = st.session_state.full_data

# --- View Raw Data ---
if menu == "View Data":
    st.subheader("Raw Data")
    st.dataframe(df)

    st.subheader("Data Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("Statistical Summary")
    st.write(df.describe(include='all'))

# --- Add New Data ---
elif menu == "Add New Data":
    st.subheader("Add a New Customer Entry")

    with st.form("new_data_form", clear_on_submit=True):
        age = st.number_input("Age", min_value=0, max_value=100, value=30)
        flyer = st.selectbox("Frequent Flyer", ["Yes", "No"])
        income = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])
        services = st.slider("Number of Services Opted", 0, 10, 3)
        synced = st.selectbox("Account Synced to Social Media", ["Yes", "No"])
        hotel = st.selectbox("Booked Hotel or Not", ["Yes", "No"])
        target = st.selectbox("Target (0 = Not Travelled, 1 = Travelled)", [0, 1])

        submitted = st.form_submit_button("Add Data")

        if submitted:
            new_row = {
                "Age": age,
                "FrequentFlyer": flyer,
                "AnnualIncomeClass": income,
                "ServicesOpted": services,
                "AccountSyncedToSocialMedia": synced,
                "BookedHotelOrNot": hotel,
                "Target": target
            }
            # Append new data to session
            st.session_state.full_data = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            st.success("New entry added successfully!")

# --- Data Exploration ---
elif menu == "Data Exploration":
    st.subheader("Target Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', ax=ax2)
    st.pyplot(fig2)

# --- Model Training & Results ---
elif menu == "Model Training & Results":
    st.subheader("Training Logistic Regression Model")

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Show results
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

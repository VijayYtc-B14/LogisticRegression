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
st.markdown("<h1 style='text-align: center;'>✈️ Customer Travel Logistic Regression Classifier</h1>", unsafe_allow_html=True)
st.markdown("---")

# --- Navigation ---
menu = st.sidebar.radio("📌 Navigation", [
    "📊 View Data", 
    "➕ Add New Data & Predict", 
    "📈 Data Exploration", 
    "🧠 Model Training & Results"
])

# --- Load Data ---
DATA_URL = "https://raw.githubusercontent.com/VijayYtc-B14/LogisticRegression/refs/heads/main/Customertravel.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_URL)

if "full_data" not in st.session_state:
    st.session_state.full_data = load_data().dropna()

df = st.session_state.full_data

# --- View Raw Data ---
if menu == "📊 View Data":
    st.subheader("🧾 Raw Dataset Preview")
    st.dataframe(df)

    st.subheader("🧠 Dataset Info")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

    st.subheader("📊 Statistical Summary")
    st.write(df.describe(include='all'))

# --- Add New Data & Predict ---
elif menu == "➕ Add New Data & Predict":
    st.subheader("➕ Add a New Customer and Predict Travel")

    with st.form("new_data_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=0, max_value=100, value=30)
            flyer = st.selectbox("Frequent Flyer", ["Yes", "No"])
            income = st.selectbox("Annual Income Class", ["Low Income", "Middle Income", "High Income"])
            services = st.slider("Number of Services Opted", 0, 10, 3)

        with col2:
            synced = st.selectbox("Account Synced to Social Media", ["Yes", "No"])
            hotel = st.selectbox("Booked Hotel or Not", ["Yes", "No"])

        submitted = st.form_submit_button("Add & Predict")

        if submitted:
            new_data = {
                "Age": age,
                "FrequentFlyer": flyer,
                "AnnualIncomeClass": income,
                "ServicesOpted": services,
                "AccountSyncedToSocialMedia": synced,
                "BookedHotelOrNot": hotel
            }

            # Add temporary row with dummy target for encoding consistency
            temp_df = pd.concat([df, pd.DataFrame([{**new_data, "Target": 0}])], ignore_index=True)

            # One-hot encoding
            df_encoded = pd.get_dummies(temp_df, drop_first=True)
            X = df_encoded.drop("Target", axis=1)

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train logistic model
            model = LogisticRegression()
            y = temp_df["Target"][:-1]
            model.fit(X_scaled[:-1], y)

            # Predict for last row (new data)
            y_pred = model.predict([X_scaled[-1]])
            travel_status = "🚶 Did Not Travel (0)" if y_pred[0] == 0 else "🧳 Travelled (1)"

            # Append new row with prediction
            new_data["Target"] = y_pred[0]
            st.session_state.full_data = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

            st.success(f"Prediction: {travel_status}")
            st.markdown(f"🎯 **Predicted Target**: `{y_pred[0]}`")

# --- Data Exploration ---
elif menu == "📈 Data Exploration":
    st.subheader("📊 Target Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("📌 Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', ax=ax2, cmap="YlGnBu")
    st.pyplot(fig2)

# --- Model Training & Results ---
elif menu == "🧠 Model Training & Results":
    st.subheader("⚙️ Logistic Regression Training")

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.metric("✅ Accuracy", f"{accuracy_score(y_test, y_pred) * 100:.2f}%")

    st.subheader("📋 Classification Report")
    st.text(classification_report(y_test, y_pred))

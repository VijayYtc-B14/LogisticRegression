import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set up the Streamlit app
st.title("Customer Travel Logistic Regression Classifier")

# Navigation Bar Below Title
menu = st.radio("Go to", ["Raw Data", "Data Exploration", "Model Training & Results"])

# Load dataset
data_url = "https://raw.githubusercontent.com/VijayYtc-B14/LogisticRegression/refs/heads/main/Customertravel.csv"

@st.cache_data
def load_data(url):
    df = pd.read_csv(url)
    df = df.dropna()
    return df

df = load_data(data_url)

if menu == "Raw Data":
    st.subheader("Raw Data")
    st.write(df.head())

    st.subheader("Data Info")
    buffer = []
    df.info(buf=buffer)
    s = '\n'.join(map(str, buffer))
    st.text(s)

    st.subheader("Statistical Summary")
    st.write(df.describe(include='all'))

elif menu == "Data Exploration":
    st.subheader("Target Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', ax=ax2)
    st.pyplot(fig2)

elif menu == "Model Training & Results":
    st.subheader("Training Logistic Regression Model")

    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))


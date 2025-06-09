# app.py
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

# Upload CSV file
uploaded_file = st.file_uploader("Customertravel.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Raw Data")
    st.write(df.head())

    st.subheader("Data Info")
    st.write(df.info())

    st.subheader("Statistical Summary")
    st.write(df.describe(include='all'))

    df = df.dropna()

    # Plot Target Distribution
    st.subheader("Target Class Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="Target", data=df, ax=ax1)
    st.pyplot(fig1)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, fmt='.2f', ax=ax2)
    st.pyplot(fig2)

    # Preprocessing
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop("Target", axis=1)
    y = df_encoded["Target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

    # Train model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Show Results
    st.subheader("Model Accuracy")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

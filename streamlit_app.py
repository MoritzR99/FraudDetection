# Import libraries

import pickle
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Streamlit App Configuration
st.set_page_config(
    page_title='Credit Card Fraud Detection',
    layout='wide',
    initial_sidebar_state='auto'
)

# Title
st.title("Credit Card Fraud Detection")

# Top Navigation Menu
page_selection = st.radio("", [
    "Introduction",
    "Data Overview",
    "Exploratory Data Analysis",
    "Feature Engineering",
    "Model Training and Evaluation",
    "Fraud Detection Simulator",
    "Download Report",
    "Feedback"
], horizontal=True)

# Display content and show balloons when a page is selected
if page_selection == "Introduction":
    st.balloons()  # Display balloons
    st.subheader("Welcome to the Credit Card Fraud Detection App")
    st.write("The app leverages a trained ANN model to predict the likelihood of fraud in credit card transactions. 
    It simplifies the analysis process for financial institutions and users concerned with fraudulent activity, providing real-time predictions based on transaction data.")

# Page selection for "Data Overview"
elif page_selection == "Data Overview":
    st.balloons()  # Display balloons
    st.subheader("Data Overview")
    st.write("Here is a summary of the dataset:")

    # Load the tables
    try:
        head_table = pd.read_pickle("head_table.pkl")  # Adjust path if necessary
        describe_table = pd.read_pickle("describe_table.pkl")

        # Display the tables
        st.write("### First Few Rows of the Dataset:")
        st.dataframe(head_table)  # Display head table

        st.write("### Dataset Description:")
        st.dataframe(describe_table)  # Display describe table

    except FileNotFoundError as e:
        st.error(f"Error loading tables: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

elif page_selection == "Exploratory Data Analysis":
    st.balloons()  # Display balloons
    st.subheader("Exploratory Data Analysis")
    st.write("Data visualizations go here...")

elif page_selection == "Feature Engineering":
    st.balloons()  # Display balloons
    st.subheader("Feature Engineering")
    st.write("Feature transformations and importance analysis...")

elif page_selection == "Model Training and Evaluation":
    st.balloons()  # Display balloons
    st.subheader("Model Training and Evaluation")
    st.write("Model performance metrics...")

elif page_selection == "Fraud Detection Simulator":
    st.balloons()  # Display balloons
    st.subheader("Fraud Detection Simulator")
    st.write("Interactive fraud detection form...")

elif page_selection == "Download Report":
    st.balloons()  # Display balloons
    st.subheader("Download Report")
    st.write("Option to download a report of the analysis...")

elif page_selection == "Feedback":
    st.balloons()  # Display balloons
    st.subheader("Feedback")
    st.write("Provide your feedback here.")


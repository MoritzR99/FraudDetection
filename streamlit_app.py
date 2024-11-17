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
    "Model Training",
    "Evaluation",
    "Fraud Detection Simulator"
], horizontal=True)

# Display content and show balloons when a page is selected
if page_selection == "Introduction":
    st.balloons()  # Display balloons
    st.subheader("Welcome to the Credit Card Fraud Detection App")
    st.write("The app leverages a trained ANN model to predict the likelihood of fraud in credit card transactions. It simplifies the analysis process for financial institutions and users concerned with fraudulent activity, providing real-time predictions based on transaction data.")

# Page selection for "Data Overview"
elif page_selection == "Data Overview":
    st.balloons()  # Display balloons
    st.subheader("Data Overview")
    st.write("Since most columns of the used dataset were PCA transformed, an extensive data overview is not meaningful here. Nevertheless, two tables will be shown below to give an general idea about the nature of the underlying dataset:")

    # Load the tables
    try:
        head_table = pd.read_pickle("head_table.pkl")  # Adjust path if necessary
        describe_table = pd.read_pickle("describe_table.pkl")

        # Display the tables
        st.write("### First Few Rows of the Dataset:")
        st.write("Here it can be seen, that the dataset consits out of 31 columns while class will be used as target, all others can be seen as possible features. Furthermore, all columns got PCA transformed, excluding time, amount and class.")
        st.dataframe(head_table)  # Display head table

        st.write("### Dataset Description:")
        st.write("Here, two of the non transformed columns get describes. This shows, that our target is binary, whereas 1 describes fraudulent transactions, while 0 describes non-fraudulent tansactions. Furthermore, the dataset is strongly biased towards non fraudulant transactions as it can be seen in the precentiles.")
        st.dataframe(describe_table)  # Display describe table

    except FileNotFoundError as e:
        st.error(f"Error loading tables: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Page selection for "Exploratory Data Analysis"
elif page_selection == "Exploratory Data Analysis":
    st.balloons()  # Display balloons
    st.subheader("Exploratory Data Analysis")
    st.write("Data visualizations go here...")

    def load_graph(file_name):
        """Helper function to load and display graphs from .pkl files."""
        try:
            with open(file_name, "rb") as file:
                fig = pickle.load(file)
            st.pyplot(fig)
        except FileNotFoundError as e:
            st.error(f"Graph not found: {file_name}")
        except Exception as e:
            st.error(f"An error occurred while loading {file_name}: {e}")

    # Display amount and count graphs side by side
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Non-Fraud Amount")
        load_graph("non_fraud_amount.pkl")

        st.write("### Non-Fraud Count")
        load_graph("non_fraud_count.pkl")

    with col2:
        st.write("### Fraud Amount")
        load_graph("fraud_amount.pkl")

        st.write("### Fraud Count")
        load_graph("fraud_count.pkl")

    # Display remaining graphs
    st.write("### Fraud Ratio")
    load_graph("fraud_ratio.pkl")

    st.write("### Average Amount by Class")
    load_graph("average_amount_by_class.pkl")

elif page_selection == "Feature Engineering":
    st.balloons()  # Display balloons
    st.subheader("Feature Engineering")
    st.write("No additional features were added, as most of them are already PCA transformed. The only transformation used, was the Standart Scaler to normalize features for the ANN model.")

elif page_selection == "Model Training":
    st.balloons()  # Display balloons
    st.subheader("Model Training and Evaluation")
    st.write("Model performance metrics...")

elif page_selection == "Evaluation":
    st.balloons()  # Display balloons
    st.subheader("Model Training and Evaluation")
    st.write("Model performance metrics...")

elif page_selection == "Fraud Detection Simulator":
    st.balloons()  # Display balloons
    st.subheader("Fraud Detection Simulator")
    st.write("Interactive fraud detection form...")


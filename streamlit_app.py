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

# Functions
# Function to load graphs from .pkl files
def load_graph(file_name):
    try:
        with open(file_name, "rb") as file:
            fig = pickle.load(file)
        return fig
    except FileNotFoundError:
        st.error(f"Graph not found: {file_name}")
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

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
    "Fraud Detection Simulator"
], horizontal=True)

# Display content and show balloons when a page is selected
if page_selection == "Introduction":
    st.balloons()  # Display balloons
    st.subheader("Welcome to the Credit Card Fraud Detection App")
    st.write(
        "The app leverages a trained ANN model to predict the likelihood of fraud in credit card transactions. "
        "It simplifies the analysis process for financial institutions and users concerned with fraudulent activity, "
        "providing real-time predictions based on transaction data."
    )

    # Display a credit card GIF
    gif_url = "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExMDhyZjY3d3A1OWVtOGk2cHl4YmF0MThhY3MyYzZ6cmQ2cDl3MGdvZCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/PqSxWQglz4dBtJgzdE/giphy.gif"
    st.markdown(
        f"""
        <div style="text-align: center;">
            <img src="{gif_url}" alt="Credit Card GIF" width="300">
            <p><em>Credit Card Fraud Detection in Action</em></p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
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
        st.write("Here, two of the non transformed columns get describes. This shows, that our target is binary, whereas 1 describes fraudulent transactions, while 0 describes non-fraudulent tansactions. Furthermore, the dataset is strongly biased towards non fraudulant transactions as it can be seen in the percentiles.")
        st.dataframe(describe_table)  # Display describe table

    except FileNotFoundError as e:
        st.error(f"Error loading tables: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")

# Page selection for "Exploratory Data Analysis"
elif page_selection == "Exploratory Data Analysis":
    st.balloons()  # Display balloons
    st.subheader("Exploratory Data Analysis")
    st.write("Select a visualization from the dropdown menu below:")

    # Dropdown menu for graph selection
    graph_selection = st.selectbox(
        "Choose a visualization:",
        [
            "Amount Over Time (Non-Fraud and Fraud)",
            "Count Over Time (Non-Fraud and Fraud)",
            "Fraudulent Ratio Over Time",
            "Average Amount by Class",
        ]
    )

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

    # Display selected graph based on dropdown choice
    if graph_selection == "Amount Over Time (Non-Fraud and Fraud)":
        st.write("### Non-Fraud Amount Over Time")
        load_graph("non_fraud_amount.pkl")
        st.write("### Fraud Amount Over Time")
        load_graph("fraud_amount.pkl")

    elif graph_selection == "Count Over Time (Non-Fraud and Fraud)":
        st.write("### Non-Fraud Count Over Time")
        load_graph("non_fraud_count.pkl")
        st.write("### Fraud Count Over Time")
        load_graph("fraud_count.pkl")

    elif graph_selection == "Fraudulent Ratio Over Time":
        st.write("### Fraudulent Ratio Over Time")
        load_graph("fraud_ratio.pkl")

    elif graph_selection == "Average Amount by Class":
        st.write("### Average Transaction Amount by Class")
        load_graph("average_amount_by_class.pkl")
        
elif page_selection == "Feature Engineering":
    st.balloons()  # Display balloons
    st.subheader("Feature Engineering")
    st.write("No additional features were added, as most of them are already PCA transformed. The only transformation used, was the Standart Scaler to normalize features for the ANN model.")

# Model Training and Evaluation Section
elif page_selection == "Model Training and Evaluation":
    st.balloons()
    st.subheader("Model Training and Evaluation")
    st.write("Model performance metrics...")

    # Dropdown for graph selection
    graph_selection = st.selectbox(
        "Choose a visualization or metrics to display:",
        [
            "Loss Curve",
            "Cutoff Analysis",
            "Classification Reports & Confusion Matrices",
        ]
    )

    if graph_selection == "Loss Curve":
        st.write("### Loss Curve")
        fig = load_graph("loss_curve.pkl")
        if fig:
            st.pyplot(fig)

    elif graph_selection == "Cutoff Analysis":
        st.write("### Cutoff Analysis")

        # Load cutoff analysis graph
        fig = load_graph("cutoff.pkl")
        if fig:
            st.pyplot(fig)

        # Add a slider to adjust the cutoff threshold
        cutoff_threshold = st.slider(
            "Select the cutoff threshold:", min_value=0.0, max_value=1.0, value=0.5, step=0.01
        )

        # Calculate and display precision and recall dynamically
        try:
            # Assuming you have access to the predicted probabilities and true labels
            y_pred_probs = ...  # Replace with your model's predicted probabilities
            y_true = ...  # Replace with your true labels

            y_pred = (y_pred_probs > cutoff_threshold).astype(int)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)

            st.write(f"**Precision Score:** {precision:.2f}")
            st.write(f"**Recall Score:** {recall:.2f}")
        except Exception as e:
            st.warning("Could not calculate precision and recall. Ensure y_pred_probs and y_true are available.")

    elif graph_selection == "Classification Reports & Confusion Matrices":
        st.write("### Classification Reports & Confusion Matrices")

        # Load and display Classification Report for Training
        st.write("#### Training Classification Report")
        train_report_fig = load_graph("classification_report_train.pkl")
        if train_report_fig:
            st.pyplot(train_report_fig)

        # Load and display Confusion Matrix for Training
        st.write("#### Training Confusion Matrix")
        train_cm_fig = load_graph("confusion_matrix_train.pkl")
        if train_cm_fig:
            st.pyplot(train_cm_fig)

        # Load and display Classification Report for Testing
        st.write("#### Testing Classification Report")
        test_report_fig = load_graph("classification_report_test.pkl")
        if test_report_fig:
            st.pyplot(test_report_fig)

        # Load and display Confusion Matrix for Testing
        st.write("#### Testing Confusion Matrix")
        test_cm_fig = load_graph("confusion_matrix_test.pkl")
        if test_cm_fig:
            st.pyplot(test_cm_fig)

elif page_selection == "Fraud Detection Simulator":
    st.balloons()  # Display balloons
    st.subheader("Fraud Detection Simulator")
    st.write("Interactive fraud detection form...")


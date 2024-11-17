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
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# Load the saved scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the cutoff_data.pkl at the start of the app
try:
    with open("cutoff_data.pkl", "rb") as file:
        cutoff_data = pickle.load(file)  # Ensure cutoff_data is available globally
except FileNotFoundError:
    st.error("cutoff_data.pkl file not found. Please upload or place the file in the correct directory.")
    cutoff_data = None  # Fallback to prevent errors later
except Exception as e:
    st.error(f"An error occurred while loading cutoff_data.pkl: {e}")
    cutoff_data = None

# Functions
def preprocess_input(df):
                """Preprocess input CSV to match the model's expected format."""
                # Ensure only numerical columns are kept
                numerical_columns = [col for col in df.columns if col.startswith('V') or col in ['Time', 'Amount']]
                return df[numerical_columns]
            
def predict_fraud(data):
    """Predict fraud using the loaded model."""
    predictions = fraud_detection_model.predict(data)
    return predictions
# Function to load graphs from .pkl files
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

def load_numpy(file_name):
    try:
        with open(file_name, "rb") as file:
            data = pickle.load(file)
            if isinstance(data, (list, tuple, dict)) or hasattr(data, "shape"):
                fig, ax = plt.subplots()
                sns.heatmap(data, annot=True, fmt="d", cmap="Blues", ax=ax)
                st.pyplot(fig)
                return data
            else:
                st.error(f"Unexpected data type: {type(data)}")
    except FileNotFoundError:
        st.error(f"Graph not found: {file_name}")
    except Exception as e:
        st.error(f"An error occurred while loading {file_name}: {e}")
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
    st.balloons()
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
    st.subheader("Feature Engineering")
    st.write("No additional features were added, as most of them are already PCA transformed. The only transformation used, was the Standart Scaler to normalize features for the ANN model.")

elif page_selection == "Model Training and Evaluation":
    st.subheader("Model Training and Evaluation")
    st.write("Explore model performance metrics using the dropdown menu below:")

    # Dropdown menu to select the graph/report to display
    evaluation_selection = st.selectbox(
        "Choose a visualization or report:",
        [
            "Loss Curve",
            "Cutoff Analysis",
            "Confusion Matrices",
            "Classification Reports",
        ]
    )

    if evaluation_selection == "Loss Curve":
        st.write("### Loss Curve")
        load_graph("loss_curve.pkl")

    elif evaluation_selection == "Cutoff Analysis":
        st.write("### Cutoff Analysis")

        # Adjustable cutoff slider
        thresholds = cutoff_data['thresholds']
        precisions = cutoff_data['precisions']
        recalls = cutoff_data['recalls']

        # Slider for cutoff adjustment
        cutoff_value = st.slider(
            "Adjust Cutoff Threshold",
            min_value=float(thresholds.min()),
            max_value=float(thresholds.max()),
            value=0.5,
            step=0.01,
        )

        # Find the closest precision and recall for the selected cutoff value
        closest_idx = (abs(thresholds - cutoff_value)).argmin()
        precision = precisions[closest_idx]
        recall = recalls[closest_idx]

        # Compute the threshold to be considered fraudulent
        threshold_to_be_fraudulent = 1 - cutoff_value

        st.write(f"**Cutoff Threshold**: {cutoff_value:.2f}")
        st.write(f"**Threshold to be considered fraudulent**: {threshold_to_be_fraudulent:.2f}")
        st.write(f"**Precision**: {precision:.2f}")
        st.text(classification_report)

        # Display cutoff graph
        load_graph("cutoff.pkl")
        
    elif evaluation_selection == "Confusion Matrices":
        st.write("### Confusion Matrices")
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
        
        # Left column: Train Confusion Matrix
        with col1:
            st.write("#### Training Set Confusion Matrix")
            load_numpy("confusion_matrix_train.pkl")
        
        # Right column: Test Confusion Matrix
        with col2:
            st.write("#### Test Set Confusion Matrix")
            load_numpy("confusion_matrix_test.pkl")
    
    elif evaluation_selection == "Classification Reports":
        st.write("#### Classification Reports")
        
        # Load the DataFrames
        classification_train = pd.read_pickle("classification_report_train.pk")
        classification_test = pd.read_pickle("classification_report_test.pk")
    
        # Create two columns for side-by-side display
        col1, col2 = st.columns(2)
    
        with col1:
            st.write("##### Train Classification Report")
            st.dataframe(classification_train)
    
        with col2:
            st.write("##### Test Classification Report")
            st.dataframe(classification_test)


elif page_selection == "Fraud Detection Simulator":
    st.subheader("Fraud Detection Simulator")
    # Dropdown to select input mode
    st.sidebar.title("Fraud Detection Simulator")
    input_mode = st.sidebar.selectbox("Select Input Mode", ["Manual Input", "Upload CSV"])
    
    if input_mode == "Manual Input":
        st.subheader("Manual Input")
        st.write("Interactive fraud detection form to predict transaction fraud. Play around with the features to explore the model!")
        
        # Load the model
        try:
            fraud_detection_model = load_model("model.h5")
        except FileNotFoundError:
            st.error("Model file not found. Please ensure 'model.h5' is in the same directory.")
            st.stop()
        
        # Sidebar for feature input
        st.sidebar.header("Input Transaction Details")
        st.sidebar.write("Use the sliders to input transaction details.")
        
        # Input sliders for transaction details
        time = st.sidebar.slider(
            "Transaction Time (seconds since first transaction)",
            min_value=0, max_value=100000, value=0, step=100
        )
        amount = st.sidebar.slider(
            "Transaction Amount",
            min_value=0.0, max_value=10000.0, value=0.0, step=0.01
        )
        
        # Input sliders for PCA-transformed V features
        st.sidebar.write("PCA-Transformed Features (V1 to V28)")
        v_features = [
            st.sidebar.slider(f"V{i}", min_value=-50.0, max_value=50.0, value=0.0, step=0.1) for i in range(1, 29)
        ]
        
        # Prepare input data
        input_data = np.array([[time, amount] + v_features], dtype=np.float32)

        # Standardize the input data using the loaded scaler
        standardized_data = scaler.transform(input_data)
        st.write("Standardized Input Data:")
        st.write(standardized_data)
        
        # Predict fraud
        if st.button("Detect Fraud"):
            try:
                # Make predictions
                prediction = fraud_detection_model.predict(standardized_data)
                fraud_probability = prediction[0][0]  # Assuming binary classification with a single output neuron
        
                # Display prediction results
                if fraud_probability > 0.5:  # Threshold for fraud
                    st.error(f"🚨 The transaction is predicted to be FRAUDULENT with a probability of {fraud_probability:.2f}.")
                else:
                    st.success(f"✅ The transaction is predicted to be LEGITIMATE with a probability of {1 - fraud_probability:.2f}.")
        
                # Display input summary as a table
                st.write("### Input Summary")
                summary = pd.DataFrame({
                    "Feature": ["Time", "Amount"] + [f"V{i}" for i in range(1, 29)],
                    "Value": [time, amount] + v_features
                })
                st.table(summary)
        
                # Provide a download button for the summary
                csv = summary.to_csv(index=False)
                st.download_button("Download Input Summary as CSV", csv, file_name="input_summary.csv")
            except Exception as e:
                st.error(f"An error occurred during prediction: {str(e)}")
        
        # Footer with instructions
        st.write("---")
        st.write("Adjust the features in the sidebar and click 'Detect Fraud' to see the results.")
        st.write("Feel free to share the public link with others!")

    if input_mode == "Upload CSV":
            st.subheader("Upload CSV")
            st.write("Upload your CSV and get your predictions!")

            # Load the model
            fraud_detection_model = load_model("model.h5")
                        
            # Fraud Detection Simulator Page
            st.subheader("Fraud Detection Simulator")
            st.write("Upload a CSV file with transaction data for fraud detection.")
        
            uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
            
            if uploaded_file is not None:
                try:
                    # Read the uploaded CSV file
                    input_data = pd.read_csv(uploaded_file)
                    
                    # Display the uploaded data
                    st.write("### Uploaded Data Preview")
                    st.dataframe(input_data.head())
                    
                    # Preprocess the input data
                    processed_data = preprocess_input(input_data)
                    
                    # Ensure the input matches the model's expected dimensions
                    if processed_data.shape[1] != 30:  # Assuming 'Time', 'Amount', and 'V1' to 'V28'
                        st.error("Uploaded CSV does not have the correct structure. Ensure it matches the required format.")
                    else:
                        # Predict fraud
                        st.write("### Predicting Fraud...")
                        predictions = predict_fraud(processed_data)
                        
                        # Add predictions to the dataframe
                        input_data['Fraud Prediction'] = (predictions > 0.5).astype(int)  # Binary classification
                        input_data['Fraud Probability'] = predictions
                        
                        # Display predictions
                        st.write("### Prediction Results")
                        st.dataframe(input_data[['Time', 'Amount', 'Fraud Prediction', 'Fraud Probability']].head())
                        
                        # Downloadable output
                        csv_output = input_data.to_csv(index=False)
                        st.download_button(
                            label="Download Prediction Results",
                            data=csv_output,
                            file_name="fraud_detection_results.csv",
                            mime="text/csv"
                        )
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        

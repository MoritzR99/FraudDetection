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

# Title and Sidebar Menu
st.title('Credit Card Fraud Detection')
st.sidebar.header("Menu")
page_selection = st.sidebar.radio("", [
    "Introduction",
    "Data Overview",
    "Exploratory Data Analysis",
    "Feature Importance",
    "Model Evaluation",
    "Simulator",
    "Download Report",
    "Feedback"
])

import streamlit as st
import pandas as pd
import joblib
import os
import warnings
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

# --- Constants ---
MODEL_FILE = 'nids_model.joblib'
TEST_DATA_FILE = 'KDDTest+.txt'

# Define the 43 column names for the KDD dataset
KDD_COL_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
    'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
    'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
    'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

# Define the feature columns (all columns except label and difficulty)
FEATURE_COLS = KDD_COL_NAMES[:-2]
CATEGORICAL_FEATURES = ['protocol_type', 'service', 'flag']
NUMERICAL_FEATURES = [col for col in FEATURE_COLS if col not in CATEGORICAL_FEATURES]


# --- Load Model and Data (Cached) ---

@st.cache_resource
def load_model(model_path):
    """
    Loads the trained model pipeline.
    Returns None if the file doesn't exist.
    """
    if not os.path.exists(model_path):
        return None
    try:
        pipeline = joblib.load(model_path)
        return pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_test_data(data_path):
    """
    Loads the KDDTest+.txt file and pre-calculates categorical options.
    Returns None if the file doesn't exist.
    """
    if not os.path.exists(data_path):
        return None
    try:
        df = pd.read_csv(data_path, header=None, names=KDD_COL_NAMES)
        
        # Get all unique options for categorical dropdowns
        categorical_options = {}
        for col in CATEGORICAL_FEATURES:
            categorical_options[col] = list(df[col].unique())
            
        return df, categorical_options
    except Exception as e:
        st.error(f"Error loading test data: {e}")
        return None, None

# --- Main Application ---
st.set_page_config(page_title="NIDS", layout="wide")
st.title("🛡️ Network Intrusion Detection System (NIDS)")

# 1. Load the trained model
model_pipeline = load_model(MODEL_FILE)

# 2. Load the test data and categorical options
test_data, categorical_options = load_test_data(TEST_DATA_FILE)

# 3. Check if model and data are loaded
if model_pipeline is None:
    st.error(f"**Model file not found: `{MODEL_FILE}`**")
    st.info("Please run `python train_nids_model.py` in your terminal first to train and save the model.")
    st.stop()

if test_data is None:
    st.error(f"**Test data file not found: `{TEST_DATA_FILE}`**")
    st.info("Please make sure `KDDTest+.txt` is in the same folder.")
    st.stop()

# --- Create UI Tabs ---
tab1, tab2 = st.tabs(["Simulate from Test File", "Manual Real-time Input"])

# ==============================================================================
# --- TAB 1: Simulate from Test File ---
# ==============================================================================
with tab1:
    st.header("Simulate a Connection from `KDDTest+.txt`")
    st.write("Use the slider to select a connection from the test file. The app will use the trained model to predict if it's an intrusion.")

    # Create a slider for the user to select a row index
    row_index = st.slider(
        "Select Connection (Row Index):",
        0,
        len(test_data) - 1,
        0,
        key="simulation_slider"
    )

    # Get the selected row
    selected_connection = test_data.iloc[[row_index]]

    # Display the features of the selected connection
    st.subheader("Selected Connection's Features:")
    features_to_display = selected_connection.drop(['label', 'difficulty'], axis=1)
    st.dataframe(features_to_display, use_container_width=True)

    # --- Prediction Section ---
    if st.button("Classify Selected Connection", type="primary"):
        
        features_to_predict = features_to_display.copy()
        
        prediction = model_pipeline.predict(features_to_predict)[0]
        probabilities = model_pipeline.predict_proba(features_to_predict)[0]
        
        actual_label = selected_connection['label'].values[0]
        actual_is_anomaly = 0 if actual_label == 'normal' else 1
        
        st.divider()
        st.subheader("Prediction Results")
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error(f"**Prediction: ANOMALY**")
                st.metric("Confidence", f"{probabilities[1]*100:.2f}%")
            else:
                st.success(f"**Prediction: NORMAL**")
                st.metric("Confidence", f"{probabilities[0]*100:.2f}%")
        with col2:
            st.write("**Ground Truth:**")
            if actual_is_anomaly == 1:
                st.error(f"Actual: **{actual_label.upper()}** (Anomaly)")
            else:
                st.success(f"Actual: **{actual_label.upper()}** (Normal)")

            if actual_is_anomaly == prediction:
                st.balloons()
                st.success("The model was **CORRECT**!")
            else:
                st.warning("The model was **INCORRECT**.")


# ==============================================================================
# --- TAB 2: Manual Real-time Input ---
# ==============================================================================
with tab2:
    st.header("Manual Real-time Input")
    st.write("Enter all 41 connection features below to get a real-time prediction.")

    st.subheader("1. (Optional) Pre-fill form with a test case")
    prefill_index = st.slider(
        "Select Connection to Pre-fill Form:",
        0,
        len(test_data) - 1,
        0,
        key="prefill_slider"
    )
    prefill_data = test_data.iloc[prefill_index]

    st.subheader("2. Enter Connection Features")
    with st.form("manual_input_form"):
        input_data = {}
        
        col1, col2, col3 = st.columns(3)
        
        # --- Categorical Features ---
        with col1:
            st.write("#### Categorical Features")
            for col in CATEGORICAL_FEATURES:
                options = categorical_options[col]
                try:
                    default_index = options.index(prefill_data[col])
                except ValueError:
                    default_index = 0
                
                input_data[col] = st.selectbox(
                    f"`{col}`",
                    options=options,
                    index=default_index
                )
        
        # --- Numerical Features ---
        with col2:
            st.write("#### Numerical Features (Part 1)")
            for col in NUMERICAL_FEATURES[:len(NUMERICAL_FEATURES)//2]:
                input_data[col] = st.number_input(
                    f"`{col}`",
                    value=float(prefill_data[col]),
                    min_value=0.0,
                    format="%.6f"
                )
        with col3:
            st.write("#### Numerical Features (Part 2)")
            for col in NUMERICAL_FEATURES[len(NUMERICAL_FEATURES)//2:]:
                input_data[col] = st.number_input(
                    f"`{col}`",
                    value=float(prefill_data[col]),
                    min_value=0.0,
                    format="%.6f"
                )
        
        submit_button = st.form_submit_button("Classify Manual Input", type="primary")

    if submit_button:
        # Convert dict to DataFrame, ensuring correct column order
        input_df = pd.DataFrame([input_data])
        input_df = input_df[FEATURE_COLS]
        
        prediction = model_pipeline.predict(input_df)[0]
        probabilities = model_pipeline.predict_proba(input_df)[0]
        
        st.divider()
        st.subheader("Prediction Result (Manual Input)")
        
        if prediction == 1:
            st.error(f"**Prediction: ANOMALY**")
            st.metric("Confidence", f"{probabilities[1]*100:.2f}%")
            st.warning("This connection is flagged as a potential intrusion.")
        else:
            st.success(f"**Prediction: NORMAL**")
            st.metric("Confidence", f"{probabilities[0]*100:.2f}%")
            st.info("This connection appears to be normal.")

        st.subheader("Data You Submitted:")
        st.dataframe(input_df, use_container_width=True)
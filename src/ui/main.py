import pandas as pd
import streamlit as st

from helpers import clean_resources, run_parsing_process, run_prediction_dialog, set_aws_credentials, transform
from src.utils.config import IS_LOCAL_RUN


if "made_ml_training" not in st.session_state:
    st.session_state.made_ml_training = False

if "query_log_result" not in st.session_state:
    st.session_state.query_log_result = {}

if "transform_result" not in st.session_state:
    st.session_state.transform_result = pd.DataFrame()

if "model" not in st.session_state:
    st.session_state.model = None

if "poly_features" not in st.session_state:
    st.session_state.poly_features = None

if "scaler" not in st.session_state:
    st.session_state.scaler = None


# Streamlit application title
st.title("Forecast AWS Athena Query Application")

st.write("### Choose the model for prediction")


# Model choice
model_choice = st.radio("## Choose Your Model:", ("Pretrained Model", "Train Your Own Model"))

# Based on model choice, perform appropriate actions
if model_choice == "Pretrained Model":
    st.write("### Choose an action")
    prediction_button = st.button("Make Prediction", use_container_width=True)
    clean_button = st.button("Clear All Local Cache", type="primary", use_container_width=True)

    if prediction_button:
        run_prediction_dialog(use_pretrained=True)

    elif clean_button:
        clean_resources()

elif model_choice == "Train Your Own Model":
    st.write("### Choose an action")

    process_button = st.button("Process Athena Executions", use_container_width=True)
    transform_button = st.button("Transform Query Logs", use_container_width=True)
    train_and_prediction_button = st.button("Train Model And Make Prediction", use_container_width=True)
    prediction_button = None
    prediction_button_disabled = None
    if not IS_LOCAL_RUN:
        if st.session_state.model and st.session_state.scaler and st.session_state.poly_features:
            prediction_button = st.button("Make Prediction (Trained on your data)", use_container_width=True)
        else:
            prediction_button_disabled = st.button(
                "Make Prediction (Trained on your data)", use_container_width=True, disabled=True
            )
    else:
        if st.session_state.made_ml_training:
            prediction_button = st.button("Make Prediction (Trained on your data)", use_container_width=True)
        else:
            prediction_button_disabled = st.button(
                "Make Prediction (Trained on your data)", use_container_width=True, disabled=True
            )

    clean_button = st.button("Clear All Local Cache", type="primary", use_container_width=True)

    # Initialize session state
    if "state" not in st.session_state:
        st.session_state.state = False
    if "aws_credentials" not in st.session_state:
        st.session_state.aws_credentials = None

    # Handle button clicks
    if process_button:
        if st.session_state.aws_credentials:
            run_parsing_process(st.session_state.aws_credentials)
        else:
            set_aws_credentials()

    elif transform_button:
        transform()

    elif train_and_prediction_button:
        run_prediction_dialog(use_pretrained=False, save_ml_attributes_in_memory=True)

    elif prediction_button:
        run_prediction_dialog(use_pretrained=True)

    elif prediction_button_disabled:
        pass

    elif clean_button:
        clean_resources()

    # Show a toast message when logs are parsed
    if st.session_state.state:
        st.toast("**Logs have been parsed!**", icon="🎯")
        st.session_state.state = False

import os
import sys
import time

import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.utils.config import DEFAULT_DIR_RAW_DATA, DEFAULT_OUTPUT_FILE, update_session
from helpers import (
    check_file_exists,
    clean_pycache,
    check_directory_empty,
    clean_docker_resources,
    clean_ml_model,
    clean_processed_data,
    clean_python_cache,
    clean_raw_data,
)

from src.app import prediction, parse_athena_executions, transform_query_logs

st.title("Forecast AWS Athena Query Application")

st.write("### Choose an action:")
process_button = st.button("Process Athena Executions", use_container_width=True)
transform_button = st.button("Transform Query Logs", use_container_width=True)
prediction_button = st.button("Make Prediction", use_container_width=True)
clean_button = st.button("Clear All Local Cache", type="primary", use_container_width=True)

if "state" not in st.session_state:
    st.session_state.state = False

if "aws_credentials" not in st.session_state:
    st.session_state.aws_credentials = None


def change_state():
    st.session_state.state = True


@check_file_exists(DEFAULT_OUTPUT_FILE)
@st.experimental_dialog("Please input the query string", width="large")
def run_prediction():
    with st.form("query-input", clear_on_submit=True):
        query_string = st.text_input(
            label="Write down Athena query to predict its scan size",
            placeholder="SELECT colums FROM tablename LIMIT 10",
        )
        submit = st.form_submit_button(label="Submit", use_container_width=True)
    if submit:
        st.info(f"Prediction for the query: {query_string[:40]}... has been started")
        with st.spinner("Operation in progress. Please wait..."):
            results = prediction.main(query_string)
            prediction_data = {
                "Metric": [
                    "Predicted Memory",
                    "Lower Bound",
                    "Upper Bound",
                    "Mean Squared Error (MSE)",
                    "Mean Absolute Error (MAE)",
                    "R-squared (R^2)",
                ],
                "Value": [
                    f"{results['predicted_memory']:.2f} bytes ({results['predicted_memory'] / 1_073_741_824:.2f} GB)",
                    f"{results['lower_bound']:.2f} bytes ({results['lower_bound'] / 1_073_741_824:.2f} GB)",
                    f"{results['upper_bound']:.2f} bytes ({results['upper_bound'] / 1_073_741_824:.2f} GB)",
                    results["mse"],
                    results["mae"],
                    results["r2"],
                ],
            }
            st.success("**Here are prediction results!**", icon="🔥")
            st.write("### Prediction Results")
            st.table(prediction_data)
            st.session_state.prediction_button = False


@st.experimental_dialog("You are trying to run the parsing process")
def run_parsing_process(session):
    st.info("That will take a while, do you want to proceed?", icon="ℹ️")
    col1, col2 = st.columns(spec=2, gap="small")
    with col1:
        if st.button("Yes", use_container_width=True, on_click=change_state):
            with st.spinner("Operation in progress. Please wait..."):
                parse_athena_executions.main(session=session)
                st.rerun()
    with col2:
        if st.button("No", type="primary", use_container_width=True):
            st.rerun()


@st.experimental_dialog("AWS Credentials")
def set_aws_credentials():
    st.info("Write down your AWS credentials")
    with st.form("get-aws-creds"):
        aws_access_key_id = st.text_input("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = st.text_input("AWS_SECRET_ACCESS_KEY", type="password")
        aws_session_token = st.text_input("AWS_SESSION_TOKEN", type="password")
        aws_default_region = st.text_input("AWS_DEFAULT_REGION", value="us-east-1")

        submit_button = st.form_submit_button("Submit", use_container_width=True, type="primary")

    if submit_button:
        if not aws_access_key_id or not aws_secret_access_key or not aws_session_token or not aws_default_region:
            st.error("All fields are required!")
        else:
            session = update_session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_default_region=aws_default_region,
            )

            session.client("athena")

            st.session_state.aws_credentials = session
            st.rerun()


@st.experimental_dialog("Clean Up Resources")
def clean_resources():
    if "clean_pycache" not in st.session_state:
        st.session_state.clean_pycache = False
    if "clean_processed_data" not in st.session_state:
        st.session_state.clean_processed_data = False
    if "clean_raw_data" not in st.session_state:
        st.session_state.clean_raw_data = False
    if "clean_ml_model" not in st.session_state:
        st.session_state.clean_ml_model = False
    if "clean_python_cache" not in st.session_state:
        st.session_state.clean_python_cache = False
    if "clean_docker_resources" not in st.session_state:
        st.session_state.clean_docker_resources = False

    col1, col2 = st.columns(spec=2, gap="small")
    with col1:
        if st.button("Check All", use_container_width=True):
            st.session_state.clean_pycache = True
            st.session_state.clean_processed_data = True
            st.session_state.clean_raw_data = True
            st.session_state.clean_ml_model = True
            st.session_state.clean_python_cache = True
            st.session_state.clean_docker_resources = True
    with col2:
        if st.button("Uncheck All", use_container_width=True):
            st.session_state.clean_pycache = False
            st.session_state.clean_processed_data = False
            st.session_state.clean_raw_data = False
            st.session_state.clean_ml_model = False
            st.session_state.clean_python_cache = False
            st.session_state.clean_docker_resources = False

    clean_pycache_checkbox = st.checkbox("Clean Python Cache (__pycache__)", value=st.session_state.clean_pycache)
    clean_processed_data_checkbox = st.checkbox("Clean Processed Data", value=st.session_state.clean_processed_data)
    clean_raw_data_checkbox = st.checkbox("Clean Raw Data", value=st.session_state.clean_raw_data)
    clean_ml_model_checkbox = st.checkbox("Clean ML Model Data", value=st.session_state.clean_ml_model)
    clean_python_cache_checkbox = st.checkbox(
        "Clean Python Cache (.pytest_cache, .ruff_cache)", value=st.session_state.clean_python_cache
    )
    clean_docker_resources_checkbox = st.checkbox(
        "Clean Docker Resources", value=st.session_state.clean_docker_resources
    )

    if st.button("Clean Selected Resources", type="primary"):
        with st.status("Cleaning resources...", expanded=True) as status:
            if clean_pycache_checkbox:
                clean_pycache()
                time.sleep(1)
                st.write("Cleaned Python Cache (__pycache__)")
            if clean_processed_data_checkbox:
                clean_processed_data()
                time.sleep(1)
                st.write("Cleaned Processed Data")
            if clean_raw_data_checkbox:
                clean_raw_data()
                time.sleep(1)
                st.write("Cleaned Raw Data")
            if clean_ml_model_checkbox:
                clean_ml_model()
                time.sleep(1)
                st.write("Cleaned ML Model Data")
            if clean_python_cache_checkbox:
                clean_python_cache()
                time.sleep(1)
                st.write("Cleaned Python Cache (.pytest_cache, .ruff_cache)")
            if clean_docker_resources_checkbox:
                clean_docker_resources()
                time.sleep(1)
                st.write("Cleaned Docker Resources")
            status.update(label="Cleaning complete!", state="complete", expanded=False)
        st.success("Selected resources cleaned successfully.")


if process_button:
    if st.session_state.aws_credentials:
        run_parsing_process(st.session_state.aws_credentials)
    else:
        set_aws_credentials()

elif transform_button:

    @check_directory_empty(DEFAULT_DIR_RAW_DATA)
    def transform():
        with st.spinner("Operation in progress. Please wait..."):
            st.success("All transformations have been applied!")
            st.write("## Result Dataframe")
            transform_query_logs.main()
            df = pd.read_csv(DEFAULT_OUTPUT_FILE)
            st.dataframe(df)

    transform()

elif prediction_button:
    run_prediction()

elif clean_button:
    clean_resources()

if st.session_state.state:
    st.toast("**Logs have been parsed!**", icon="🎯")
    st.session_state.state = False

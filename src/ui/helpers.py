"""
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

This script/module provides functions for cleaning up various resources and
running predictions on query sizes using ML models trained on AWS Athena query logs.

Copyright [2024] [Stanislav Kazanov]
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os
import shutil
import subprocess
import time
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.app import parse_athena_executions, prediction, transform_query_logs
from src.utils.config import (
    DEFAULT_DIR_RAW_DATA,
    DEFAULT_MODEL_FILE,
    DEFAULT_OUTPUT_FILE,
    IS_LOCAL_RUN,
    update_aws_session,
)


def clean_pycache():
    for root, dirs, files in os.walk(".", topdown=True):
        for directory in dirs:
            if directory == "__pycache__":
                shutil.rmtree(os.path.join(root, directory))


def clean_processed_data():
    processed_data_dir = "data/processed"
    for filename in os.listdir(processed_data_dir):
        if filename != ".gitkeep":
            os.remove(os.path.join(processed_data_dir, filename))


def clean_raw_data():
    raw_data_dir = "data/raw"
    for filename in os.listdir(raw_data_dir):
        if filename != ".gitkeep":
            os.remove(os.path.join(raw_data_dir, filename))


def clean_ml_model():
    ml_model_dir = "src/model"
    for filename in os.listdir(ml_model_dir):
        if filename != ".gitkeep":
            os.remove(os.path.join(ml_model_dir, filename))


def clean_python_cache():
    shutil.rmtree(".pytest_cache", ignore_errors=True)
    shutil.rmtree(".ruff_cache", ignore_errors=True)


def clean_docker_resources():
    subprocess.run(
        ["docker", "stop"] + subprocess.getoutput("docker ps -q").split(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["docker", "rm"] + subprocess.getoutput("docker ps -a -q").split(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["docker", "rmi"] + subprocess.getoutput("docker images -q").split(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(["docker", "image", "prune", "-a", "-f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "container", "prune", "-f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "volume", "prune", "-f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["docker", "system", "prune", "-f"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def clean_up_resources():
    clean_pycache()
    clean_processed_data()
    clean_raw_data()
    clean_ml_model()
    clean_python_cache()
    clean_docker_resources()


@st.experimental_dialog("Please input the query string", width="large")
def run_prediction(
    use_pretrained=False, transform_result=None, in_memory_ml_attributes={}, save_ml_attributes_in_memory=False
):
    with st.form("query-input"):
        query_string = st.text_input(
            label="Write down Athena query to predict its scan size",
            placeholder="SELECT colums FROM tablename LIMIT 10",
        )
        submit = st.form_submit_button(label="Submit", use_container_width=True)
    if submit:
        if not query_string:
            st.warning("Please provide a query string")
            st.stop()
        st.info(f"Prediction for the query: {query_string[:40]}... has been started")
        with st.spinner("Operation in progress. Please wait..."):
            results = prediction.main(
                query_string, use_pretrained, transform_result, in_memory_ml_attributes, save_ml_attributes_in_memory
            )
            if isinstance(results, dict):
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
            elif isinstance(results, np.float32) or isinstance(results, float):
                try:
                    predicted_memory = results
                    formatted_memory = f"{predicted_memory:.2f} bytes ({predicted_memory / 1_073_741_824:.2f} GB)"

                    prediction_data = {
                        "Metric": ["Predicted Memory"],
                        "Value": [formatted_memory],
                    }

                    st.success("**Here are prediction results!**", icon="🔥")
                    st.write("### Prediction Results")
                    st.table(prediction_data)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.error("Unexpected result type. Expected numpy.float32 or float.")


def transform():
    with st.spinner("Operation in progress. Please wait..."):
        if IS_LOCAL_RUN and any(file for file in os.listdir(DEFAULT_DIR_RAW_DATA) if file != ".gitkeep"):
            st.success("All transformations have been applied!")
            st.write("## Result Dataframe")
            result = transform_query_logs.main()
            df = pd.read_csv(DEFAULT_OUTPUT_FILE)
            st.dataframe(df)
        elif not IS_LOCAL_RUN and st.session_state.query_log_result:
            st.success("All transformations have been applied!")
            st.write("## Result Dataframe")
            result = transform_query_logs.main(query_log_result=st.session_state.query_log_result)
            st.session_state.transform_result = result.copy()
            st.dataframe(result)
        else:
            st.warning("No files found in the directory, please parse logs first :)")


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
    if "clean_docker_resources" not in st.session_state:
        st.session_state.clean_docker_resources = False

    col1, col2 = st.columns(spec=2, gap="small")
    with col1:
        if st.button("Check All", use_container_width=True):
            st.session_state.clean_pycache = True
            st.session_state.clean_processed_data = True
            st.session_state.clean_raw_data = True
            st.session_state.clean_docker_resources = True
    with col2:
        if st.button("Uncheck All", use_container_width=True):
            st.session_state.clean_pycache = False
            st.session_state.clean_processed_data = False
            st.session_state.clean_raw_data = False
            st.session_state.clean_ml_model = False
            st.session_state.clean_docker_resources = False

    clean_pycache_checkbox = st.checkbox("Clean Python Cache", value=st.session_state.clean_pycache)
    clean_processed_data_checkbox = st.checkbox("Clean Processed Data", value=st.session_state.clean_processed_data)
    clean_raw_data_checkbox = st.checkbox("Clean Raw Data", value=st.session_state.clean_raw_data)
    clean_docker_resources_checkbox = None
    if IS_LOCAL_RUN and "clean_docker_resources" in st.session_state:
        clean_docker_resources_checkbox = st.checkbox(
            "Clean Docker Resources", value=st.session_state.clean_docker_resources
        )
    clean_ml_model_checkbox = st.checkbox(
        ":red[Clean ML Model Data]",
        value=st.session_state.clean_ml_model,
        help="If you drop initial model, new model training will be too slow!",
    )
    if st.button("Clean Selected Resources", type="primary", use_container_width=True):
        if (
            not clean_pycache_checkbox
            and not clean_processed_data_checkbox
            and not clean_raw_data_checkbox
            and not clean_ml_model_checkbox
            and not clean_docker_resources_checkbox
        ):
            st.warning("Please mark necessary points first")
            st.stop()
        else:
            with st.status("Cleaning resources...", expanded=True) as status:
                if clean_pycache_checkbox:
                    clean_pycache()
                    clean_python_cache()
                    time.sleep(1)
                    st.write("Cleaned Python Cache: .pytest_cache, .ruff_cache, etc.")
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
                if clean_docker_resources_checkbox:
                    clean_docker_resources()
                    time.sleep(1)
                    st.write("Cleaned Docker Resources")
                status.update(label="Cleaning complete!", state="complete", expanded=False)
            st.success("Selected resources cleaned successfully.")


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
            st.stop()
        else:
            session = update_aws_session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                aws_default_region=aws_default_region,
            )
            try:
                session.client("athena").list_work_groups()
                st.session_state.aws_credentials = session
                st.rerun()
            except Exception:
                st.warning("Please add valid AWS credentials")
                st.stop()


@st.experimental_dialog("You are trying to run the parsing process")
def run_parsing_process(session):
    st.info("That will take a while, do you want to proceed?")
    col1, col2 = st.columns(spec=2, gap="small")
    with col1:
        if st.button("Yes", use_container_width=True, on_click=change_state):
            with st.spinner("Operation in progress. Please wait..."):
                parse_athena_executions.main(session=session)
                st.rerun()
    with col2:
        if st.button("No", type="primary", use_container_width=True):
            st.rerun()


def change_state():
    st.session_state.state = True


def run_prediction_dialog(use_pretrained, save_ml_attributes_in_memory=False):
    if use_pretrained and os.path.exists(DEFAULT_MODEL_FILE):
        run_prediction(use_pretrained=use_pretrained)
    elif use_pretrained and not os.path.exists(DEFAULT_MODEL_FILE):
        st.warning("ML model not found, please train yours")
    elif not use_pretrained and IS_LOCAL_RUN and os.path.exists(DEFAULT_OUTPUT_FILE):
        run_prediction()
    elif not use_pretrained and not IS_LOCAL_RUN and not st.session_state.transform_result.empty:
        run_prediction(
            transform_result=st.session_state.transform_result,
            save_ml_attributes_in_memory=save_ml_attributes_in_memory,
        )
    elif use_pretrained and not IS_LOCAL_RUN and not st.session_state.transform_result.empty:
        run_prediction(
            use_pretrained=use_pretrained,
            transform_result=st.session_state.transform_result,
            in_memory_ml_attributes={
                "model": st.session_state.model,
                "poly_features": st.session_state.poly_features,
                "scaler": st.session_state.scaler,
            },
        )
    else:
        st.warning("You can't work with the model, please parse logs first :)")

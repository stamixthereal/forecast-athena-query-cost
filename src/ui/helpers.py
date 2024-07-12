# Copyright 2024 Stanislav Kazanov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import boto3
import docker
import numpy as np
import pandas as pd
import streamlit as st

from src.app import parse_athena_executions, prediction, transform_query_logs
from src.utils.config import (
    BYTES_IN_GB,
    DEFAULT_DIR_RAW_DATA,
    DEFAULT_MODEL_FILE,
    DEFAULT_OUTPUT_FILE,
    IS_LOCAL_RUN,
    update_aws_session,
)


def clean_pycache() -> None:
    """Remove all __pycache__ directories recursively."""
    for root, dirs, _ in os.walk(Path()):
        for directory in dirs:
            if directory == "__pycache__":
                shutil.rmtree(Path(root) / directory)


def clean_processed_data() -> None:
    """Remove all files in the processed data directory except .gitkeep."""
    processed_data_dir = Path("data/processed")
    for filename in processed_data_dir.iterdir():
        if filename.name != ".gitkeep":
            filename.unlink()


def clean_raw_data() -> None:
    """Remove all files in the raw data directory except .gitkeep."""
    raw_data_dir = Path("data/raw")
    for filename in raw_data_dir.iterdir():
        if filename.name != ".gitkeep":
            filename.unlink()


def clean_ml_model() -> None:
    """Remove all files in the ML model directory except .gitkeep."""
    ml_model_dir = Path("src/model")
    for filename in ml_model_dir.iterdir():
        if filename.name != ".gitkeep":
            filename.unlink()


def clean_python_cache() -> None:
    """Remove Python-related cache directories."""
    shutil.rmtree(".pytest_cache", ignore_errors=True)
    shutil.rmtree(".ruff_cache", ignore_errors=True)


def clean_docker_resources() -> None:
    """Clean up all Docker resources including containers, images, and volumes."""
    client = docker.from_env()

    for container in client.containers.list():
        container.stop()
        container.remove()

    for image in client.images.list():
        client.images.remove(image.id, force=True)

    client.containers.prune()
    client.images.prune()
    client.volumes.prune()
    client.networks.prune()


def clean_up_resources() -> None:
    """Clean up various system and application resources."""
    clean_pycache()
    clean_processed_data()
    clean_raw_data()
    clean_ml_model()
    clean_python_cache()
    clean_docker_resources()


@st.experimental_dialog("Please input the query string", width="large")
def run_prediction(
        use_pretrained: bool = False,
        transform_result: pd.DataFrame | None = None,
        in_memory_ml_attributes: dict[str, Any] | None = None,
        save_ml_attributes_in_memory: bool = False,
) -> None:
    """Run prediction on the provided Athena query string."""
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
                query_string,
                use_pretrained,
                transform_result,
                in_memory_ml_attributes,
                save_ml_attributes_in_memory,
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
                        f"{results['predicted_memory']:.2f} bytes ({results['predicted_memory'] / BYTES_IN_GB:.2f} GB)",
                        f"{results['lower_bound']:.2f} bytes ({results['lower_bound'] / BYTES_IN_GB:.2f} GB)",
                        f"{results['upper_bound']:.2f} bytes ({results['upper_bound'] / BYTES_IN_GB:.2f} GB)",
                        results["mse"],
                        results["mae"],
                        results["r2"],
                    ],
                }
                st.success("**Here are prediction results!**", icon="🔥")
                st.write("### Prediction Results")
                st.table(prediction_data)
            elif isinstance(results, float | np.float32):
                predicted_memory = results
                formatted_memory = f"{predicted_memory:.2f} bytes ({predicted_memory / BYTES_IN_GB:.2f} GB)"

                prediction_data = {
                    "Metric": ["Predicted Memory"],
                    "Value": [formatted_memory],
                }

                st.success("**Here are prediction results!**", icon="🔥")
                st.write("### Prediction Results")
                st.table(prediction_data)
            else:
                st.error("Unexpected result type. Expected numpy.float32 or float.")


def transform() -> None:
    """Transform raw query logs into a processed format for prediction."""
    with st.spinner("Operation in progress. Please wait..."):
        if IS_LOCAL_RUN and any(file for file in os.listdir(DEFAULT_DIR_RAW_DATA) if file != ".gitkeep"):
            st.write("## Result Dataframe")
            result = transform_query_logs.main()
            st.success("All transformations have been applied!")
            st.dataframe(result)
        elif not IS_LOCAL_RUN and st.session_state.query_log_result:
            st.write("## Result Dataframe")
            result = transform_query_logs.main(query_log_result=st.session_state.query_log_result)
            st.success("All transformations have been applied!")
            st.session_state.transform_result = result.copy()
            st.dataframe(result)
        else:
            st.warning("No files found in the directory, please parse logs first :)")


@st.experimental_dialog("Clean Up Resources")
def clean_resources() -> None:
    """Provide an interactive dialog for cleaning various resources."""
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
            "Clean Docker Resources",
            value=st.session_state.clean_docker_resources,
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
def set_aws_credentials() -> None:
    """Parse Athena query logs and display the parsed dataframe."""
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
def run_parsing_process(session: boto3.Session) -> None:
    """Run parsing process dialog window."""
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


def change_state() -> None:
    """Change state."""
    st.session_state.state = True


def run_prediction_dialog(use_pretrained: bool = False, save_ml_attributes_in_memory: bool = False) -> None:
    """Run dialog window according to the parameters."""
    if use_pretrained and Path.exists(DEFAULT_MODEL_FILE):
        run_prediction(use_pretrained=use_pretrained)
    elif use_pretrained and not Path.exists(DEFAULT_MODEL_FILE):
        st.warning("ML model not found, please train yours")
    elif not use_pretrained and IS_LOCAL_RUN and Path.exists(DEFAULT_OUTPUT_FILE):
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


def preview_raw_data() -> None:
    """Preview raw data."""
    raw_data = None
    if IS_LOCAL_RUN:
        raw_data_files = [file for file in Path(DEFAULT_DIR_RAW_DATA).iterdir() if file.name != ".gitkeep"]
        if raw_data_files:
            random_raw_data_file = random.SystemRandom().choice(raw_data_files)
            with random_raw_data_file.open() as f:
                raw_data = json.load(f)
    elif not IS_LOCAL_RUN and st.session_state.query_log_result:
        random_key = random.SystemRandom().choice(list(st.session_state.query_log_result.keys()))
        raw_data = st.session_state.query_log_result[random_key]

    if raw_data is not None:
        st.header("Sample (Random) File From Raw Data")
        st.json(raw_data)
    else:
        st.write("No raw data yet...")


def get_transformed_data() -> pd.DataFrame:
    """Get transformed data."""
    transformed_data_df = pd.DataFrame()
    if IS_LOCAL_RUN and Path(DEFAULT_OUTPUT_FILE).exists():
        transformed_data_df = pd.read_csv(DEFAULT_OUTPUT_FILE)
    elif not IS_LOCAL_RUN and not st.session_state.transform_result.empty:
        transformed_data_df = st.session_state.transform_result

    return transformed_data_df


def preview_transformed_data() -> None:
    """Preview transformed data."""
    transformed_data = get_transformed_data()

    if not transformed_data.empty:
        st.header("Transformed Dataframe")
        st.dataframe(transformed_data)
    else:
        st.write("No transformed data yet...")

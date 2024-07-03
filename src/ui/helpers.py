# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/

# Copyright [2024] [Stanislav Kazanov]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
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


def clean_directory(directory: Path, exclude: str = ".gitkeep") -> None:
    """Remove all files in a directory except the specified file."""
    for filename in directory.iterdir():
        if filename.name != exclude:
            filename.unlink()


def clean_pycache() -> None:
    """Remove all __pycache__ directories recursively."""
    for root, dirs, _ in os.walk(Path()):
        for directory in dirs:
            if directory == "__pycache__":
                shutil.rmtree(Path(root) / directory)


def clean_python_cache() -> None:
    """Remove Python-related cache directories."""
    for cache_dir in [".pytest_cache", ".ruff_cache"]:
        shutil.rmtree(cache_dir, ignore_errors=True)


def clean_docker_resources() -> None:
    """Clean up all Docker resources including containers, images, and volumes."""
    client = docker.from_env()

    for container in client.containers.list():
        container.stop()
        container.remove()

    for image in client.images.list():
        client.images.remove(image.id, force=True)

    for resource in [client.containers, client.images, client.volumes, client.networks]:
        resource.prune()


def format_memory(memory: float) -> str:
    """Format memory size in bytes to a human-readable string."""
    return f"{memory:.2f} bytes ({memory / BYTES_IN_GB:.2f} GB)"


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
            placeholder="SELECT columns FROM tablename LIMIT 10",
        )
        submit = st.form_submit_button(label="Submit", use_container_width=True)

    if submit and query_string:
        st.info(f"Prediction for the query: {query_string[:40]}... has been started")
        with st.spinner("Operation in progress. Please wait..."):
            results = prediction.main(
                query_string,
                use_pretrained,
                transform_result,
                in_memory_ml_attributes,
                save_ml_attributes_in_memory,
            )
            display_prediction_results(results)
    elif submit:
        st.warning("Please provide a query string")


def display_prediction_results(results: dict[str, Any] | float | np.float32) -> None:
    """Display prediction results in a formatted table."""
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
                format_memory(results["predicted_memory"]),
                format_memory(results["lower_bound"]),
                format_memory(results["upper_bound"]),
                results["mse"],
                results["mae"],
                results["r2"],
            ],
        }
    elif isinstance(results, (float | np.float32)):
        prediction_data = {
            "Metric": ["Predicted Memory"],
            "Value": [format_memory(results)],
        }
    else:
        st.error("Unexpected result type. Expected dict, float, or numpy.float32.")
        return

    st.success("**Here are prediction results!**", icon="🔥")
    st.write("### Prediction Results")
    st.table(prediction_data)


def transform() -> None:
    """Transform raw query logs into a processed format for prediction."""
    with st.spinner("Operation in progress. Please wait..."):
        if IS_LOCAL_RUN and any(file for file in os.listdir(DEFAULT_DIR_RAW_DATA) if file != ".gitkeep"):
            result = transform_query_logs.main()
        elif not IS_LOCAL_RUN and st.session_state.query_log_result:
            result = transform_query_logs.main(query_log_result=st.session_state.query_log_result)
            st.session_state.transform_result = result.copy()
        else:
            st.warning("No files found in the directory, please parse logs first :)")
            return

        st.success("All transformations have been applied!")
        st.write("## Result Dataframe")
        st.dataframe(result)


@st.experimental_dialog("Clean Up Resources")
def clean_resources() -> None:
    """Provide an interactive dialog for cleaning various resources."""
    resources = {
        "clean_pycache": "Clean Python Cache",
        "clean_processed_data": "Clean Processed Data",
        "clean_raw_data": "Clean Raw Data",
        "clean_ml_model": "Clean ML Model Data",
    }
    if IS_LOCAL_RUN:
        resources["clean_docker_resources"] = "Clean Docker Resources"

    col1, col2 = st.columns(spec=2, gap="small")
    with col1:
        if st.button("Check All", use_container_width=True):
            for key in resources:
                st.session_state[key] = True
    with col2:
        if st.button("Uncheck All", use_container_width=True):
            for key in resources:
                st.session_state[key] = False

    checkboxes = {}
    for key, label in resources.items():
        if key == "clean_ml_model":
            checkboxes[key] = st.checkbox(
                f":red[{label}]",
                value=st.session_state.get(key, False),
                help="If you drop initial model, new model training will be too slow!",
            )
        else:
            checkboxes[key] = st.checkbox(label, value=st.session_state.get(key, False))

    if st.button("Clean Selected Resources", type="primary", use_container_width=True):
        if not any(checkboxes.values()):
            st.warning("Please mark necessary points first")
        else:
            clean_selected_resources(checkboxes)


def clean_selected_resources(checkboxes: dict[str, bool]) -> None:
    """Clean the resources selected by the user."""
    with st.status("Cleaning resources...", expanded=True) as status:
        clean_functions = {
            "clean_pycache": lambda: (clean_pycache(), clean_python_cache()),
            "clean_processed_data": lambda: clean_directory(Path("data/processed")),
            "clean_raw_data": lambda: clean_directory(Path("data/raw")),
            "clean_ml_model": lambda: clean_directory(Path("src/model")),
            "clean_docker_resources": clean_docker_resources,
        }
        for key, checked in checkboxes.items():
            if checked:
                clean_functions[key]()
                time.sleep(1)
                st.write(f"Cleaned {key.replace('clean_', '').replace('_', ' ').title()}")
        status.update(label="Cleaning complete!", state="complete", expanded=False)
    st.success("Selected resources cleaned successfully.")


@st.experimental_dialog("AWS Credentials")
def set_aws_credentials() -> None:
    """Set AWS credentials for the session."""
    st.info("Write down your AWS credentials")
    with st.form("get-aws-creds"):
        aws_access_key_id = st.text_input("AWS_ACCESS_KEY_ID")
        aws_secret_access_key = st.text_input("AWS_SECRET_ACCESS_KEY", type="password")
        aws_session_token = st.text_input("AWS_SESSION_TOKEN", type="password")
        aws_default_region = st.text_input("AWS_DEFAULT_REGION", value="us-east-1")
        submit_button = st.form_submit_button("Submit", use_container_width=True, type="primary")

    if submit_button:
        if all([aws_access_key_id, aws_secret_access_key, aws_session_token, aws_default_region]):
            try:
                session = update_aws_session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token,
                    aws_default_region=aws_default_region,
                )
                session.client("athena").list_work_groups()
                st.session_state.aws_credentials = session
                st.rerun()
            except Exception:
                st.warning("Please add valid AWS credentials")
        else:
            st.error("All fields are required!")


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

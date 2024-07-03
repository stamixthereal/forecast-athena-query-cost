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

from collections.abc import Callable
from typing import Any

import pandas as pd
import streamlit as st

from src.ui.helpers import clean_resources, run_parsing_process, run_prediction_dialog, set_aws_credentials, transform
from src.utils.config import IS_LOCAL_RUN


def initialize_session_state() -> None:
    """Initialize session state variables."""
    default_values: dict[str, Any] = {
        "made_ml_training": False,
        "query_log_result": {},
        "transform_result": pd.DataFrame(),
        "model": None,
        "poly_features": None,
        "scaler": None,
        "state": False,
        "aws_credentials": None,
        "clean_pycache": False,
        "clean_processed_data": False,
        "clean_raw_data": False,
        "clean_ml_model": False,
        "clean_docker_resources": False,
    }

    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def create_button(label: str, callback: Callable, **kwargs) -> None:
    """Create a button with a callback function."""
    if st.button(label, **kwargs):
        callback()


def handle_pretrained_model() -> None:
    """Handle actions for the pretrained model."""
    st.write("### Choose an action")
    create_button("Make Prediction", lambda: run_prediction_dialog(use_pretrained=True), use_container_width=True)
    create_button("Clear All Local Cache", clean_resources, type="primary", use_container_width=True)


def handle_custom_model() -> None:
    """Handle actions for training a custom model."""
    st.write("### Choose an action")
    create_button("Process Athena Executions", process_athena_executions, use_container_width=True)
    create_button("Transform Query Logs", transform, use_container_width=True)
    create_button(
        "Train Model And Make Prediction",
        lambda: run_prediction_dialog(use_pretrained=False, save_ml_attributes_in_memory=True),
        use_container_width=True,
    )

    prediction_button_enabled = (
        not IS_LOCAL_RUN and all([st.session_state.model, st.session_state.scaler, st.session_state.poly_features])
    ) or (IS_LOCAL_RUN and st.session_state.made_ml_training)

    if prediction_button_enabled:
        create_button(
            "Make Prediction (Trained on your data)",
            lambda: run_prediction_dialog(use_pretrained=True),
            use_container_width=True,
        )
    else:
        st.button("Make Prediction (Trained on your data)", use_container_width=True, disabled=True)

    create_button("Clear All Local Cache", clean_resources, type="primary", use_container_width=True)


def process_athena_executions() -> None:
    """Process Athena executions with AWS credentials."""
    if st.session_state.aws_credentials:
        run_parsing_process(st.session_state.aws_credentials)
    else:
        set_aws_credentials()


def main() -> None:
    """Run the Streamlit application."""
    initialize_session_state()

    st.title("Forecast AWS Athena Query Application")
    st.write("### Choose the model for prediction")

    model_choice = st.radio("## Choose Your Model:", ("Pretrained Model", "Train Your Own Model"))

    if model_choice == "Pretrained Model":
        handle_pretrained_model()
    else:
        handle_custom_model()

    if st.session_state.state:
        st.toast("**Logs have been parsed!**", icon="🎯")
        st.session_state.state = False


if __name__ == "__main__":
    main()

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

import pandas as pd
import streamlit as st

from src.ui.helpers import (
    clean_resources,
    get_transformed_data,
    preview_raw_data,
    preview_transformed_data,
    run_parsing_process,
    run_prediction_dialog,
    set_aws_credentials,
    transform,
)
from src.utils.config import IS_LOCAL_RUN

default_session_state = {
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

for key, value in default_session_state.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("Forecast AWS Athena Query Application")

st.header("Choose the model for prediction", divider="red")

model_choice = st.radio(
    label="Choose the model",
    label_visibility="collapsed",
    options=("Pretrained Model", "Train Your Own Model"),
)

with st.sidebar:
    with st.popover("Preview Raw Data", use_container_width=True):
        preview_raw_data()

    with st.popover("Preview Transformed Data", use_container_width=True):
        preview_transformed_data()

    transformed_data_df = get_transformed_data()
    if not transformed_data_df.empty:
        st.download_button(
            label="Download Transformed Data",
            data=transformed_data_df.to_csv(),
            file_name="transformed_data.csv",
            use_container_width=True,
            type="primary",
        )

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
    prediction_button_enabled = (not IS_LOCAL_RUN
                                 and st.session_state.model
                                 and st.session_state.scaler
                                 and st.session_state.poly_features) or st.session_state.made_ml_training

    prediction_button = st.button(
        "Make Prediction (Trained on your data)",
        use_container_width=True,
        disabled=not prediction_button_enabled,
    )

    clean_button = st.button("Clear All Local Cache", type="primary", use_container_width=True)

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

    elif clean_button:
        clean_resources()

    if st.session_state.state:
        st.toast("**Logs have been parsed!**", icon="🎯")
        st.session_state.state = False

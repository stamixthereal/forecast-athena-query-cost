import streamlit as st

from helpers import clean_resources, run_parsing_process, run_prediction, set_aws_credentials, transform


# Streamlit application title
st.title("Forecast AWS Athena Query Application")

# User action buttons
st.write("### Choose an action:")
process_button = st.button("Process Athena Executions", use_container_width=True)
transform_button = st.button("Transform Query Logs", use_container_width=True)
prediction_button = st.button("Make Prediction", use_container_width=True)
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

elif prediction_button:
    run_prediction()

elif clean_button:
    clean_resources()

# Show a toast message when logs are parsed
if st.session_state.state:
    st.toast("**Logs have been parsed!**", icon="🎯")
    st.session_state.state = False

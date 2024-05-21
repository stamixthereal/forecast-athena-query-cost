import subprocess
import streamlit as st
import pandas as pd
import os

st.title("Forecast AWS Athena Query Application")

st.write("### Choose an action:")
process_button = st.button("Process Athena Executions")
transform_button = st.button("Transform Query Logs")
pipeline_button = st.button("Run The Whole Pipeline")
prediction_button = st.button("Make Prediction")

def run_script_and_display_output(script_path):
    with st.spinner("Operation in progress. Please wait..."):
        env = os.environ.copy()
        env['PYTHONPATH'] = f"{os.getcwd()}"
        result = subprocess.run(["python", script_path], capture_output=True, text=True, env=env)
        stdout = result.stdout
        stderr = result.stderr
        if stdout:
            st.success("Operation completed successfully")
            st.text(stdout)
        if stderr:
            st.info("Logs for the job:")
            st.text(stderr)

if process_button:
    run_script_and_display_output('src/app/parse_athena_executions.py')

elif transform_button:
    run_script_and_display_output('src/app/transform_query_logs.py')
    st.write("## Result Dataframe")
    df = pd.read_csv('./data/processed/processed_data.csv')
    st.dataframe(df)

elif pipeline_button:
    scripts = [
        'src/app/parse_athena_executions.py',
        'src/app/transform_query_logs.py',
        'src/app/prediction.py'
    ]
    for script in scripts:
        run_script_and_display_output(script)

elif prediction_button:
    run_script_and_display_output('src/app/prediction.py')

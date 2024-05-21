import streamlit as st
import pandas as pd
from helpers import clean_up_resources, run_script_and_display_output

st.title("Forecast AWS Athena Query Application")

st.write("### Choose an action:")
process_button = st.button("Process Athena Executions")
transform_button = st.button("Transform Query Logs")
pipeline_button = st.button("Run The Whole Pipeline")
prediction_button = st.button("Make Prediction")
clean_button = st.button("Clear All Local Cache", type='primary')

if process_button:
    run_script_and_display_output('src/app/parse_athena_executions.py')

elif transform_button:
    st.write("## Result Dataframe")
    run_script_and_display_output('src/app/transform_query_logs.py')
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

elif clean_button:
    clean_up_resources()
    st.success("Operation completed successfully")

import os
import shutil
import subprocess
import streamlit as st


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


def run_script_and_display_output(script_path):
    with st.spinner("Operation in progress. Please wait..."):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.getcwd()}"
        result = subprocess.run(["python3", script_path], capture_output=True, text=True, env=env)
        stdout, stderr = result.stdout, result.stderr
        if stdout:
            st.success("Operation completed successfully")
            with st.expander("Output results:"):
                st.text(stdout)
        if stderr:
            with st.expander("Logs for the job:"):
                st.text(stderr)

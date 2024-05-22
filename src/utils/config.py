import os
import logging

import boto3

from dotenv import load_dotenv

load_dotenv()


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Global defaults and constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR_RAW_DATA = os.path.join(SCRIPT_DIR, "../../data/raw")
DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "../../data/processed/processed_data.csv")
DEFAULT_MODEL_FILE = os.path.join(SCRIPT_DIR, "../../src/model/model.h5")
DEFAULT_REGION_NAME = "us-east-1"

# Load AWS credentials from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_SESSION_TOKEN = os.getenv("AWS_SESSION_TOKEN")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION") or DEFAULT_REGION_NAME

# Create a session using the loaded credentials
session = boto3.Session(
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    aws_session_token=AWS_SESSION_TOKEN,
    region_name=AWS_DEFAULT_REGION,
)

import os
import logging
import boto3

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Global defaults and constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR_RAW_DATA = os.path.join(SCRIPT_DIR, "../../data/raw")
DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "../../data/processed/processed_data.csv")
DEFAULT_MODEL_FILE = os.path.join(SCRIPT_DIR, "../../src/model/model.h5")
DEFAULT_REGION_NAME = "us-east-1"


def update_session(aws_access_key_id, aws_secret_access_key, aws_session_token, aws_default_region):
    logger.info("Updating AWS session with new credentials")

    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    os.environ["AWS_SESSION_TOKEN"] = aws_session_token
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region

    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )

    logger.info("Session updated: %s", session)

    return session

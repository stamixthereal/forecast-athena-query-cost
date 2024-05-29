import os
import logging
import platform
from typing import Optional
import boto3

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Global defaults and constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR_RAW_DATA = os.path.join(SCRIPT_DIR, "../../data/raw")
DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "../../data/processed/processed_data.csv")
DEFAULT_SCALER_FILE = os.path.join(SCRIPT_DIR, "../../src/model/scaler.pkl")
DEFAULT_POLY_FILE = os.path.join(SCRIPT_DIR, "../../src/model/poly_features.pkl")
DEFAULT_MODEL_FILE = os.path.join(SCRIPT_DIR, "../../src/model/model.json")
DEFAULT_REGION_NAME = "us-east-1"
IS_LOCAL_RUN = True if platform.processor() else False


def update_aws_session(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: Optional[str] = None,
    aws_default_region: str = DEFAULT_REGION_NAME,
) -> boto3.Session:
    """
    Update AWS session with provided credentials and region.
    """
    logger.info("Updating AWS session with new credentials")

    os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
    os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
    if aws_session_token:
        os.environ["AWS_SESSION_TOKEN"] = aws_session_token
    os.environ["AWS_DEFAULT_REGION"] = aws_default_region

    session = boto3.Session(
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ["AWS_SESSION_TOKEN"],
        region_name=os.environ["AWS_DEFAULT_REGION"],
    )

    logger.info("AWS session updated successfully")
    return session

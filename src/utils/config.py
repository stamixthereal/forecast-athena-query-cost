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

import logging
import os
import platform
from pathlib import Path

import boto3

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DIR_RAW_DATA = SCRIPT_DIR / "../../data/raw"
DEFAULT_OUTPUT_FILE = SCRIPT_DIR / "../../data/processed/processed_data.csv"
DEFAULT_SCALER_FILE = SCRIPT_DIR / "../../src/model/scaler.pkl"
DEFAULT_POLY_FILE = SCRIPT_DIR / "../../src/model/poly_features.pkl"
DEFAULT_MODEL_FILE = SCRIPT_DIR / "../../src/model/model.json"
DEFAULT_REGION_NAME = "us-east-1"
IS_LOCAL_RUN = bool(platform.processor())
BYTES_IN_GB = 1_073_741_824


def update_aws_session(
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_session_token: str | None = None,
    aws_default_region: str = DEFAULT_REGION_NAME,
) -> boto3.Session:
    """Update AWS session with provided credentials and region."""
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

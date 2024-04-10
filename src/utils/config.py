import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Global defaults and constants
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR_RAW_DATA = os.path.join(SCRIPT_DIR, "../../data/raw")
DEFAULT_OUTPUT_FILE = os.path.join(SCRIPT_DIR, "../../data/processed/processed_data.csv")
DEFAULT_MODEL_FILE = os.path.join(SCRIPT_DIR, "../../src/model/model.h5")
DEFAULT_REGION_NAME = "us-east-1"

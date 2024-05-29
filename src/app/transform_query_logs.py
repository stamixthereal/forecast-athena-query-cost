import csv
import glob
import json
import os
import re
from typing import List, Optional

import pandas as pd

from src.utils.config import DEFAULT_OUTPUT_FILE, DEFAULT_DIR_RAW_DATA, IS_LOCAL_RUN, logger


def clean_query(query: str) -> str:
    """Clean the query by removing excessive white spaces."""
    return re.sub(r"\s+", " ", query.strip())


def is_dml_and_succeeded(data: dict) -> bool:
    """Check if the query is of DML type and has succeeded."""
    return data.get("StatementType") == "DML" and data.get("Status", {}).get("State") == "SUCCEEDED"


def get_csv_row(data: dict) -> Optional[List]:
    """Convert the query log to a CSV row format."""
    if not is_dml_and_succeeded(data):
        return None

    query_id = data["QueryExecutionId"]
    user = "random_user"  # Placeholder
    source = data["WorkGroup"]
    environment = data["QueryExecutionContext"]["Database"]
    catalog = environment
    query = clean_query(data["Query"])
    statistics = data.get("Statistics", {})
    peak_memory_bytes = statistics.get("DataScannedInBytes", 0)
    cpu_time_ms = statistics.get("TotalExecutionTimeInMillis", 0)

    if peak_memory_bytes == 0:
        return None

    return [query_id, user, source, environment, catalog, query, peak_memory_bytes, cpu_time_ms]


def process_file(file_path: str) -> Optional[List]:
    """Process a single JSON file and return its CSV row representation."""
    with open(file_path, "r") as f:
        data = json.load(f)
        if not data:
            logger.warning(f"No data found in {file_path}. Skipping.")
            return None
        return get_csv_row(data)


def process_file_in_memory(query_log: dict) -> Optional[List]:
    """Process a single JSON file from in-memory data and return its CSV row representation."""
    if not query_log:
        logger.warning("No data found. Skipping.")
        return None
    return get_csv_row(query_log)


def initialize_csv(file_path: str, headers: List[str]):
    """Initialize the CSV file with headers if it doesn't exist."""
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(headers)


def write_row_to_csv(file_path: str, row: List):
    """Write a single row to the CSV file."""
    with open(file_path, "a", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(row)


def process_query_logs(input_dir: str, output_file: str, query_log_result):
    """Process query logs from input directory and write to output CSV file."""
    logger.info(f"Starting the processing of query logs from directory: {input_dir}.")

    column_names = [
        "query_id",
        "user_",
        "source",
        "environment",
        "catalog",
        "query",
        "peak_memory_bytes",
        "cpu_time_ms",
    ]
    if IS_LOCAL_RUN:
        initialize_csv(output_file, column_names)

    if IS_LOCAL_RUN:
        json_files = glob.glob(os.path.join(input_dir, "*.json"))
        total_files = len(json_files)
        logger.info(f"Found {total_files} JSON files for processing.")

        for count, json_file in enumerate(json_files, start=1):
            row = process_file(json_file)
            if row:
                write_row_to_csv(output_file, row)

            if count % (total_files // 20) == 0:  # Log every 5%
                logger.info(f"Processed {count}/{total_files} files ({(count/total_files)*100:.2f}%).")

        logger.info(f"Processing completed. Data written to {output_file}.")
        return
    else:
        rows = []
        for file_path, query_log in enumerate(query_log_result.items()):
            row = process_file_in_memory(query_log[1])
            if row:
                rows.append(row)
        result_df = pd.DataFrame(rows, columns=column_names)
        logger.info("Data has been processed in memory")
        return result_df


def main(query_log_result=None):
    return process_query_logs(DEFAULT_DIR_RAW_DATA, DEFAULT_OUTPUT_FILE, query_log_result=query_log_result)


if __name__ == "__main__":
    main()

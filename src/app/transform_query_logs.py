# Apache License
# Version 2.0, January 2004
# http://www.apache.org/licenses/

# Copyright [2024] [Stanislav Kazanov]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.config import DEFAULT_DIR_RAW_DATA, DEFAULT_OUTPUT_FILE, IS_LOCAL_RUN, logger


def clean_query(query: str) -> str:
    """Clean the query by removing excessive white spaces."""
    return re.sub(r"\s+", " ", query.strip())


def is_dml_and_succeeded(data: dict) -> bool:
    """Check if the query is of DML type and has succeeded."""
    return data.get("StatementType") == "DML" and data.get("Status", {}).get("State") == "SUCCEEDED"


def get_row(data: dict) -> list | None:
    """Convert the query log to a CSV row format."""
    if not is_dml_and_succeeded(data):
        return None

    query_id = data["QueryExecutionId"]
    query = clean_query(data["Query"])
    statistics = data.get("Statistics", {})
    peak_memory_bytes = statistics.get("DataScannedInBytes", 0)
    cpu_time_ms = statistics.get("TotalExecutionTimeInMillis", 0)

    if peak_memory_bytes == 0:
        return None

    return [query_id, query, peak_memory_bytes, cpu_time_ms]


def process_file(file_path: str) -> list | None:
    """Process a single JSON file and return its CSV row representation."""
    with Path.open(file_path) as f:
        data = json.load(f)
        return get_row(data)


def process_data_in_memory(query_log: dict[str, Any]) -> list | None:
    """Process a single JSON file from in-memory data and return its CSV row representation."""
    return get_row(query_log)


def process_query_logs(
    input_dir: str,
    output_file: str,
    query_log_result: dict[str:Any] | None,
) -> pd.DataFrame:
    """Process query logs from input directory and write to output CSV file."""
    logger.info("Starting the processing of query logs")

    column_names = [
        "query_id",
        "query",
        "peak_memory_bytes",
        "cpu_time_ms",
    ]
    if IS_LOCAL_RUN:
        json_files = list(Path(input_dir).glob("*.json"))
        total_files = len(json_files)
        logger.info(f"Found {total_files} JSON files for processing.")
        rows = []
        for count, json_file in enumerate(json_files, start=1):
            row = process_file(json_file)
            if row:
                rows.append(row)
            if count % (total_files // 20) == 0:  # Log every 5%
                logger.info(f"Processed {count}/{total_files} files ({(count/total_files)*100:.2f}%).")
        result_df = pd.DataFrame(rows, columns=column_names)
        result_df.to_csv(output_file)
        logger.info(f"Processing completed. Data written to {output_file}.")
        return result_df
    rows = [result for query_log in query_log_result.values() if query_log
        for result in [process_data_in_memory(query_log)] if result is not None]
    result_df = pd.DataFrame(rows, columns=column_names)
    logger.info("Data has been processed in memory.")
    return result_df


def main(query_log_result: dict[str:Any] | None = None) -> pd.DataFrame:
    """Process query logs."""
    return process_query_logs(DEFAULT_DIR_RAW_DATA, DEFAULT_OUTPUT_FILE, query_log_result=query_log_result)


if __name__ == "__main__":
    main()

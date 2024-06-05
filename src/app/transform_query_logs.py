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
from typing import Any, ClassVar

import pandas as pd

from src.utils.config import DEFAULT_DIR_RAW_DATA, DEFAULT_OUTPUT_FILE, IS_LOCAL_RUN, logger


class QueryLogProcessor:

    """Base class for processing query logs."""

    column_names: ClassVar = [
        "query_id",
        "query",
        "peak_memory_bytes",
        "cpu_time_ms",
    ]

    def __init__(self) -> None:
        pass

    @staticmethod
    def clean_query(query: str) -> str:
        """Clean the query by removing excessive white spaces."""
        return re.sub(r"\s+", " ", query.strip())

    @staticmethod
    def is_dml_and_succeeded(data: dict) -> bool:
        """Check if the query is of DML type and has succeeded."""
        return data.get("StatementType") == "DML" and data.get("Status", {}).get("State") == "SUCCEEDED"

    @staticmethod
    def get_row(data: dict) -> list[Any] | None:
        """Convert the query log to a CSV row format."""
        if not QueryLogProcessor.is_dml_and_succeeded(data):
            return None

        query_id = data["QueryExecutionId"]
        query = QueryLogProcessor.clean_query(data["Query"])
        statistics = data.get("Statistics", {})
        peak_memory_bytes = statistics.get("DataScannedInBytes", 0)
        cpu_time_ms = statistics.get("TotalExecutionTimeInMillis", 0)

        if peak_memory_bytes == 0:
            return None

        return [query_id, query, peak_memory_bytes, cpu_time_ms]

    def process_query_logs(self) -> pd.DataFrame:
        """Process query logs and write to output CSV file."""
        error_msg = "Subclasses should implement this method"
        raise NotImplementedError(error_msg)


class LocalQueryLogProcessor(QueryLogProcessor):

    """Processes query logs from local JSON files."""

    def __init__(self, input_dir: str, output_file: str) -> None:
        super().__init__()
        self.input_dir = input_dir
        self.output_file = output_file

    def process_file(self, file_path: str) -> list[Any] | None:
        """Process a single JSON file and return its CSV row representation."""
        with Path.open(file_path) as f:
            data = json.load(f)
            return self.get_row(data)

    def process_query_logs(self) -> pd.DataFrame:
        """Process query logs from input directory and write to output CSV file."""
        logger.info("Starting the processing of query logs")

        json_files = list(Path(self.input_dir).glob("*.json"))
        total_files = len(json_files)
        logger.info(f"Found {total_files} JSON files for processing.")

        rows = []

        for count, json_file in enumerate(json_files, start=1):
            row = self.process_file(json_file)
            if row:
                rows.append(row)
            if count % (total_files // 20) == 0:  # Log every 5%
                logger.info(f"Processed {count}/{total_files} files ({(count/total_files)*100:.2f}%).")

        result_df = pd.DataFrame(rows, columns=self.column_names)
        result_df.to_csv(self.output_file)
        logger.info(f"Processing completed. Data written to {self.output_file}.")

        return result_df


class InMemoryQueryLogProcessor(QueryLogProcessor):

    """Processes query logs from in-memory data."""

    def __init__(self, query_log_result: dict[str, Any]) -> None:
        super().__init__()
        self.query_log_result = query_log_result

    def process_data_in_memory(self, query_log: dict[str, Any]) -> list[Any] | None:
        """Process a single JSON file from in-memory data and return its CSV row representation."""
        return self.get_row(query_log)

    def process_query_logs(self) -> pd.DataFrame:
        """Process query logs from in-memory data."""
        logger.info("Starting the processing of query logs in memory")

        rows = [
            self.process_data_in_memory(query_log)
            for query_log in self.query_log_result.values()
            if query_log and self.process_data_in_memory(query_log) is not None
        ]

        result_df = pd.DataFrame(rows, columns=self.column_names)
        logger.info("Data has been processed in memory.")

        return result_df


def main(query_log_result: dict[str, Any] | None = None) -> pd.DataFrame:
    """Process query logs."""
    if IS_LOCAL_RUN:
        processor = LocalQueryLogProcessor(input_dir=DEFAULT_DIR_RAW_DATA, output_file=DEFAULT_OUTPUT_FILE)
    else:
        processor = InMemoryQueryLogProcessor(query_log_result=query_log_result)

    return processor.process_query_logs()


if __name__ == "__main__":
    main()

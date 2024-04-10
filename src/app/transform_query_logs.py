import argparse
import csv
import glob
import json
import os
import re

from src.utils.config import DEFAULT_OUTPUT_FILE, DEFAULT_DIR_RAW_DATA, logger


class QueryLog:
    """A class to represent a single Query Log.

    :param data: The JSON data for a query log.
    :type data: dict
    """

    query_cleaner = re.compile(r"\s+")

    def __init__(self, data):
        self.data = data

    @staticmethod
    def get_user():
        """
        Extract user information from the data.

        Currently, this method returns a placeholder value.
        TODO: Implement extraction of real user data from the log or other sources.

        :return: The username or identifier.
        :rtype: str
        """
        # Placeholder, replace with actual user data extraction logic
        return "random_user"

    def is_dml_and_succeeded(self):
        """Check if the query is of DML type and has succeeded.

        :return: True if query is DML and succeeded, False otherwise.
        :rtype: bool
        """
        return self.data.get("StatementType") == "DML" and self.data.get("Status", {}).get("State") == "SUCCEEDED"

    def to_csv_row(self):
        """Convert the query log to a CSV row format.

        :return: List of values representing a CSV row.
        :rtype: list
        """
        query_id = self.data["QueryExecutionId"]
        user = self.get_user()
        source = self.data["WorkGroup"]
        environment = self.data["QueryExecutionContext"]["Database"]
        catalog = environment
        query = self.clean_query(self.data["Query"])
        statistics = self.data.get("Statistics", {})
        peak_memory_bytes = statistics.get("DataScannedInBytes", 0)
        cpu_time_ms = statistics.get("TotalExecutionTimeInMillis", 0)
        return [
            query_id,
            user,
            source,
            environment,
            catalog,
            query,
            peak_memory_bytes,
            cpu_time_ms,
        ]

    @staticmethod
    def clean_query(query):
        """Clean the query by removing excessive white spaces.

        :param query: The raw query string.
        :type query: str
        :return: The cleaned query string.
        :rtype: str
        """
        return QueryLog.query_cleaner.sub(" ", query.strip())


class QueryLogReader:
    """A class to handle reading and processing query logs from JSON files."""

    @staticmethod
    def process(file_to_process):
        """Convert a JSON file to a QueryLog object.

        :param file_to_process: Path to the JSON file.
        :type file_to_process: str
        :return: A processed QueryLog object or None.
        :rtype: QueryLog or None
        """
        with open(file_to_process, "r") as f_in:
            data = json.load(f_in)
            if data is None or data == {}:
                logger.warning(f"No data found in {file_to_process}. Skipping.")
                return None

            query_log = QueryLog(data)
            if query_log.is_dml_and_succeeded():
                return query_log


class CSVWriter:
    """A class to handle writing data rows to a CSV file."""

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

    def __init__(self, csv_file):
        self.csv_file = csv_file

    def write_row(self, data_row):
        """Write a single row to the CSV file.

        :param data_row: Data row to be written to the CSV.
        :type data_row: list
        :return: None
        """
        try:
            if not os.path.exists(os.path.dirname(self.csv_file)):
                os.makedirs(os.path.dirname(self.csv_file))

            with open(self.csv_file, "a", newline="") as f_out:
                csv_writer = csv.writer(f_out)

                if os.stat(self.csv_file).st_size == 0:
                    csv_writer.writerow(self.column_names)
                csv_writer.writerow(data_row)
        except IOError:
            logger.error(f"Error while trying to write to the file: {self.csv_file}. Check permissions or disk space.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while writing to the file {self.csv_file}: {str(e)}")


def process_file(file_to_process):
    """
    Process a single JSON file and return its CSV row representation.
    This function is created to be used with concurrent.futures.
    """
    reader = QueryLogReader()
    query_log = reader.process(file_to_process)
    if query_log:
        new_line = query_log.to_csv_row()
        # Skip the row if peak_memory_bytes is 0 (index 6 in the row)
        if new_line[6] == 0:
            return None
        return new_line
    return None


def main() -> None:
    """
    Entry point for the script which processes SQL query logs in JSON format and writes the results to a CSV file.

    This function performs the following steps:

    1. **Setting up the Argument Parser**:
        - Accepts input and output directory/file paths.

    2. **Validation**:
        - Ensures the provided arguments correspond to valid directory or file paths.

    3. **Retrieving JSON Files**:
        - Loads a list of JSON files from the specified input directory.

    4. **Processing Each JSON File**:
        - Extracts the relevant query log data.
        - Converts qualifying data (meeting conditions of DML type and success) into a CSV row format.
        - Writes rows with a non-zero peak memory bytes value to the output CSV file.

    5. **Logging**:
        - Provides feedback on the processing status at regular intervals (every 5% by default).
        - Logs a message indicating completion and the path to the output CSV file once all files are processed.

    **Command-Line Arguments**:

    - `--input_dir (str)`:
        - Description: Path to the directory containing JSON files with SQL query logs.
        - Default: '../../data/raw'. Must be a valid directory path.

    - `--output_file (str)`:
        - Description: Path to the output CSV file where processed data will be written.
        - Default: '../../data/processed/processed_data.csv'. Must end with '.csv'.

    **Example Usage**:

        $ python script_name.py --input_dir "./json_logs" --output_file "./results/processed_data.csv"

    :raises argparse.ArgumentTypeError: Raised if `input_dir` isn't a valid directory or `output_file`
        isn't a valid CSV file path.
    :return: None
    """
    parser = argparse.ArgumentParser(
        description="This script processes SQL query logs in JSON format from a specified "
        "input directory and writes the processed data to a specified CSV "
        "file."
    )

    def directory_type(arg_value, pat=re.compile(r"^[./\\a-zA-Z0-9_-]+$")):
        if not pat.match(arg_value) or not os.path.isdir(arg_value):
            raise argparse.ArgumentTypeError(f"'{arg_value}' is not a valid directory path")
        return arg_value

    def file_type(arg_value, pat=re.compile(r"^[./\\a-zA-Z0-9_-]+\.csv$")):
        if not pat.match(arg_value):
            raise argparse.ArgumentTypeError(f"'{arg_value}' is not a valid CSV file path")
        return arg_value

    parser.add_argument(
        "--input_dir",
        default=DEFAULT_DIR_RAW_DATA,
        type=directory_type,
        help="Path to the input directory containing JSON files with SQL query logs. "
        f"Default is '{DEFAULT_DIR_RAW_DATA}'.",
    )

    parser.add_argument(
        "--output_file",
        default=DEFAULT_OUTPUT_FILE,
        type=file_type,
        help="Path to the output CSV file where the processed data will be written. "
        f"Default is '{DEFAULT_OUTPUT_FILE}'.",
    )

    args = parser.parse_args()

    input_directory = os.path.abspath(args.input_dir)
    output_csv_file = os.path.abspath(args.output_file)

    logger.info(f"Starting the processing of query logs from directory: {input_directory}.")

    writer = CSVWriter(output_csv_file)
    json_files = glob.glob(os.path.join(input_directory, "*.json"))

    total_files = len(json_files)
    logger.info(f"Found {total_files} JSON files for processing.")

    processed_count = 0

    # Before the loop, initialize the last logged percentage to None.
    last_logged_percentage = None
    percentage_threshold = 5  # Log every 5% increment in progress

    for json_file in json_files:
        row = process_file(json_file)
        processed_count += 1

        # Calculate the current progress as a percentage
        current_percentage = (processed_count / total_files) * 100

        # Check if the progress has crossed the threshold since the last logged percentage
        if last_logged_percentage is None or (current_percentage - last_logged_percentage) >= percentage_threshold:
            logger.info(f"Processed {processed_count}/{total_files} files ({current_percentage:.2f}%).")
            last_logged_percentage = current_percentage

        if row:
            writer.write_row(row)

    logger.info(f"Processing completed. Data written to {output_csv_file}.")


if __name__ == "__main__":
    main()

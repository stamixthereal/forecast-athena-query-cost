import argparse
import datetime
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, Dict, List

import boto3

from src.utils.config import DEFAULT_REGION_NAME, DEFAULT_DIR_RAW_DATA, logger

# Load AWS credentials from environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
aws_session_token = os.getenv("AWS_SESSION_TOKEN")
aws_default_region = os.getenv("AWS_DEFAULT_REGION")

# Create a session using the loaded credentials
session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,
    region_name=aws_default_region,
)


class QueryLogDownloader:
    """
    QueryLogDownloader is a class for downloading query logs from AWS Athena in parallel for multiple workgroups.

    :param output_dir: The directory where query logs will be stored.
    :param region_name: The AWS region name (default is "us-east-1").
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_DIR_RAW_DATA,
        region_name: str = DEFAULT_REGION_NAME,
    ):
        """
        Initialize the QueryLogDownloader.

        :param output_dir: The directory where query logs will be stored.
        :param region_name: The AWS region name (default is "us-east-1").
        """
        self.output_dir = output_dir
        self.region_name = region_name
        self.max_workers = min(cpu_count(), 10)  # Set max_workers based on CPU cores (up to 10)

    def download_query_logs(self) -> None:
        """
        Download query logs for multiple workgroups in parallel.

        This method initiates the process of downloading query logs for multiple workgroups in parallel. It performs the
        following steps:

        - Creates the specified output directory or ensures it exists.
        - Retrieves a list of workgroups in AWS Athena.
        - Downloads query logs for each workgroup concurrently using ThreadPoolExecutor.
        - Logs the completion of the download process.

        Note:
        If any errors occur during the download process, they are logged using the `logger.error` method.

        :return: None
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            workgroups = WorkgroupManager(self.region_name).list_workgroups()

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self._download_query_logs_for_workgroup,
                        self.output_dir,
                        workgroup,
                    )
                    for workgroup in workgroups
                ]

                for future in as_completed(futures):
                    try:
                        future.result()  # Get the result of the completed task (this may raise an exception)
                    except Exception as e:
                        logger.error(f"Error during download: {type(e).__name__} - {str(e)}")

            logger.info(f"Download complete. Query logs are stored in: {self.output_dir}")
        except Exception as e:
            logger.error(f"Error during download: {type(e).__name__} - {str(e)}")

    @staticmethod
    def _download_query_logs_for_workgroup(output_dir: str, workgroup: Dict[str, Any]) -> None:
        """
        Download query logs for a specific workgroup.

        :param output_dir: The directory where query logs will be stored.
        :param workgroup: Workgroup information.

        This method is responsible for downloading query logs for a specific workgroup. It executes the following steps:

        - Constructs an AWS CLI command to list query executions for the workgroup and retrieve the query execution IDs.
        - Downloads each query log using the AWS Athena client.
        - Logs the progress and errors during the download process.

        Note:
        If any errors occur during the download process, they are logged using the `logger.error` method.

        :return: None
        """
        logger.info(f"Downloading query logs for workgroup: {workgroup['Name']}")
        query_log_manager = QueryLogManager(output_dir, workgroup["Name"])
        query_log_manager.download_query_logs()


class WorkgroupManager:
    """
    WorkgroupManager is a class for managing AWS Athena workgroups.

    :param region_name: The AWS region name (default is "us-east-1").
    """

    def __init__(self, region_name: str = DEFAULT_REGION_NAME):
        """
        Initialize the WorkgroupManager.

        :param region_name: The AWS region name (default is "us-east-1").
        """
        self.region_name = region_name
        self.athena = boto3.client("athena", region_name=self.region_name)

    def list_workgroups(self) -> List[Dict[str, Any]]:
        """
        List workgroups in AWS Athena.

        :return: A list of workgroup dictionaries.
        """
        try:
            return self.athena.list_work_groups()["WorkGroups"]
        except Exception as e:
            logger.error(f"Error listing workgroups: {str(e)}")
            return []


class QueryLogManager:
    """
    QueryLogManager is a class for downloading query logs for a specific workgroup.

    :param output_dir: The directory where query logs will be stored.
    :param workgroup_name: The name of the workgroup.
    """

    def __init__(self, output_dir: str, workgroup_name: str):
        """
        Initialize the QueryLogManager.

        :param output_dir: The directory where query logs will be stored.
        :param workgroup_name: The name of the workgroup.
        """
        self.output_dir = output_dir
        self.workgroup_name = workgroup_name
        self.athena = boto3.client("athena")

    def download_query_logs(self) -> None:
        """
        Download query logs for a specific workgroup.

        This method executes the following steps:

        - Constructs an AWS CLI command to list query executions for the workgroup and retrieve the query execution IDs.
        - Downloads each query log using the AWS Athena client.
        - Logs the progress and errors during the download process.

        Note:
        If any errors occur during the download process, they are logged using the `logger.error` method.

        :return: None
        """
        aws_cli_command = f"aws athena list-query-executions --work-group {self.workgroup_name} --output json | jq -r"
        try:
            result = subprocess.run(aws_cli_command, shell=True, capture_output=True, text=True, check=True)
            output_list = json.loads(result.stdout)["QueryExecutionIds"]
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running AWS CLI command: {type(e).__name__} - {str(e)}")
            return
        except Exception as e:
            logger.error(f"Error during query log download: {type(e).__name__} - {str(e)}")
            return

        query_execution_ids = [execution_id for execution_id in output_list]

        for query_execution_id in query_execution_ids:
            logger.info(f"Downloading query log for execution ID: {query_execution_id}")
            try:
                response = self.athena.get_query_execution(QueryExecutionId=query_execution_id)
                query_log = json.dumps(
                    response["QueryExecution"],
                    indent=4,
                    default=lambda x: x.isoformat() if isinstance(x, datetime.datetime) else x,
                )

                log_file_path = os.path.join(self.output_dir, f"{self.workgroup_name}-{query_execution_id}.json")
                with open(log_file_path, "w") as f:
                    f.write(query_log)

                logger.info(f"Downloaded query log for execution ID: {query_execution_id}")
            except Exception as e:
                logger.error(f"Error during query log download: {type(e).__name__} - {str(e)}")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the Query Log Downloader script.

    :return: An object containing the parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Download query logs from AWS Athena for multiple workgroups.")
    parser.add_argument(
        "--output-dir",
        "-o",
        default=DEFAULT_DIR_RAW_DATA,
        help="The directory where query logs will be stored. Default is './logs'.",
    )
    parser.add_argument(
        "--region-name",
        "-r",
        default=DEFAULT_REGION_NAME,
        help="The AWS region name where Athena is located. Default is 'us-east-1'.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to download query logs from AWS Athena for multiple workgroups.

    This function serves as the entry point for the Query Log Downloader script.
    The tasks it performs include:

    - **Parsing command-line arguments** to customize the script's behavior.
    - **Initializing** an instance of the `QueryLogDownloader` class with the provided configuration.
    - **Initiating** the process to download query logs for multiple workgroups in parallel.

    **Command-Line Arguments**:

    - `--output-dir, -o` (optional):
        - Description: Directory where query logs will be stored.
        - Default: './logs' if not provided.

    - `--region-name, -r` (optional):
        - Description: AWS region where Athena is located.
        - Default: 'us-east-1' if not provided.

    **Example Usage**:

        $ python query_log_downloader.py --output-dir /path/to/output --region-name us-west-2

    **Notes**:

    - Ensure the directory for storing query logs exists. Otherwise, the script will attempt to create it.
    - Make sure the AWS region name corresponds to a valid AWS region where Athena is accessible.

    :return: None
    """
    args = parse_args()
    downloader = QueryLogDownloader(output_dir=args.output_dir, region_name=args.region_name)
    downloader.download_query_logs()


if __name__ == "__main__":
    main()

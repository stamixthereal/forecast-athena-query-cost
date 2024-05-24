import argparse
import datetime
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import Any, Dict, List

from src.utils.config import AWS_DEFAULT_REGION, DEFAULT_DIR_RAW_DATA, logger, session


class QueryLogDownloader:
    """
    QueryLogDownloader is a class for downloading query logs from AWS Athena in parallel for multiple workgroups.

    :param output_dir: The directory where query logs will be stored.
    :param region_name: The AWS region name (default is "us-east-1").
    """

    def __init__(
        self,
        output_dir: str = DEFAULT_DIR_RAW_DATA,
        region_name: str = AWS_DEFAULT_REGION,
    ):
        """
        Initialize the QueryLogDownloader.

        :param output_dir: The directory where query logs will be stored.
        :param region_name: The AWS region name (default is "us-east-1").
        """
        self.output_dir = output_dir
        self.region_name = region_name
        self.max_workers = min(cpu_count(), 20)

    def download_query_logs(self) -> None:
        """
        Download query logs for multiple workgroups in parallel.

        This method initiates the process of downloading query logs for multiple workgroups in parallel. It performs the
        following steps:

        - Creates the specified output directory or ensures it exists.
        - Retrieves a list of workgroups in AWS Athena.
        - Downloads query logs for each workgroup concurrently using ThreadPoolExecutor.
        - Logs the completion of the download process.

        :return: None
        """
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
                future.result()  #NOTE: Get the result of the completed task (this may raise an exception)

        logger.info(f"Download complete. Query logs are stored in: {self.output_dir}")

    @staticmethod
    def _download_query_logs_for_workgroup(output_dir: str, workgroup: Dict[str, Any]) -> None:
        """
        Download query logs for a specific workgroup.

        :param output_dir: The directory where query logs will be stored.
        :param workgroup: Workgroup information.

        This method is responsible for downloading query logs for a specific workgroup. It executes the following steps:

        - Constructs an AWS CLI command to list query executions for the workgroup and retrieve the query execution IDs.
        - Downloads each query log using the AWS Athena client.
        - Logs the progress during the download process.

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

    def __init__(self, region_name: str = AWS_DEFAULT_REGION):
        """
        Initialize the WorkgroupManager.

        :param region_name: The AWS region name (default is "us-east-1").
        """
        self.region_name = region_name
        self.athena = session.client("athena")

    def list_workgroups(self) -> List[Dict[str, Any]]:
        """
        List workgroups in AWS Athena.

        :return: A list of workgroup dictionaries.
        """
        return self.athena.list_work_groups()["WorkGroups"]


class QueryLogManager:
    """
    QueryLogManager is a class for downloading query logs for a specific workgroup.

    :param output_dir: The directory where query logs will be stored.
    :param workgroup_name: The name of the workgroup.
    """

    def __init__(self, output_dir: str, workgroup_name: str, region_name: str = AWS_DEFAULT_REGION):
        """
        Initialize the QueryLogManager.

        :param output_dir: The directory where query logs will be stored.
        :param workgroup_name: The name of the workgroup.
        """
        self.output_dir = output_dir
        self.workgroup_name = workgroup_name
        self.athena = session.client("athena", region_name=region_name)

    def download_query_logs(self) -> None:
        """
        Download query logs for a specific workgroup.

        This method executes the following steps:

        - Constructs an AWS CLI command to list query executions for the workgroup and retrieve the query execution IDs.
        - Downloads each query log using the AWS Athena client.
        - Logs the progress during the download process.

        :return: None
        """
        query_execution_ids = self._list_query_executions()

        for query_execution_id in query_execution_ids:
            logger.info(f"Downloading query log for execution ID: {query_execution_id}")
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
    
    def _list_query_executions(self) -> List[str]:
        """
        List query execution IDs for the workgroup using boto3 instead of subprocess.

        :return: A list of query execution IDs.
        """
        response = self.athena.list_query_executions(WorkGroup=self.workgroup_name)
        return response.get("QueryExecutionIds", [])


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
        default=AWS_DEFAULT_REGION,
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

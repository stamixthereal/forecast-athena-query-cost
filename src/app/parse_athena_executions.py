import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from typing import List
import streamlit as st

from src.utils.config import DEFAULT_DIR_RAW_DATA, IS_LOCAL_RUN, logger


class QueryLogDownloader:
    """
    QueryLogDownloader is a class for downloading query logs from AWS Athena in parallel for multiple workgroups.
    """

    def __init__(self, session, output_dir: str = DEFAULT_DIR_RAW_DATA):
        """
        Initialize the QueryLogDownloader.
        """
        self.output_dir = output_dir
        self.session = session
        self.max_workers = min(cpu_count(), 20)

    def download_query_logs(self) -> None:
        """
        Download query logs for multiple workgroups.

        This method initiates the process of downloading query logs for multiple workgroups.
        It performs the following steps:
        - Creates the specified output directory or ensures it exists.
        - Retrieves a list of workgroups in AWS Athena.
        - Downloads query logs for each workgroup concurrently using ThreadPoolExecutor if running locally.
        - Otherwise, downloads query logs for each workgroup sequentially.
        - Logs the completion of the download process.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        workgroups = self.session.client("athena").list_work_groups()["WorkGroups"]
        logger.info(workgroups)

        if IS_LOCAL_RUN:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [
                    executor.submit(
                        self._download_query_logs_for_workgroup,
                        workgroup["Name"],
                        self.output_dir,
                    )
                    for workgroup in workgroups
                ]

                for future in as_completed(futures):
                    try:
                        future.result()  # NOTE: Get the result of the completed task (this may raise an exception)
                    except Exception as e:
                        logger.error(f"ERROR WITH {e}")
            logger.info(f"Download complete. Query logs are stored in: {self.output_dir}")
        else:
            for workgroup in workgroups:
                try:
                    self._download_query_logs_for_workgroup(workgroup_name=workgroup["Name"])
                except Exception as e:
                    logger.error(f"ERROR WITH {e}")


    def _download_query_logs_for_workgroup(self, workgroup_name: str, output_dir: str = None) -> None:
        """
        Download query logs for a specific workgroup.

        This method is responsible for downloading query logs for a specific workgroup. It executes the following steps:
        - Constructs an AWS CLI command to list query executions for the workgroup and retrieve the query execution IDs.
        - Downloads each query log using the AWS Athena client.
        - Logs the progress during the download process.
        """
        logger.info(f"Downloading query logs for workgroup: {workgroup_name}")
        query_log_manager = QueryLogManager(self.session, output_dir, workgroup_name)
        query_log_manager.download_query_logs()


class QueryLogManager:
    """
    QueryLogManager is a class for downloading query logs for a specific workgroup.
    """

    def __init__(self, session, output_dir: str, workgroup_name: str):
        """
        Initialize the QueryLogManager.
        """
        self.output_dir = output_dir
        self.workgroup_name = workgroup_name
        self.session = session
        self.athena = session.client("athena")
    
    def download_query_logs(self) -> None:
        """
        Download query logs for a specific workgroup.

        This method executes the following steps:

        - Retrieves the query execution IDs for the workgroup.
        - Downloads each query log using the AWS Athena client.
        - Logs the progress during the download process.
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
            if IS_LOCAL_RUN:
                log_file_path = os.path.join(self.output_dir, f"{self.workgroup_name}-{query_execution_id}.json")
                with open(log_file_path, "w") as f:
                    f.write(query_log)
            else:
                dict_query_log = json.loads(query_log)
                log_file_path = f"{self.workgroup_name}-{query_execution_id}.json"
                st.session_state.query_log_result[log_file_path] = dict_query_log

            logger.info(f"Downloaded query log for execution ID: {query_execution_id}")

    def _list_query_executions(self) -> List[str]:
        """
        List query execution IDs for the workgroup using boto3.
        """
        response = self.athena.list_query_executions(WorkGroup=self.workgroup_name)
        return response.get("QueryExecutionIds", [])


def main(session) -> None:
    """
    Main function to download query logs from AWS Athena for multiple workgroups.

    This function serves as the entry point for the Query Log Downloader script.
    The tasks it performs include:

    - Initializing an instance of the `QueryLogDownloader` class with the provided configuration.
    - Initiating the process to download query logs for multiple workgroups in parallel.
    """
    downloader = QueryLogDownloader(session=session)
    downloader.download_query_logs()


if __name__ == "__main__":
    main()

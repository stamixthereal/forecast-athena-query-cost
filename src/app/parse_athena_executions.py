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

import datetime
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import boto3
import streamlit as st

from src.utils.config import DEFAULT_DIR_RAW_DATA, IS_LOCAL_RUN

logging.basicConfig(level=logging.INFO)


class QueryLogDownloader:

    """QueryLogDownloader is a class for downloading query logs from AWS Athena in parallel for multiple workgroups."""

    def __init__(self, session: boto3.Session, output_dir: str = DEFAULT_DIR_RAW_DATA) -> None:
        self.session = session
        self.output_dir = output_dir
        self.max_workers = min(cpu_count(), 20)
        self.logger = logging.getLogger(self.__class__.__name__)
        Path.mkdir(self.output_dir, exist_ok=True)

    def download_query_logs(self) -> None:
        """Download query logs for multiple workgroups."""
        workgroups = self._get_workgroups()
        if IS_LOCAL_RUN:
            self._download_in_parallel(workgroups)
        else:
            self._download_sequentially(workgroups)
        self.logger.info("All logs have been downloaded!")

    def _get_workgroups(self) -> list[str]:
        """Retrieve a list of workgroups in AWS Athena and log only their names."""
        workgroups = self.session.client("athena").list_work_groups()["WorkGroups"]
        workgroup_names = [wg["Name"] for wg in workgroups]
        self.logger.info(f"Found workgroups: {workgroup_names}")
        return workgroup_names

    def _download_in_parallel(self, workgroups: list[dict]) -> None:
        """Download query logs in parallel using ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._download_query_logs_for_workgroup, workgroup) for workgroup in workgroups]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception:
                    self.logger.exception("Error downloading logs")

    def _download_sequentially(self, workgroups: list[dict]) -> None:
        """Download query logs sequentially."""
        for workgroup in workgroups:
            self._download_query_logs_for_workgroup(workgroup)

    def _download_query_logs_for_workgroup(self, workgroup_name: str) -> None:
        """Download query logs for a specific workgroup."""
        self.logger.info(f"Downloading query logs for workgroup: {workgroup_name}")
        manager = QueryLogManager(self.session, self.output_dir, workgroup_name)
        manager.download_query_logs()
        self.logger.info(f"Logs have been downloaded for workgroup: {workgroup_name}")


class QueryLogManager:

    """QueryLogManager is a class for downloading query logs for a specific workgroup."""

    def __init__(self, session: boto3.Session, output_dir: str, workgroup_name: str) -> None:
        self.output_dir = output_dir
        self.workgroup_name = workgroup_name
        self.session = session
        self.athena = session.client("athena")
        self.logger = logging.getLogger(self.__class__.__name__)

    def download_query_logs(self) -> None:
        """Download query logs for a specific workgroup."""
        query_execution_ids = self._list_query_executions()
        for query_execution_id in query_execution_ids:
            self._download_query_log(query_execution_id)

    def _list_query_executions(self) -> list[str]:
        """List query execution IDs for the workgroup using boto3."""
        response = self.athena.list_query_executions(WorkGroup=self.workgroup_name)
        return response.get("QueryExecutionIds", [])

    def _download_query_log(self, query_execution_id: str) -> None:
        """Download a query log for a specific execution ID."""
        self.logger.info(f"Downloading query log for execution ID: {query_execution_id}")
        response = self.athena.get_query_execution(QueryExecutionId=query_execution_id)
        query_log = json.dumps(
            response["QueryExecution"],
            indent=4,
            default=lambda x: x.isoformat() if isinstance(x, datetime.datetime) else x,
        )
        log_file_path = Path(self.output_dir) / f"{self.workgroup_name}-{query_execution_id}.json"
        self._write_log_to_file(log_file_path, query_log)
        self.logger.info(f"Downloaded query log for execution ID: {query_execution_id}")

    def _write_log_to_file(self, file_path: str, log: str) -> None:
        """Write log to file."""
        if IS_LOCAL_RUN:
            with Path.open(file_path, "w") as f:
                f.write(log)
        else:
            dict_query_log = json.loads(log)
            st.session_state.query_log_result[file_path] = dict_query_log


def main(session: boto3.Session) -> None:
    """Download query logs from AWS Athena for multiple workgroups."""
    downloader = QueryLogDownloader(session)
    downloader.download_query_logs()


if __name__ == "__main__":
    main()

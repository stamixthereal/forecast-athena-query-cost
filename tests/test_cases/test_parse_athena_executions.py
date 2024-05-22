import argparse
import os
import unittest
from unittest.mock import patch, MagicMock
from src.app.parse_athena_executions import (
    QueryLogDownloader,
    WorkgroupManager,
    QueryLogManager,
    main,
)

# Define the output directory as a global variable
OUTPUT_DIR = os.path.abspath("..data/logs") or "us-east-1"


class TestQueryLogDownloader(unittest.TestCase):
    def test_init(self):
        downloader = QueryLogDownloader(output_dir=OUTPUT_DIR)
        self.assertEqual(downloader.output_dir, OUTPUT_DIR)  # Use the global variable
        self.assertEqual(downloader.region_name, "us-east-1")

    @patch("src.app.parse_athena_executions.WorkgroupManager.list_workgroups")
    def test_download_query_logs(self, mock_list_workgroups):
        mock_list_workgroups.return_value = [
            {"Name": "workgroup1"},
            {"Name": "workgroup2"},
        ]
        downloader = QueryLogDownloader()
        downloader.download_query_logs()

    @patch("src.app.parse_athena_executions.WorkgroupManager.list_workgroups")
    def test_download_query_logs_no_workgroups(self, mock_list_workgroups):
        mock_list_workgroups.return_value = []
        downloader = QueryLogDownloader()
        downloader.download_query_logs()


class TestQueryLogManager(unittest.TestCase):
    @patch("boto3.client")
    def test_download_query_logs(self, mock_boto3_client):
        mock_athena = MagicMock()
        mock_boto3_client.return_value = mock_athena
        mock_athena.get_query_execution.return_value = {
            "QueryExecution": {
                "QueryExecutionId": "123",
                "Status": {"State": "SUCCEEDED"},
            }
        }
        manager = QueryLogManager(output_dir=OUTPUT_DIR, workgroup_name="workgroup1")  # Use the global variable
        manager.download_query_logs()

    @patch("boto3.client")
    def test_download_query_logs_failed_execution(self, mock_boto3_client):
        mock_athena = MagicMock()
        mock_boto3_client.return_value = mock_athena
        mock_athena.get_query_execution.return_value = {
            "QueryExecution": {"QueryExecutionId": "123", "Status": {"State": "FAILED"}}
        }
        manager = QueryLogManager(output_dir=OUTPUT_DIR, workgroup_name="workgroup1")  # Use the global variable
        manager.download_query_logs()


class TestMainFunction(unittest.TestCase):
    @patch("src.app.parse_athena_executions.QueryLogDownloader.download_query_logs")
    @patch("src.app.parse_athena_executions.parse_args")
    def test_main(self, mock_parse_args, mock_download_query_logs):
        mock_parse_args.return_value = argparse.Namespace(output_dir=OUTPUT_DIR, region_name="us-east-1")
        main()
        mock_download_query_logs.assert_called_once()


class TestIntegration(unittest.TestCase):
    @patch("src.app.parse_athena_executions.QueryLogDownloader.download_query_logs")
    def test_integration(self, mock_download_query_logs):
        downloader = QueryLogDownloader()
        workgroup_manager = WorkgroupManager()
        query_log_manager = QueryLogManager(output_dir=OUTPUT_DIR, workgroup_name="workgroup1")

        mock_download_query_logs.return_value = None  # Mock the download_query_logs method

        downloader.download_query_logs()
        workgroup_manager.list_workgroups()
        query_log_manager.download_query_logs()


if __name__ == "__main__":
    unittest.main()

import json
import unittest
from unittest.mock import patch, mock_open

from src.app.transform_query_logs import (
    QueryLog,
    QueryLogReader,
    CSVWriter,
    process_file,
)

MOCK_DATA = json.dumps(
    {
        "StatementType": "DML",
        "Status": {"State": "SUCCEEDED"},
        "QueryExecutionId": "id123",
        "WorkGroup": "WG",
        "QueryExecutionContext": {"Database": "DB"},
        "Query": "   SELECT  * FROM  table  ",
        "Statistics": {"DataScannedInBytes": 1024, "TotalExecutionTimeInMillis": 500},
    }
)


class TestQueryLog(unittest.TestCase):
    """Tests for the QueryLog class."""

    def test_is_dml_and_succeeded(self):
        """Test if a DML statement succeeded."""
        data = {"StatementType": "DML", "Status": {"State": "SUCCEEDED"}}
        log = QueryLog(data)
        self.assertTrue(log.is_dml_and_succeeded())

    def test_to_csv_row(self):
        """Test conversion of data to CSV row format."""
        data = {
            "QueryExecutionId": "id123",
            "WorkGroup": "WG",
            "QueryExecutionContext": {"Database": "DB"},
            "Query": "   SELECT  * FROM  table  ",
            "Statistics": {
                "DataScannedInBytes": 1024,
                "TotalExecutionTimeInMillis": 500,
            },
        }
        log = QueryLog(data)
        expected = [
            "id123",
            "random_user",
            "WG",
            "DB",
            "DB",
            "SELECT * FROM table",
            1024,
            500,
        ]
        self.assertEqual(log.to_csv_row(), expected)


class TestQueryLogReader(unittest.TestCase):
    """Tests for the QueryLogReader class."""

    @patch(
        target="builtins.open",
        new_callable=mock_open,
        read_data='{"StatementType": "DML", "Status": {"State": "SUCCEEDED"}}',
    )
    def test_process(self, _mock_file):
        """Test processing of the QueryLogReader."""
        processed = QueryLogReader.process("fakepath.json")
        self.assertIsInstance(processed, QueryLog)


class TestCSVWriter(unittest.TestCase):
    """Tests for the CSVWriter class."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.path.exists", return_value=False)
    @patch("os.makedirs")
    def test_write_row(self, _mock_makedirs, _mock_exists, _mock_file):
        """Test writing of a row with CSVWriter."""
        writer = CSVWriter("output.csv")
        data = [
            "id123",
            "random_user",
            "WG",
            "DB",
            "DB",
            "SELECT * FROM table",
            1024,
            500,
        ]
        writer.write_row(data)
        _mock_file.assert_called_with("output.csv", "a", newline="")


class TestProcessFile(unittest.TestCase):
    """Tests for the process_file function."""

    @patch("builtins.open", new_callable=mock_open, read_data=MOCK_DATA)
    def test_process_file(self, _mock_file):
        """Test processing of a file."""
        row = process_file("fakepath.json")
        self.assertIsNotNone(row)


if __name__ == "__main__":
    unittest.main()

## Overview

The `parse_athena_executions.py` script provides functionality to download query logs from AWS Athena for multiple workgroups in parallel. It uses a combination of the Boto3 Python SDK for AWS and the AWS CLI tool for seamless execution.

## Features

1. **Parallel Downloads**: The script uses Python's concurrent futures module to download logs for multiple Athena workgroups simultaneously.
2. **Error Handling**: Throughout its execution, the script actively logs both progress and error messages, ensuring users are informed of any issues that may arise.
3. **Flexible Configuration**: Users can specify the output directory and AWS region as command-line arguments.

## Pre-requisites

1. **Python 3.7 or above** should be installed.
2. The **Boto3 Python SDK** and **AWS CLI** tools should be properly installed and configured.
3. Ensure AWS credentials are correctly configured either in `~/.aws/credentials` or set up through environment variables or an AWS configuration.

## How to Use

### Command-Line Arguments

1. `--output-dir, -o`: Specifies the directory where the downloaded query logs will be stored. The default is `../../data/raw`.

2. `--region-name, -r`: Specifies the AWS region where Athena is located. The default is `us-east-1`.

### Example

To download query logs and save them to `/path/to/output` directory in the `us-west-2` AWS region, you can run:

```bash
python parse_athena_executions.py --output-dir /path/to/output --region-name us-west-2
```

### Note

- The script will attempt to create the output directory if it doesn't exist.
- Ensure you provide a valid AWS region where Athena is accessible.

## Classes and Methods Overview

1. `QueryLogDownloader`: The main class responsible for downloading query logs for multiple workgroups.
    - `download_query_logs()`: Downloads query logs for multiple workgroups in parallel.
    - `_download_query_logs_for_workgroup()`: Downloads query logs for a specific workgroup.

2. `WorkgroupManager`: A helper class for managing and retrieving Athena workgroups.
    - `list_workgroups()`: Returns a list of all workgroups in AWS Athena.

3. `QueryLogManager`: A class for downloading query logs for a specific Athena workgroup.
    - `download_query_logs()`: Downloads query logs for a specific workgroup using both AWS CLI and Boto3.

## Logging

The script uses Python's logging module. Errors encountered during execution are logged using the `logger.error` method, and general progress is logged using the `logger.info` method.

## Future Enhancements

- Allow pagination for Athena results when the number of executions exceeds a single page.
- Integrate with other AWS services for more advanced use cases, like sending notifications on successful or failed downloads.

## License

This script is provided under the [MIT License](https://opensource.org/licenses/MIT).
## Project Overview

This project comprises a suite of Python scripts designed to analyze SQL query logs and predict the memory usage of SQL queries. It involves downloading query logs from AWS Athena, processing these logs, and using machine learning to predict the memory usage of SQL queries. The project is split into three main parts:

1. **`parse_athena_executions.py`**: Downloads query logs from AWS Athena for multiple workgroups in parallel.
2. **`transform_query_logs.py`**: Processes SQL query logs stored in JSON format and converts them into a CSV file.
3. **`prediction.py`**: Predicts the memory usage of SQL queries based on extracted features using an XGBoost model.

## Part 1: `parse_athena_executions.py`

### Features

- **Parallel Downloads**: Downloads logs for multiple Athena workgroups simultaneously.
- **Error Handling**: Actively logs progress and error messages.
- **Flexible Configuration**: Users can specify output directory and AWS region as command-line arguments.

### Pre-requisites

- Python 3.12+
- Boto3 Python SDK
- AWS CLI

### Usage

```bash
python parse_athena_executions.py --output-dir /path/to/output --region-name us-west-2
```

## Part 2: `transform_query_logs.py`

### Description

Processes SQL query logs from JSON to CSV format, extracting specific data like Query ID, User, Source, etc.

### Requirements

- Python 3.x
- Libraries: argparse, csv, glob, json, logging, os, re

### Usage

```bash
python transform_query_logs.py [--input_dir INPUT_DIRECTORY] [--output_file OUTPUT_CSV_FILE]
```

### Output

A CSV file with columns such as `query_id`, `user_`, `source`, `environment`, `catalog`, `query`, `peak_memory_bytes`, `cpu_time_ms`.

## Part 3: Database Query Performance Prediction Model

### Features

- Data Preprocessing, Feature Extraction, Engineering.
- Data Scaling, Imputation, Dimensionality Reduction.
- Hyperparameter Optimization, Model Training, and Evaluation.

### Technical Requirements

- Python 3.x
- Libraries: pandas, numpy, sklearn, xgboost, optuna, re

### Usage

```python
new_query = """SELECT * FROM "database"."table" WHERE date = '2023-07-07';"""
predicted_memory = model.predict(new_query)
```

### Evaluation

Performance evaluated using MSE, MAE, and R^2 metrics.

## General Usage

### Running the Complete Pipeline

The project includes a menu-driven CLI to run different parts of the pipeline:

1. **Process Athena Executions**
2. **Transform Query Logs**
3. **Run The Whole Pipeline**

### Example

```bash
Choose an action:
1. Process Athena Executions
2. Transform Query Logs
3. Run The Whole Pipeline
4. Make Prediction
5. Exit

Enter your choice (1/2/3/4/5): 2
```

### Execution

Run the main script to access the menu:

```bash
python main.py
```

## Logging and Error Handling

Each script logs its progress and errors, ensuring smooth execution and easy troubleshooting.

## Customization

Parts of the scripts, such as user data fetching methods, are placeholders for customization. Look for "TODO" comments in the code.

## Future Enhancements

- Pagination for large Athena result sets.
- Integration with other AWS services for advanced functionalities.

## License

The project is released under the [MIT License](https://opensource.org/licenses/MIT).

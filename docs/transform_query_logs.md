### Description

The script `transform_query_logs.py` processes SQL query logs stored in JSON format from a specified input directory. The processed data is then written to a designated CSV file. The main functionality includes extracting specific data from each JSON file (like Query ID, User, Source, etc.), converting it to a CSV row format, and then writing these rows to a CSV file.

### Requirements

- Python 3.x
- Required Python libraries:
  - argparse
  - csv
  - glob
  - json
  - logging
  - os
  - re

### Usage

```bash
python transform_query_logs.py [--input_dir INPUT_DIRECTORY] [--output_file OUTPUT_CSV_FILE]
```

#### Arguments:

- `--input_dir`: (Optional) The path to the input directory containing JSON files with SQL query logs.
  - Default: `../../data/raw`
  - Example: `--input_dir "./logs"`

- `--output_file`: (Optional) The path to the output CSV file where the processed data will be written.
  - Default: `../../data/processed/processed_data.csv`
  - Example: `--output_file "./processed_data.csv"`

### Output

A CSV file with the following columns:

- `query_id`: The ID of the executed query.
- `user_`: Identifier of the user executing the query. (Currently uses a placeholder value.)
- `source`: The workgroup where the query was executed.
- `environment`: The database context where the query ran.
- `catalog`: Synonymous with the `environment` field, represents the database.
- `query`: The executed SQL query string.
- `peak_memory_bytes`: Memory used during the execution of the query.
- `cpu_time_ms`: Total CPU time taken to execute the query in milliseconds.

Note: Queries with `peak_memory_bytes` equal to 0 are not included in the CSV.

### Logging

The script also provides a logging mechanism that logs the processing steps, including:

- Starting and completing the processing.
- Number of files found for processing.
- Progress update for every 5% increment in processed files.
- Successful write operations to the CSV.
- Warnings or errors encountered during processing.

### Customization

Certain parts of the code, like the method to fetch user data, are left as placeholders for further customization. Check the code comments marked with "TODO" for areas that might need modification or improvement as per your use-case.

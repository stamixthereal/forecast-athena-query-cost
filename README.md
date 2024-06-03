# Forecast Athena SQL Queries

This project comprises a suite of Python scripts designed to analyze SQL query logs and predict the memory usage of SQL queries. It involves downloading query logs from AWS Athena, processing these logs, and using machine learning to predict the memory usage of SQL queries.

## Current Architecture Diagram

![current-architecture](images/current-architecture.png)

## Getting Started

> Make sure you have Docker installed and configured properly for these commands to work as expected. Additionally, ensure that the scripts in the scripts directory are correctly configured and that any necessary environment variables or configurations are set before running the commands

Follow these steps to set up the project and start using it:

### 0. (Optional) Set Up the Virtual Environment and Install Dependencies

```bash
make local-venv-setup
```

This command will create a virtual environment named venv and install the dependencies listed in requirements.txt.

### 1. Start the Application using Docker

```bash
make start-app-docker
```

This command will execute the start-app.sh script located in the scripts directory, which presumably starts your application using Docker.

### 2. Clean Up Resources

If necessary, you can run the following command to clean up any resources used by your application:

```bash
make clean-up-resources
```

This will execute the clean-up-resources.sh script located in the scripts directory.

## License

This project is licensed under the [Apache License 2.0](LICENSE)

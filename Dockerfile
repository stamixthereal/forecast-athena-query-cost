FROM python:3.12.3-slim

# Combine package installations
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        awscli \
        jq \
    && rm -rf /var/lib/apt/lists/*

COPY . /forecast-sql-query

RUN pip install --no-cache-dir -r /forecast-sql-query/requirements.txt

ENV ENV_FILE_LOCATION .env

WORKDIR /forecast-sql-query

# Use the loaded credentials to configure AWS CLI
RUN aws configure set aws_access_key_id $(grep -oP '^AWS_ACCESS_KEY_ID=\K.*' $ENV_FILE_LOCATION) \
    && aws configure set aws_secret_access_key $(grep -oP '^AWS_SECRET_ACCESS_KEY=\K.*' $ENV_FILE_LOCATION)

# Set up a volume to sync local changes with the container
VOLUME /forecast-sql-query

CMD ["python", "main.py"]

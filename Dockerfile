FROM python:3.12.3-slim

# Combine package installations
RUN apt-get -qq update \
    && apt-get install -y --no-install-recommends \
        awscli \
        jq \
    && apt-get -qq -y install curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://get.docker.com | sh

COPY . /forecast-sql-query

RUN pip install --no-cache-dir -r /forecast-sql-query/requirements.txt

WORKDIR /forecast-sql-query

CMD ["streamlit", "run", "main.py"]

#!/bin/bash

# Define a directory to store query logs
output_dir="/logs"

# Create a timestamp directory to organize the downloaded logs
timestamp=$(date +"%Y%m%d%H%M%S")
log_dir="$output_dir"
mkdir -p "$log_dir"

# Get all workgroups
workgroups=$(aws athena list-work-groups --output json | jq -r '.WorkGroups[].Name')

for workgroup in $workgroups; do
  echo "Downloading query logs for workgroup: $workgroup"

  # Get all query execution IDs for the workgroup
  query_execution_ids=$(aws athena list-query-executions --work-group "$workgroup" --output json | jq -r '.QueryExecutionIds[]')

  # Download all query logs
  for query_execution_id in $query_execution_ids; do
    aws athena get-query-execution --query-execution-id "$query_execution_id" --output json > "$log_dir/$workgroup-$query_execution_id.json"
  done
done

echo "Download complete. Query logs are stored in: $log_dir"

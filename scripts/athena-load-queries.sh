#!/bin/bash

: '
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

Copyright [2024] [Stanislav Kazanov]
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'

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

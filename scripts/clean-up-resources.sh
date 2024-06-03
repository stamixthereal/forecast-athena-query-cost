#!/usr/bin/env bash

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

echo "Cleaning __pychache__ directories..."
sudo find . -type d -name __pycache__ -exec rm -r {} +

echo "Cleaning up processed data directory..."
sudo find data/processed -type f ! -name ".gitkeep" -delete

echo "Cleaning up raw data directory..."
sudo find data/raw -type f ! -name ".gitkeep" -delete

echo "Cleaning up other python cache"
sudo rm -rf .pytest_cache
sudo rm -rf .ruff_cache

echo "Cleaning up docker resourses"
# Stop running containers
if [[ -n $(sudo docker ps -q) ]]; then
    sudo docker stop $(sudo docker ps -q) || true
fi

# Remove stopped containers
if [[ -n $(sudo docker ps -a -q) ]]; then
    sudo docker rm $(sudo docker ps -a -q) || true
fi

# Remove unused images
if [[ -n $(sudo docker images -q) ]]; then
    sudo docker rmi $(sudo docker images -q) || true
fi
sudo docker image prune -a -f
sudo docker container prune -f
sudo docker volume prune -f
sudo docker system prune -f

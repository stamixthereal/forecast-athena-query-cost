#!/usr/bin/env bash

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

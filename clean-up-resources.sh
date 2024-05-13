#!/usr/bin/env bash

echo "Cleaning __pychache__ directories..."
sudo find . -type d -name __pycache__ -exec rm -r {} +

echo "Cleaning up processed data directory..."
sudo find data/processed -type f ! -name ".gitkeep" -delete

echo "Cleaning up raw data directory..."
sudo find data/raw -type f ! -name ".gitkeep" -delete

echo "Cleaning up ML model directory..."
sudo find src/model -type f ! -name ".gitkeep" -delete

echo "Cleaning up ML model directory..."
sudo find src/model -type f ! -name ".gitkeep" -delete

echo "Cleaning up other python cache"
sudo rm -rf .pytest_cache
sudo rm -rf .ruff_cache

echo "Cleaning up docker resourses"
sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)
sudo docker rmi $(sudo docker images -q)
sudo docker image prune -a
sudo docker container prune
sudo docker volume prune
sudo docker system prune

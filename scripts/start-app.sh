#!/usr/bin/env bash

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

# Build the Docker image
sudo docker build -t forecast-sql-image .

# Run the Docker Container
sudo docker run -v .:/forecast-sql-query --rm --name forecast-sql-container -it forecast-sql-image

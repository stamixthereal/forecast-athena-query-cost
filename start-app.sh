#!/usr/bin/env bash

sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)
sudo docker rmi $(sudo docker images -q)

# Build the Docker image
sudo docker build -t forecast-sql-image .

# Run the Docker Container
sudo docker run -v .:/forecast-sql-query --rm --name forecast-sql-container -it forecast-sql-image 
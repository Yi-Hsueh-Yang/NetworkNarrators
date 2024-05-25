#!/bin/bash

# Name of the Docker image
IMAGE_NAME="movie-recommender"

# Tag for the Docker image, using current date and time (e.g., 20240101-120101)
IMAGE_TAG=$(date +"%Y%m%d-%H%M%S")

# Path to the directory containing the Dockerfile
DOCKERFILE_PATH="/home/team18/deploy/"

TARGET_PORT=8082

echo "Building Docker image ${IMAGE_NAME}:${IMAGE_TAG}..."

# Running the Docker build command
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ${DOCKERFILE_PATH}

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Docker image ${IMAGE_NAME}:${IMAGE_TAG} built successfully."
    CONTAINERS_USING_IMAGE=$(docker ps --filter "status=running" --format '{{.ID}}' | xargs -I {} docker inspect --format '{{.Id}} {{.NetworkSettings.Ports}}' {} | grep '8081' | awk '{print $1}')
    
    if [ ! -z "$CONTAINERS_USING_IMAGE" ]; then
        echo "Stopping container using port 8081 ..."
        docker stop $CONTAINERS_USING_IMAGE
        
        if [ $? -eq 0 ]; then
            echo "Container using port 8081 stopped successfully."
        else
            echo "Failed to stop container $CONTAINERS_USING_IMAGE."
            exit 1 
        fi

    else
        echo "No container is currently running on port 8081."
    fi

    echo "Running Docker container from image: ${IMAGE_NAME}:${IMAGE_TAG} on Port 8081"
    
    docker run -d -p 8081:${TARGET_PORT} ${IMAGE_NAME}:${IMAGE_TAG}
    
    sleep 300
    
    CONTAINERS_USING_IMAGE=$(docker ps --filter "status=running" --format '{{.ID}}' | xargs -I {} docker inspect --format '{{.Id}} {{.NetworkSettings.Ports}}' {} | grep '8083' | awk '{print $1}')
    
    if [ ! -z "$CONTAINERS_USING_IMAGE" ]; then
        echo "Stopping container using port 8083 ..."
        docker stop $CONTAINERS_USING_IMAGE
        
        
        if [ $? -eq 0 ]; then
            echo "Container using port 8083 stopped successfully."
        else
            echo "Failed to stop container $CONTAINERS_USING_IMAGE."
            
            exit 1 
        fi

    else
        echo "No container is currently running on port 8083."
    fi
    
    echo "Running Docker container from image: ${IMAGE_NAME}:${IMAGE_TAG} on Port 8083 "
    
    docker run -d -p 8083:${TARGET_PORT} ${IMAGE_NAME}:${IMAGE_TAG}
    
    if [ $? -eq 0 ]; then
    	echo "Docker container started successfully"
    else
    	echo "Failed to start docker container"
    	exit 1
    fi
else
    echo "Failed to build Docker image ${IMAGE_NAME}:${IMAGE_TAG}."
    exit 1
fi

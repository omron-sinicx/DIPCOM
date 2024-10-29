#!/bin/bash
# Generates a docker image with the relevant settings for the DOCKER_PROJECT.
# Context-sensitive behaviour: If the <user> parameter "gitlab-ci" is used, 
# the script builds the image without trying to download it.
#
# Usage: ./BUILD-DOCKER-IMAGE.bash <optional: user>
#
# @param <user> [optional parameter] for docker container naming during spin-up and to set different behavior.
#                  Default value: $USER - the image is pulled from repo, or built as fallback.
#                  If <user> is "gitlab-ci" the image is directly build from scratch - as if done in gitlab-ci.
################################################################################

# Set the Docker container name from a project name (first argument).
# If no argument is given, use the current user name as the project name.
# Source the environment file if it exists
if [ -f "docker.env" ]; then
    source docker.env
fi

# Set the Docker container name from a project name (first argument).
# If no argument is given, use DOCKER_PROJECT_NAME from env file, or fall back to current user name
PROJECT=$1
if [ -z "${PROJECT}" ]; then
    if [ -n "${DOCKER_PROJECT_NAME}" ]; then
        PROJECT=${DOCKER_PROJECT_NAME}
    else
        PROJECT=${USER}
    fi
fi
CONTAINER="${PROJECT}-dipcom_env-1"
echo "$0: PROJECT=${PROJECT}"
echo "$0: CONTAINER=${CONTAINER}"

# Stop and remove the Docker container.
EXISTING_DOCKER_CONTAINER_ID=`docker ps -aq -f name=${CONTAINER}`
if [ ! -z "${EXISTING_DOCKER_CONTAINER_ID}" ]; then
  echo "Stop the container ${CONTAINER} with ID: ${EXISTING_DOCKER_CONTAINER_ID}."
  docker stop ${EXISTING_DOCKER_CONTAINER_ID}
  echo "Remove the container ${CONTAINER} with ID: ${EXISTING_DOCKER_CONTAINER_ID}."
  docker rm ${EXISTING_DOCKER_CONTAINER_ID}
fi

# try to pull it from the server. If this fails, it has to be build locally.
docker compose -p ${PROJECT} -f ./docker/docker-compose.yml build

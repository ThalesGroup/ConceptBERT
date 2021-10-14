#/bin/bash
set -e

BASE_PATH=$(dirname "$0")/..

TAG_VERSION=$(cat $BASE_PATH/dist/__version__)
FULL_IMAGE_NAME=$IMAGE_NAME:${TAG_VERSION}
# Use local collaborative address to deploy the docker image in kubernetes
COLLABORATIVE_REGISTRY="collaborative-docker-registry.collaborative.vlan:5100/"

if [ -z "$CI" ]
then
    # Build from local node
    echo "Building locally using docker client"
    docker build --build-arg http_proxy=$HTTP_PROXY \
                 --build-arg https_proxy=$HTTPS_PROXY \
                 --build-arg no_proxy=$NO_PROXY \
                 -t $FULL_IMAGE_NAME \
                 -f $BASE_PATH/deployment/Dockerfile \
                 .
                 
    echo "Pushing to collaborative docker registry"
    docker tag $FULL_IMAGE_NAME $COLLABORATIVE_REGISTRY$FULL_IMAGE_NAME
    docker push $COLLABORATIVE_REGISTRY$FULL_IMAGE_NAME
else
    # Build from Gitlab-CI
    echo "Building and pushing to collaborative docker registry from CI using kaniko (since no priviledged mode)"
    /kaniko/executor --context $CI_PROJECT_DIR \
             --dockerfile $BASE_PATH/deployment/Dockerfile \
             --destination ${DOCKER_REGISTRY}/$FULL_IMAGE_NAME \
             --build-arg docker_registry="$DOCKER_REGISTRY_MIRROR/" \
             --build-arg http_proxy=$http_proxy \
             --build-arg https_proxy=$https_proxy \
             --build-arg no_proxy=$no_proxy \
             --insecure \
             --skip-tls-verify \
             --insecure-registry=${DOCKER_REGISTRY}
fi

printf "\n\e[1;32mBuilded ${FULL_IMAGE_NAME} docker image successfully.\e[0m\n\n"

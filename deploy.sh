#/bin/bash
set -e

BASE_PATH=$(dirname "$0")

source $BASE_PATH/environment.sh

if [ -z "$1" ]; then
  JOB_TEMPLATE_FILE_PATH=$BASE_PATH/$JOB_TEMPLATE_FILE_PATH
else
  JOB_TEMPLATE_FILE_PATH=$BASE_PATH/$1
fi

if [[ -z $JOB_NAME ]]; then
  export JOB_NAME=$(echo $JOB_TEMPLATE_FILE_PATH | rev | cut -d'/' -f1 | rev | sed 's/\./-/g')
fi

echo "JOB_TEMPLATE_FILE_PATH is ($JOB_TEMPLATE_FILE_PATH)"
echo "JOB_NAME ($JOB_NAME)"

echo "Package python wheel"  
$BASE_PATH/ci/package.sh

echo "Build docker image"
$BASE_PATH/ci/build-docker.sh

echo "Deploy kubernetes job"
$BASE_PATH/ci/deploy.sh

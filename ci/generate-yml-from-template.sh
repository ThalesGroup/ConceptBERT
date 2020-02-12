#/bin/bash

BASE_PATH=$(dirname "$0")/..

TAG_VERSION=$(cat $BASE_PATH/dist/__version__)
JOB_TEMPLATE_PATH=$BASE_PATH/${JOB_TEMPLATE_FILE_PATH}

JOB_TEMPLATE_FILE_NAME=$(basename -- "$JOB_TEMPLATE_PATH")
JOB_NAME_PREFIX="${JOB_TEMPLATE_FILE_NAME%.*}"
echo "Generating yml ${TAG_VERSION} with template ${JOB_TEMPLATE_PATH}"

MY_JOB_NAME=${JOB_NAME_PREFIX}-${TAG_VERSION}
JOB_FILE_NAME=${MY_JOB_NAME}.yml
JOB_FOLDER=$BASE_PATH/jobs

echo "Cleaning last build"
rm -f $BASE_PATH/jobs/${JOB_FILE_NAME}

FULL_IMAGE_NAME=$IMAGE_NAME:${TAG_VERSION}
    
mkdir -p $JOB_FOLDER
cp $JOB_TEMPLATE_PATH $JOB_FOLDER/${JOB_FILE_NAME}
sed -i 's/JOB_ID_PLACEHOLDER/'${TAG_VERSION}'/g' $JOB_FOLDER/${JOB_FILE_NAME}
sed -i 's/IMAGE_NAME_PLACEHOLDER/'${FULL_IMAGE_NAME//\//\\/}'/g' $JOB_FOLDER/${JOB_FILE_NAME}
sed -i 's/JOB_NAME_PLACEHOLDER/'${JOB_NAME//\//\\/}'/g' $JOB_FOLDER/${JOB_FILE_NAME}
sed -i 's/NAMESPACE_PLACEHOLDER/'${NAMESPACE//\//\\/}'/g' $JOB_FOLDER/${JOB_FILE_NAME}
sed -i 's/NAS_SHARED_FOLDER_PLACEHOLDER/'${NAS_SHARED_FOLDER//\//\\/}'/g' $JOB_FOLDER/${JOB_FILE_NAME}

if [ -z "$CI" ]
then
  # Build from local node
  export AUTHOR_NAME=$(git config user.name)
  export AUTHOR_EMAIL=$(git config user.email)
else
  # Build from Gitlab-CI
  export AUTHOR_NAME=$(git log -1 --pretty=format:'%an')
  export AUTHOR_EMAIL=$(git log -1 --pretty=format:'%ae')
fi

echo "${AUTHOR_NAME} ${AUTHOR_EMAIL}"
sed -i 's/AUTHOR_NAME_PLACEHOLDER/'"${AUTHOR_NAME//\//\\/}"'/g' $JOB_FOLDER/${JOB_FILE_NAME}
sed -i 's/AUTHOR_EMAIL_PLACEHOLDER/'"${AUTHOR_EMAIL//\//\\/}"'/g' $JOB_FOLDER/${JOB_FILE_NAME}

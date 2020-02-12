#/bin/bash
set -e

BASE_PATH=$(dirname "$0")/..

# Use downloader template
export JOB_TEMPLATE_FILE_PATH=deployment/job-outputs-downloader.tpl
source $BASE_PATH/ci/generate-yml-from-template.sh

if [ -z "$CI" ]
then
  echo "Please download outputs manually from NAS ${NAS_SHARED_FOLDER}/outputs/${JOB_NAME}-${TAG_VERSION}"
else  
  OUTPUTS_FOLDER=$BASE_PATH/outputs/${JOB_NAME}-${TAG_VERSION}
  rm -rf $OUTPUTS_FOLDER
  mkdir -p $OUTPUTS_FOLDER

  # Create
  kubectl create -f $JOB_FOLDER/${JOB_FILE_NAME}
  
  echo "Waiting for pod ${DEBUG_POD_NAME}..."
  kubectl wait pod/download-${JOB_NAME}-${TAG_VERSION} -n ${NAMESPACE} --for=condition=ready --timeout=600s 
  
  kubectl cp ${NAMESPACE}/download-${JOB_NAME}-${TAG_VERSION}:/outputs $OUTPUTS_FOLDER -n ${NAMESPACE}
  
  # Delete
  kubectl delete -f $JOB_FOLDER/${JOB_FILE_NAME}

  printf "\n\e[1;32mDownloaded outputs successfully.\e[0m\n\n"
fi 


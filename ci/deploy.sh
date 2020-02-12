#/bin/bash

BASE_PATH=$(dirname "$0")/..

source $BASE_PATH/ci/generate-yml-from-template.sh

JOB_URL="https://dashboard.k8s.collaborative.local/#!/job/${NAMESPACE}/${MY_JOB_NAME}?namespace=${NAMESPACE}"

if [ -z "$CI" ]
then
  # Deploy from local node
  printf "\n\n\e[1;35mTo run job, use command:\e[0m\n\n"
  printf "\n\e[1;35m    kubectl create -f $JOB_FOLDER/${JOB_FILE_NAME}\e[0m\n\n"

  printf "\e[1;35mTo delete job, use command:\e[0m\n\n"
  printf "\n\e[1;35m    kubectl delete -f $JOB_FOLDER/${JOB_FILE_NAME}\e[0m\n\n"

  printf "\n\e[1;32mDeploying $JOB_FOLDER/${JOB_FILE_NAME} to kubernetes cluster...\e[0m\n\n"

  kubectl create -f $JOB_FOLDER/${JOB_FILE_NAME}

  printf "\n\e[1;32mDeployed successfully.\e[0m\n\n"
  printf "\n\e[1;35mJob Url:\e[0m\n\n"
  printf "\n\e[1;35m    $JOB_URL.\e[0m\n\n"
else
  # Deploy from Gitlab-CI
  kubectl create -f $JOB_FOLDER/${JOB_FILE_NAME}
  
  retry=0
  while :
  do
    echo "Getting pod name from job ${JOB_NAME}-${TAG_VERSION}..."
    DEBUG_POD_NAME=$(kubectl get pod -l job-name=${JOB_NAME}-${TAG_VERSION} -n ${NAMESPACE} -o=jsonpath='{.items[0].metadata.name}');RESULT=$?
    
    echo "result: ${RESULT}"
    echo "name: ${DEBUG_POD_NAME}"
    [[ $RESULT = 0 ]] && break || ((n++))
    
    echo "retrying in 30 seconds..."
    sleep 30
    
    (( n >= 20160 )) && exit 1
  done

  echo "Waiting for pod ${DEBUG_POD_NAME}..."
  kubectl wait pod/$DEBUG_POD_NAME -n ${NAMESPACE} --for=condition=ready --timeout=600s 

  echo "Tailling logs..."
  kubectl logs -f $DEBUG_POD_NAME -n ${NAMESPACE}
  
  printf "\n\e[1;32mCompleted ${JOB_FILE_NAME} job.\e[0m\n\n"
fi 


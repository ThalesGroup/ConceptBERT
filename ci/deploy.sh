#/bin/bash

BASE_PATH=$(dirname "$0")/..

source $BASE_PATH/ci/generate-yml-from-template.sh

deploy_locally () {
  local job_url="https://dashboard.prod.kubernetes.collaborative.vlan/#!/job/${NAMESPACE}/${MY_JOB_NAME}?namespace=${NAMESPACE}"

  printf "\n\n\e[1;35mTo run job, use command:\e[0m\n\n"
  printf "\n\e[1;35m    kubectl create -f $JOB_FOLDER/${JOB_FILE_NAME}\e[0m\n\n"

  printf "\e[1;35mTo delete job, use command:\e[0m\n\n"
  printf "\n\e[1;35m    kubectl delete -f $JOB_FOLDER/${JOB_FILE_NAME}\e[0m\n\n"

  printf "\n\e[1;32mDeploying $JOB_FOLDER/${JOB_FILE_NAME} to kubernetes cluster...\e[0m\n\n"

  kubectl create -f $JOB_FOLDER/${JOB_FILE_NAME}

  printf "\n\e[1;32mDeployed successfully.\e[0m\n\n"
  printf "\n\e[1;35mJob Url:\e[0m\n\n"
  printf "\n\e[1;35m    $job_url.\e[0m\n\n"
}

exit_on_error() {
  exit_code=$1
  message=$2
  if [ $exit_code -ne 0 ]; then
      >&2 echo $message
      exit $exit_code
  fi
}

evaluate_job_pods_status () {
  local k8s_job_name=$1
  is_job_active=
  
  echo "Evaluating pod status for job ${k8s_job_name}"
  
  job_has_active_pods=$(kubectl get jobs $k8s_job_name -n ${NAMESPACE} -o jsonpath='{.status.active}' --ignore-not-found)    
  echo "job_has_active_pods : ${job_has_active_pods}"

  job_has_failed_pods=$(kubectl get jobs $k8s_job_name -n ${NAMESPACE} -o jsonpath='{.status.failed}' --ignore-not-found)
  echo "job_has_failed_pods : ${job_has_failed_pods}"

  job_has_succeeded_pods=$(kubectl get jobs $k8s_job_name -n ${NAMESPACE} -o jsonpath='{.status.succeeded}' --ignore-not-found)
  echo "job_has_succeeded_pods : ${job_has_succeeded_pods}"

  if [ -z "${job_has_active_pods}" ]; then
    if [ -z "${job_has_failed_pods}" ] && [ -z "${job_has_succeeded_pods}" ]; then
      # Verifying if job exists (might have been manually deleted).
      kubectl get jobs $k8s_job_name -n ${NAMESPACE}
      exit_on_error $? "Job ${k8s_job_name} does not exist anymore. Job was deleted."
  
      # If job do exists, consider it may not have started anything yet.
      echo "Job has not started any pod yet (considered still active)."
      is_job_active=true
    else
      echo "Job is not active anymore."
      break
    fi
  else
    echo "Job is still active."
    is_job_active=true
  fi
}

evaluate_final_job_status () {
  local k8s_job_name=$1
  
  is_job_completed=$(kubectl get jobs $k8s_job_name -n ${NAMESPACE} -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' --ignore-not-found)
  
  if [[ "$is_job_completed" != "True" ]]; then
    printf "\n\e[1;91mJob ${k8s_job_name} has failed.\e[0m\n\n"
    exit 1
  else
    printf "\n\e[1;32mCompleted ${k8s_job_name} job.\e[0m\n\n"
  fi
}

still_running() {
    echo "Started log heartbeat"
    SLEEP_FOR_SECS=900 # 15 minutes
    while true ; do
        sleep ${SLEEP_FOR_SECS}
        echo "(still running...)"
    done
}

monitor_pod_execution () {
  local current_pod_name=$1
  
  if [ "$current_pod_name" ]; then
    echo "Found active pod ${current_pod_name}"
    
    # Wait for pod to be ready
    echo "Waiting for pod ${current_pod_name}..."
    kubectl wait pod/$current_pod_name -n ${NAMESPACE} --for=condition=ready --timeout=600s
    
    echo "Tailling logs..."
    kubectl logs -f $current_pod_name -n ${NAMESPACE}
    
    echo "Pod $current_pod_name state messages:"
    kubectl get pod $current_pod_name -n ${NAMESPACE} -o go-template='{{range .status.containerStatuses}}{{printf "\n" }}{{ range $key, $value := .state }} State: {{$key}} {{printf "\n" }} Message: {{ index $value "message" }}{{printf "\n" }} Reason: {{ index $value "reason" }}{{printf "\n " }}{{printf "\n " }}{{end}}{{end}}'
  else  
    echo "No active pod found."
  fi
}

deploy_from_ci () {  
  # Add minimum of log to make sure job is not killed after 30 minutes if it does not output logs.
  # Reference: https://gitlab.com/gitlab-org/gitlab/-/issues/25359
  still_running &

  local k8s_job_name=${JOB_NAME}-${TAG_VERSION}
  
  # Deploy job to kubernetes cluster
  kubectl create -f $JOB_FOLDER/${JOB_FILE_NAME}
  exit_on_error $? "Job ${k8s_job_name} was already deployed on the cluster. Please change version or job name."

  local is_job_active=true
    
  # While job is still active
  while [ "${is_job_active}" ]
  do
    echo "waiting 5 seconds..."
    sleep 5
    
    # Retrieve non-failed pod from job
    current_pod_name=$(kubectl get pod -l job-name=$k8s_job_name -n ${NAMESPACE} --field-selector=status.phase!=Failed -o=jsonpath='{.items[*].metadata.name}' --ignore-not-found)
    
    # Monitor pod execution
    monitor_pod_execution $current_pod_name
    
    # Verify if job is still active (will update local variable $is_job_active)
    evaluate_job_pods_status $k8s_job_name
  done
  
  kill $(jobs -p)

  # Verify if job succeeded or failed
  evaluate_final_job_status $k8s_job_name
}

if [ -z "$CI" ]
then
  # Deploy from local node
  deploy_locally
else
  # Deploy from Gitlab-CI
  deploy_from_ci
fi 


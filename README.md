# ConceptBert-Job
       
This version of the project is based on Kubernetes cluster.
The ConceptBert documentation is available in the [ConceptBert folder](conceptBert/README.md).

## Versioning

Version is based on Git commit hash, timestamp of build, and tag using [setuptools-scm](https://pypi.org/project/setuptools-scm/)

**Please note that unique job identifier/name will be generated using, in part, the git commit hash and timestamp (second).**

**If you use custom project folder instead of the conceptbert-job, make sure you put a '\_\_init\_\_.py' in the folder to add the code in the packaging**

Example:
```
# <job_name>-<last_tag_version>.<number_of_commit_since_last_tag>-<commit_hash>.d<YYYYMMDDHHMMSS>
conceptbert-job-0.1.dev10-h583cdddc8.d20191203090038
```

## Build & Deployment

Make sure project values are exported as environment variable in **environment.sh** file.


### Manual Deployment (RECOMMENDED)

To build and deploy, simply run:
```
# Build and deploy to the cluster
./deploy.sh <template-file-path>
```

Previous commands will result in:
1. Build a .whl (wheel) file containing the application code and copy it in the dist/ folder;
2. Build a Docker image containing the application code and all its dependencies (using deployment/Dockerfile);
3. Push the Docker image to the docker registry
4. Create a kubernetes job file from the (using deployment/kilbert-job.tpl template) and copy it in the jobs/ folder;
5. Deploy the kubernetes job to the collaborative cluster.

## Stop/delete job

Once job is completed (or failed to complete) and you don't to access its logs anymore, you have to manually delete it using the following command.

```
kubectl delete -f ./jobs/<JOB_NAME>.yml
```

## Outputs

This project is configured to redirect print output to //isilon.storage.vlan/HUMAN_AI_DIALOG_SHARED/vilbert/outputs/.

## Inputs

This project is configured to read inputs from //isilon.storage.vlan/HUMAN_AI_DIALOG_SHARED/vilbert/data2/

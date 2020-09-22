# ConceptBert
   
For kubernetes usages, refer to the [wiki](https://wiki-trt.thales-systems.ca/bin/view/Trt%20Quebec/Engineering/Infrastructure/VLANs/Collaborative_VLAN/Kubernetes/Usage/).

The kilbert documentation is available in the [kilbert folder](kilbert/README.md).

## Versioning

Version is based on Git commit hash, timestamp of build, and tag using [setuptools-scm](https://pypi.org/project/setuptools-scm/)

**Please note that unique job identifier/name will be generated using, in part, the git commit hash and timestamp (second).**

Example:
```
# <job_name>-<last_tag_version>.<number_of_commit_since_last_tag>-<commit_hash>.d<YYYYMMDDHHMMSS>
vilbert-job-0.1.dev10-g3472458.d20191203090038
```

## Build & Deployment

Make sure project values are exported as environment variable in **environment.sh** file.


### Manual Deployment (RECOMMENDED)

To build and deploy, simply run:
```
# Renew kubectl token (expire every 7 days)
1. Login to https://kubectl.k8s.collaborative.local/login
2. Copy the text from second black box into a terminal
3. Try with ```kubectl get nodes```

# Login using your TGI credentials
docker login ${collaborative_docker_registry}

# Build and deploy to the cluster
./deploy.sh <template-file-path>
```

Previous commands will result in:
1. Build a .whl (wheel) file containing the application code and copy it in the dist/ folder;
2. Build a Docker image containing the application code and all its dependencies (using deployment/Dockerfile);
3. Push the Docker image to the [collaborative docker registry](http://collaborative-docker-registry.collaborative.local/)
4. Create a kubernetes job file from the (using deployment/kilbert-job.tpl template) and copy it in the jobs/ folder;
5. Deploy the kubernetes job to the collaborative cluster.


### CI/CD Deployment (GitLab-CI) (NOT USED)

The .gitlab-ci.yml can be used by GitLab-CI to automatically deploy the job after each push.

The following environment variables need to be set in the project/group [CI/CD setting](https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/kilbert/-/settings/ci_cd):
1. variable DOCKER_REGISTRY=collaborative-docker-registry.collaborative.local:5100
2. variable DOCKER_REGISTRY_USER=<namespace service account username>
3. variable DOCKER_REGISTRY_PASSWORD=<namespace service account password>
4. variable DOCKER_REGISTRY_MIRROR=common-docker-registry.common.local:5100
5. file KUBECONFIG=<namespace CI/CD service account kubernetes configuration>

To generate the KUBECONFIG file, follow instructions at [Service Account](https://wiki-trt.thales-systems.ca/bin/view/Trt%20Quebec/Engineering/Infrastructure/VLANs/Collaborative_VLAN/Kubernetes/Usage/#HCreateServiceAccount). See also ci/ci-service-account.yaml file into vilbert project.

**Important**: You may also have to increase the pipeline timeout value (default value is 1h). If your job may take more than 1 hour, go in your [gitlab project page](https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/kilbert/-/settings/ci_cd):
1. Under Settings
2. Select CI/CD
3. Expand "General pipelines"
4. Increase "Timeout" value. Example: 7d

**Important**: to enable/disable automatic deployment, simply rename the .gitlab-ci.yml with .disabled suffix or not.

#### Outputs (GitLab-CI)

You can save your job outputs with every job pipeline. In order to achieve that, edit the **deployment/job-outputs-downloader.tpl** template and make sure you set the volumeMount subPath to refer to your job outputs folder.

You will then be able to download outputs for pipeline by selecting **Artifacts->Download download-outputs artifacts** from the download button to the [right side of the pipeline](https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/kilbert/pipelines).

## Monitor Job

You can monitor your job using different tools

1. [Dashboard](https://dashboard.k8s.collaborative.local/#!/job?namespace=cad-xray)
2. [kubectl commands](https://wiki-trt.thales-systems.ca/bin/view/Trt%20Quebec/Engineering/Infrastructure/VLANs/Collaborative_VLAN/Kubernetes/Usage/#HJob27slogs)
3. [#k8s-collaborative](https://thales-quebec.slack.com/messages/CLALVMM6U) Slack channel
3. [Grafana](https://grafana.k8s.collaborative.local/) Dashboards (guest/guest)

## Stop/delete job

Once job is completed (or failed to complete) and you don't to access its logs anymore, you have to manually delete it using the following command.

```
kubectl delete -f ./jobs/<JOB_NAME>.yml
```

## Outputs

This project is configured to redirect print output to //isilon.storage.vlan/HUMAN_AI_DIALOG_SHARED/vilbert/outputs/.

## Inputs

This project is configured to read inputs from //isilon.storage.vlan/HUMAN_AI_DIALOG_SHARED/vilbert/data2/


## Import / Organize Dataset(s)

### From kubernetes

You need to:
1. Create and launch an Ubuntu container that mounts your HUMAN_AI_DIALOG_SHARED NAS shared folder.
2. Copy/download/re-organize your datasets.
3. Stop the ubuntu container.

Reference: [Wiki](https://wiki-trt.thales-systems.ca/bin/view/Trt%20Quebec/Engineering/Infrastructure/VLANs/Collaborative_VLAN/Kubernetes/Usage/#HCopyDataset28s29toPV)


First, the container named *ubuntu-host* can be launch using:
```
kubectl create -f ./deployment/ubuntu-host.yaml -n <NAMESPACE>
```

*Important*: This will start a container that last 48 hours. Modify the ubuntu-host.yaml to increase the sleep time if needed.

Once properly started (verify Pod state in [dashboard](https://dashboard.k8s.collaborative.local/#!/pod?namespace=human-ai-dialog), you can then connect to its terminal:

```
kubectl exec -ti ubuntu-host -n human-ai-dialog -- bash

# Note: volume is mounted at the /nas-data folder
```

You can then copy/download re-organize the selected dataset(s) from the /nas-data folder.

Once everything is ok, you can delete the container:

```
kubectl delete -f ./deployment/ubuntu-host.yaml
```

### From your workstation

Mount the //isilon.storage.vlan/HUMAN_AI_DIALOG_SHARED SAMBDA folder on your host and simply copy or reorganize the files.


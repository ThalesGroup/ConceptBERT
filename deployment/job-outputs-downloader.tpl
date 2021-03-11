apiVersion: v1
kind: Pod
metadata:
  name: download-JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER
  namespace: NAMESPACE_PLACEHOLDER
  annotations:
    author-name: "AUTHOR_NAME_PLACEHOLDER"
    author-email: "AUTHOR_EMAIL_PLACEHOLDER"
spec:
  containers:
  - name: ubuntu
    image: ubuntu:18.04
    # Will stay up 48h
    command:
      - sleep
      - "172800"
    imagePullPolicy: IfNotPresent
    volumeMounts:
      - name: nas-data-volume
        mountPath: /outputs
        # NOTE: Make sure the subpath refer to your job output folder
        subPath: outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER
  volumes:
  - name: nas-data-volume
    flexVolume:
      driver: "fstab/cifs"
      fsType: "cifs"
      secretRef:
        name: "NAMESPACE_PLACEHOLDER-cifs-service-user-secret"
      options:
        networkPath: "//isilon.storage.vlan/NAS_SHARED_FOLDER_PLACEHOLDER/"
        mountOptions: "nosuid,nodev,vers=3.0,nosetuids,noperm,mfsymlinks"
  restartPolicy: Never

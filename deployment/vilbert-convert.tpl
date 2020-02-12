apiVersion: batch/v1
kind: Job
metadata:
  name: JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER
  namespace: NAMESPACE_PLACEHOLDER
  annotations:
    author-name: "AUTHOR_NAME_PLACEHOLDER"
    author-email: "AUTHOR_EMAIL_PLACEHOLDER"
spec:
  ttlSecondsAfterFinished: 172800
  backoffLimit: 1
  template:
    spec:
      imagePullSecrets:
      - name: NAMESPACE_PLACEHOLDER-collaborative-docker-registry-secret
      initContainers:
      - name: init-output-folder
        image: busybox
        command: ["mkdir","-p","/nas-data/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER"]
        volumeMounts:
        - name: nas-data-volume
          mountPath: /nas-data
      containers:
      - name: vilbert-pod
        image: "collaborative-docker-registry.collaborative.local:5100/IMAGE_NAME_PLACEHOLDER"
        command: ["/bin/sh","-c"]
        args: ["cd vilbert && python convert_trainval_lmdb.py --input-dir=/nas-data/vilbert/data2/ --output-dir=/nas-data/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER"]
        env: 
          - name: LC_ALL
            value: C.UTF-8
          - name: LANG
            value: C.UTF-8
        volumeMounts:
          - name: nas-data-volume
            mountPath: /nas-data
      restartPolicy: Never    
      volumes:
      # Mount SAMBA volume from common-nas-server.common.local
      - name: nas-data-volume
        flexVolume:
          driver: "fstab/cifs"
          fsType: "cifs"
          secretRef:
            name: "NAMESPACE_PLACEHOLDER-cifs-service-user-secret"
          options:
            networkPath: "//common-nas-server.common.local/NAS_SHARED_FOLDER_PLACEHOLDER"
            mountOptions: "dir_mode=0755,file_mode=0644,noperm,vers=3.0,iocharset=utf8"

apiVersion: batch/v1
kind: Job
metadata:
  name: JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER
  namespace: NAMESPACE_PLACEHOLDER
  annotations:
    job-type: gpu
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
        command: ["mkdir","-p","/nas-data/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER"]
        volumeMounts:
        - name: nas-data-volume
          mountPath: /nas-data
      containers:
      - name: vilbert-pod
        image: "collaborative-docker-registry.collaborative.local:5100/IMAGE_NAME_PLACEHOLDER"
        resources:
          limits:
            nvidia.com/gpu: 1
        command: ["/bin/sh","-c"]
        args: ["python3 PythonEvaluationTools/vqaEval_okvqa.py --json_dir /nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/ --output_dir /nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/"]
        volumeMounts:
          - name: nas-data-volume
            mountPath: /nas-data
          - name: dshm
            mountPath: /dev/shm
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3"
      restartPolicy: Never
      volumes:
      # DSHM to raise Shared memory
      - name: dshm
        emptyDir:
          medium: Memory
      # Mount SAMBA volume from isilon.storage.vlan
      - name: nas-data-volume
        flexVolume:
          driver: "fstab/cifs"
          fsType: "cifs"
          secretRef:
            name: "NAMESPACE_PLACEHOLDER-cifs-service-user-secret"
          options:
            networkPath: "//isilon.storage.vlan/NAS_SHARED_FOLDER_PLACEHOLDER"
            mountOptions: "dir_mode=0755,file_mode=0644,noperm,vers=3.0,iocharset=utf8"
      affinity:
        # Schedule this job on a GPU worker node
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: worker-type
                operator: In
                values:
                - gpu
      tolerations:
      # Allow this job to be executed on a dedicated GPU worker node
      - key: "dedicated-processing"
        operator: Equal
        value: "gpu"

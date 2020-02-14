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
        command: ["mkdir","-p","/nas-data/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER"]
        volumeMounts:
        - name: nas-data-volume
          mountPath: /nas-data
      containers:
      - name: kilbert-pod
        image: "collaborative-docker-registry.collaborative.local:5100/IMAGE_NAME_PLACEHOLDER"
        resources:
          limits:
            nvidia.com/gpu: 8
        command: ["/bin/sh","-c"]
        args: ["cd kilbert && python3 -u train_concap.py --from_pretrained bert-base-uncased --bert_model bert-base-uncased --config_file config/bert_base_6layer_6conect.json --learning_rate 1e-4 --train_batch_size 512 --save_name kilbert_pretrained"]
        volumeMounts:
          - name: nas-data-volume
            mountPath: /nas-data
          - name: dshm
            mountPath: /dev/shm
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0,1,2,3,4,5,6,7"
      restartPolicy: Never    
      volumes:
      # DSHM to raise Shared memory
      - name: dshm
        emptyDir: 
          medium: Memory 
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

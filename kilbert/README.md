# ConceptBert

This repository is the implementation of ConceptBert: Concept-Aware Representation for Visual QuestionAnswering

For an overview of the pipleline, please refere [here](https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/kilbert/blob/master/kilbert/misc/pipeline.png)

This repository is based on and inspired by [Facebook research](https://github.com/facebookresearch/vilbert-multi-task). We sincerely thank for their sharing of the codes.

## Data

Check `README.md` under `data` for more details.  Check  `vlbert_tasks.yml` for more details. 


## Pre-trained model for Evaluation

| Model | Objective | Link |
|:-------:|:------:|:------:|
|ViLBERT 2-Layer| Conceptual Caption |[Google Drive]()|
|ViLBERT 4-Layer| Conceptual Caption |[Google Drive]()|
|ViLBERT 6-Layer| Conceptual Caption |[Google Drive](https://drive.google.com/drive/folders/1Re0L75uazH3Qrep_aRgtaVelDEz4HV9c?usp=sharing)|
|ViLBERT 8-Layer| Conceptual Caption |[Google Drive]()|
|ViLBERT 6-Layer| VQA |[Google Drive](https://drive.google.com/drive/folders/1nrcVww0u_vozcFRQVr58-YH5LOU1ZiWT?usp=sharing)|
|ViLBERT 6-Layer| VCR |[Google Drive](https://drive.google.com/drive/folders/1QJuMzBarTKU_hAWDSZm60rWiDnbAVEVZ?usp=sharing)|
|ViLBERT 6-Layer| RefCOCO+ |[Google Drive](https://drive.google.com/drive/folders/1GWY2fEbZCYHkcnxd0oysU0olfPdzcD3l?usp=sharing)|
|ViLBERT 6-Layer| Image Retrieval |[Google Drive](https://drive.google.com/drive/folders/18zUTF3ZyOEuOT1z1aykwtIkBUhfROmJo?usp=sharing)|

## Training and Validation




### VQA

1: Download the pretrained model with objective `VQA` and put it under `save`

2: To test on held out validation split, use the following command: 

```
python eval_tasks.py --bert_model bert-base-uncased --from_pretrained save/VQA_bert_base_6layer_6conect-pretrained/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --task 0 --split minval
```




### VQA 

To fintune a 6-layer ViLBERT model for VQA with 8 GPU. `--tasks 0` means VQA tasks. Check `vlbert_tasks.yml` for more settings for VQA tasks.  

```bash
python -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train_tasks.py --bert_model bert-base-uncased --from_pretrained save/bert_base_6_layer_6_connect_freeze_0/pytorch_model_8.bin  --config_file config/bert_base_6layer_6conect.json  --learning_rate 4e-5 --num_workers 16 --tasks 0 --save_name pretrained
```


## Evaluation

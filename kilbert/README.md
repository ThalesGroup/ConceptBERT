# ConceptBert

This repository is the implementation of ConceptBert: Concept-Aware Representation for Visual QuestionAnswering

For an overview of the pipleline, please refere [here](https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/kilbert/blob/master/kilbert/misc/pipeline.png)

This repository is based on and inspired by [Facebook research](https://github.com/facebookresearch/vilbert-multi-task). We sincerely thank for their sharing of the codes.

## Data

Our implementation uses the pretrained features from bottom-up-attention, 100 fixed features per image and the GloVe vectors. The data has been saved in NAS folder: human-ai-dialog/vilbert/data2. The data folder and pretrained_models folder are organized as shown below:

├── data2
│   ├── coco (visual features)
│   ├── conceptnet (conceptnet facts)
│   ├── conceptual_captions (captions for each image, extracted from [here](https://github.com/google-research-datasets/conceptual-captions))
│   ├── kilbert_base_model (pre-trained weights for initial kilbert model)
│   ├── OK-VQA (OK-VQA dataset)
│   ├── save_final (final saved models and outputs)
│   ├── tensorboards (location to save tensorboard files)
│   ├── VQA (VQA dataset)
│   ├── VQA_bert_base_6layer_6conect-pretrained (pre-trained weights for initial vilbert model trained on vqa)



The model checkpoints will be saved in NAS folder: human-ai-dialog/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER/



## Training and Validation

1: First we use VQA dataset to train a baseline model. Use the following job template: vilbert-job-train-model3_vqa_MZ.tpl  

2: Then we use OK-VQA dataset and the trained model from step 1 to train a model. Use the following job template: vilbert-job-train-model3_okvqa_MZ.tpl

3: To validate on held out validation split, we use the model trained in step 2 using following job template: vilbert-job-eval-model3_okvqa_MZ.tpl

Note: In the job templates, `--tasks 0` means VQA dataset and `--tasks 42` means OK-VQA dataset.

Note: The validation step 3 generates a json file ("val_result.json") that will be used in the evaluation.


## Evaluation

cd PythonEvaluationTools

python  vqaEval_okvqa.py #edit the path in the script to "val_result.json"

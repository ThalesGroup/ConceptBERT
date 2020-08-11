# ConceptBert

This repository is the implementation of ConceptBert: Concept-Aware Representation for Visual QuestionAnswering

For an overview of the pipleline, please refere [here](https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/kilbert/blob/master/kilbert/misc/pipeline.png)

This repository is based on and inspired by [Facebook research](https://github.com/facebookresearch/vilbert-multi-task). We sincerely thank for their sharing of the codes.

## Data

Our implementation uses the pretrained features from bottom-up-attention, 100 fixed features per image and the GloVe vectors. The data has been saved in NAS folder: human-ai-dialog/vilbert/data2. The data folder and pretrained_models folder are organized as shown below:

```bash
├── data2
│   ├── coco (visual features)
│   ├── conceptnet (conceptnet facts)
│   ├── conceptual_captions (captions for each image, extracted from (https://github.com/google-research-datasets/conceptual-captions))
│   ├── kilbert_base_model (pre-trained weights for initial kilbert model)
│   ├── OK-VQA (OK-VQA dataset)
│   ├── save_final (final saved models and outputs)
│   ├── tensorboards (location to save tensorboard files)
│   ├── VQA (VQA dataset)
│   ├── VQA_bert_base_6layer_6conect-pretrained (pre-trained weights for initial vilbert model trained on vqa)
```


The model checkpoints will be saved in NAS folder: human-ai-dialog/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER/



## Training and Validation

### Train with VQA
1: First we use VQA dataset to train a baseline model. Use the following job template: vilbert-job-train-model3_vqa_MZ.tpl  
```
./deploy.sh deployment/vilbert-job-train-model3_vqa_MZ.tpl  
```
In the template the command is :
```
args: ["cd kilbert && python3 -u train_tasks.py --model_version 3 --bert_model=bert-base-uncased --from_pretrained_kilbert None --from_pretrained=/nas-data/vilbert/data2/kilbert_base_model/pytorch_model_9.bin --config_file config/bert_base_6layer_6conect.json --output_dir=/nas-data/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER --num_workers 16 --tasks 0"]
```
| Parameter | Description |
|-----------|-------------|
| model_version |  Which version of the model you want to use |
| bert_model | Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese. |
| from_pretrained_kilbert | folder of the previous trained model. In this case, it's the first train, so the value is`None`  |
| from_pretrained  | pre-trained Bert model (VQA) |
| config_file  | 3 config files are available in `kilbert/config/` |
| output_dir  | folder where the results are saved  |
| task  |  task = 0, we use VQA dataset |


### Train with OK-VQA (fine-tuning)
2: Then we use OK-VQA dataset and the trained model from step 1 to train a model. Use the following job template: vilbert-job-train-model3_okvqa_MZ.tpl
```
./deploy.sh deployment/vilbert-job-train-model3_okvqa_MZ.tpl  
```
In the template the command is :
```
args: ["cd kilbert && python3 -u train_tasks.py --model_version 3 --bert_model=bert-base-uncased --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin --from_pretrained_kilbert /nas-data/vilbert/outputs/vilbert-job-0.1.dev752-g896be56.d20200807135547/VQA_bert_base_6layer_6conect/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --output_dir=/nas-data/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER --num_workers 16 --tasks 42"]
```
The parameters are the same as above, but theses values change:

| Parameter | Description |
|-----------|-------------|
| from_pretrained_kilbert | The path of the model trained previously (step1 VQA). Corresponding of the last `pytorch_model_**.bin` file generated |
| from_pretrained  | pre-trained Bert model (OK-VQA) |
| task  |  task = 42 OKVQA dataset is used |

### Validation with OK-VQA
3: To validate on held out validation split, we use the model trained in step 2 using following job template: vilbert-job-eval-model3_okvqa_MZ.tpl
```
./deploy.sh deployment/vilbert-job-eval-model3_okvqa_MZ.tpl  
```
In the template the command is :
```
args: ["cd kilbert && python3 -u eval_tasks.py --model_version 3 --bert_model=bert-base-uncased --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin  --from_pretrained_kilbert=/nas-data/vilbert/outputs/vilbert-job-0.1.dev752-g896be56.d20200810140504/OK-VQA_bert_base_6layer_6conect/pytorch_model_99.bin --config_file config/bert_base_6layer_6conect.json --output_dir=/nas-data/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER --num_workers 16 --tasks 42 --split val"]
```
The parameters are the same as above, but theses values change:

| Parameter | Description |
|-----------|-------------|
| from_pretrained_kilbert | The path of the model trained previously (step2 OKVQA). Corresponding of the last `pytorch_model_**.bin` file generated |
| from_pretrained  | same pre-trained Bert model (OK-VQA) as step2 |
| task  |  task = 42 OKVQA is used |


Note: In the job templates, `--tasks 0` means VQA dataset and `--tasks 42` means OK-VQA dataset.

Note: The validation step 3 generates a json file ("val_result.json") that will be used in the evaluation.


## Evaluation

Use the `vilbert-job-evaluation.tpl` to run the evaluation :
```
args: ["ls && python /app/kilbert/PythonEvaluationTools/vqaEval_okvqa.py --json_dir /nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/ --output_dir /nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/"]
```
* `json_dir`: path where is located the `val_result.json`
* `output_path`: folder where the accuracy will be saved
* `/nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/`: is the last best model. *You must change this by the path of model you want to evaluate*.

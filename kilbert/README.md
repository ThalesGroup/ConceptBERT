# ConceptBert

This repository is the implementation of ConceptBert: Concept-Aware Representation for Visual QuestionAnswering

For an overview of the pipleline, please refere [here](https://sc01-trt.thales-systems.ca/gitlab/human-ai-dialog/kilbert/blob/master/kilbert/misc/pipeline.png)

This repository is based on and inspired by [Facebook research](https://github.com/facebookresearch/vilbert-multi-task). We sincerely thank for their sharing of the codes.

# :electric_plug: Data

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



# :rocket: Training and Validation
Note: models and json used in the following examples are the current best results

## 1. Train with VQA
First we use VQA dataset to train a baseline model. Use the following job template: `vilbert-job-train-model3_vqa_MZ.tpl`
### Start the training with:
```console
./deploy.sh deployment/vilbert-job-train-model3_vqa_MZ.tpl
```


### Command description
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


## 2. Train with OK-VQA (fine-tuning)
Then we use OK-VQA dataset and the trained model from step 1 to train a model. Use the following job template: vilbert-job-train-model3_okvqa_MZ.tpl
### Start the training with:
```
./deploy.sh deployment/vilbert-job-train-model3_okvqa_MZ.tpl  
```
    
### Command description   
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

## 3. Validation with OK-VQA
To validate on held out validation split, we use the model trained in step 2 using following job template: vilbert-job-eval-model3_okvqa_MZ.tpl
### Start the training with:
```
./deploy.sh deployment/vilbert-job-eval-model3_okvqa_MZ.tpl  
```
    
Two files will be generated:
* `Val_other` give 8 top answers for each questions
* `val_result` used in the evaluation

### Command description
```
args: ["cd kilbert && python3 -u eval_tasks.py --model_version 3 --bert_model=bert-base-uncased --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin  --from_pretrained_kilbert=/nas-data/vilbert/outputs/vilbert-job-0.1.dev752-g896be56.d20200810140504/OK-VQA_bert_base_6layer_6conect/pytorch_model_99.bin --config_file config/bert_base_6layer_6conect.json --output_dir=/nas-data/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER --num_workers 16 --tasks 42 --split val"]
```
The parameters are the same as above, but theses values change:

| Parameter | Description |
|-----------|-------------|
| from_pretrained_kilbert | The path of the model trained previously (step2 OKVQA). Corresponding of the last `pytorch_model_**.bin` file generated |
| from_pretrained  | same pre-trained Bert model (OK-VQA) as step2 |
| task  |  task = 42 OKVQA is used |



# :rocket: Evaluation

Use the `vilbert-job-evaluation.tpl` to run the evaluation :
## Start the training with:
```console
./deploy.sh deployment/vilbert-job-evaluation.tpl
```

## Command description
```
args: ["ls && python /app/kilbert/PythonEvaluationTools/vqaEval_okvqa.py --json_dir /nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/ --output_dir /nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/"]
```
* `json_dir`: path where is located the `val_result.json`
* `output_path`: folder where the accuracy will be saved
* `/nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/`: is the final json. *You must change this by the path of the json you want to evaluate*.


# :bug: Known issues
Sometimes the `deploy.sh` command don't display the correct name of the job and the link given doesn't work (but the job is still active on kubernetes).
You can empty the `JOB_NAME` variable to fix the problem.

Ex.
```console
export JOB_NAME= && ./deploy.sh deployment/vilbert-job-train-model3_vqa_MZ.tpl
```




# :bulb: Compare the results
## Step 1: Training with VQA
* 20 checkpoints must have been created (`last file name must be pytorch_model_19.bin`)

## Step 2: Training with OK-VQA
* 100 checkpoints must have been created (`last file name must be pytorch_model_99.bin`)

## Step 3: Validation with OK-VQA
* The validation generates two json file. `val_result.json` will be used in the evaluation.
* Open the logs in the kubernetes pod to check the result of the `eval_score`:

```bash
08/12/2020 13:09:46 - INFO - utils -   Validation [OK-VQA]: loss 3.681 score 33.040
```

If you want to optimize your model the `loss` and `score` must be at least be the same as above.

## Evaluation
Compare the result of the `accuracy.json` generated with the json of the last best model (`/nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/accuracy.json`). \
The results must be at least as good as the previous ones.



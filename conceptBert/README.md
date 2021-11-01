# ConceptBert

This repository is the implementation of ConceptBert: Concept-Aware Representation for Visual Question Answering.

Original paper:
*François Gardères, Maryam Ziaeefard, Baptiste Abeloos, Freddy Lécué: ConceptBert: Concept-Aware Representation for
Visual Question Answering. EMNLP (Findings) 2020: 489-498
https://aclanthology.org/2020.findings-emnlp.44.pdf*

For an overview of the pipleline, please refere to the following picture:

![Pipeline](/conceptBert/misc/pipeline.png)

## License

This work is dual-licensed under the `Thales Digital Solutions Canada` license and `MIT License`.

* **The main license is the `Thales Digital Solutions Canada` one**. You can find the [license](LICENSE) file here.
* This repository is based on and inspired
  by [Facebook research (vilbert-multi-task)](https://github.com/facebookresearch/vilbert-multi-task). We sincerely
  thank for their sharing of the codes.
  **The code related to `vilbert-multi-task` is licensed by the MIT License, please for more information
  refer [to the file](LICENSE-VILBERT-MULTI-TASK).**

### Pre-requisite

* python 3.6.12
* docker environment

### Recommended

If you want to be able to develop on docker, we recommend you to use VSCODE with the container plugin.

* [VSCode](https://code.visualstudio.com/) work
  with [containers](https://code.visualstudio.com/docs/containers/overview)

### Disclaimer

Currently, the project requires a lot of resources to be able to run correctly.

It is necessary to count at least 6 days of training for the first training with a `GTX 1080 Ti`(11Go RAM), and 17hours
in an Kubernetes environment with 7GPU (7 `Titan-v`(32Go)). All the pipelines were tested on GPU server with
four `GeForce RTX 2080 Ti` (12Go)


# :electric_plug: Data

**All information regarding the datasets or models used is specified in the [original paper](https://aclanthology.org/2020.findings-emnlp.44.pdf).**

Our implementation uses the pretrained features from bottom-up-attention, 100 fixed features per image and the GloVe
vectors. The data might be saved in a folder along with pretrained_models and organized as shown below:

```text
vilbert
├── data2
│   ├── coco (visual features)
│   ├── conceptnet (conceptnet facts)
│   ├── conceptual_captions (captions for each image, extracted from (https://github.com/google-research-datasets/conceptual-captions))
│   ├── kilbert_base_model (pre-trained weights for initial conceptBert model)
│   ├── OK-VQA (OK-VQA dataset)
│   ├── save_final (final saved models and outputs)
│   ├── tensorboards (location to save tensorboard files)
│   ├── VQA (VQA dataset)
│   ├── VQA_bert_base_6layer_6conect-pretrained (pre-trained weights for initial vilbert model trained on vqa)
```

The model checkpoints will be saved in the output : ./outputs/

# :whale2: Docker installation (recommended)

You can choose to run ConceptBert with Docker or from your environment

## Build

```bash
  docker build -t conceptbert .
```

## Start the container

```bash
  docker run -it -v /path/to/you/nas/:/nas-data/ conceptbert:latest bash
```

### Additional parameters

```bash
  docker run -it -v --shm-size=10g -e CUDA_VISIBLE_DEVICES=0,1,2,3 /path/to/you/nas/:/nas-data/ conceptbert:latest bash
```

* `--shm-size` is used to prevent Shared Memory error. Here the value is
  10Go ([refer docker documentation](https://docs.docker.com/engine/reference/run/))
* `-e CUDA_VISIBLE_DEVICES` is used to use specific GPU available. Here we want to use 4 GPU.

When the container is up, go to the section [1. Train with VQA](#1.-train-with-vqa)

# Other installation

You can use the `requirements.txt` file to install the dependencies of the project.

Pre-requisite:

* Compile the tools `cd conceptBert/tools/refer && make`
* python 3.6.x

**If you have difficulties to create your environment, look at the contents of the Dockerfile for the necessary
dependencies that you might miss.**

# :rocket: Training and Validation
Note: models and json used in the following examples are the current best results

## 1. Train with VQA
First we use VQA dataset to train a baseline model. Use the following command:

```bash
  python3 -u train_tasks.py --model_version 3 --bert_model=bert-base-uncased --from_pretrained_conceptBert None \
      --from_pretrained=/nas-data/vilbert/data2/kilbert_base_model/pytorch_model_9.bin \
      --config_file config/bert_base_6layer_6conect.json \
      --output_dir=/nas-data/outputs/train1_vqa_trained_model/ \
      --summary_writer /nas-data/tensorboards/ \
      --num_workers 16 \
      --tasks 0
```

### Kubernetes Command
```console
./deploy.sh deployment/train-model3-vqa-mz.tpl
```

#### Template command
```
args: ["cd conceptBert && python3 -u train_tasks.py --model_version 3 --bert_model=bert-base-uncased --from_pretrained_conceptBert None --from_pretrained=/nas-data/vilbert/data2/kilbert_base_model/pytorch_model_9.bin --config_file config/bert_base_6layer_6conect.json --output_dir=/nas-data/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER --num_workers 16 --tasks 0"]
```


### Command description

| Parameter | Description |
|-----------|-------------|
| u | -u is used to force stdin, stdout and stderr to be totally unbuffered, which otherwise is line buffered on the terminal |
| model_version |  Which version of the model you want to use |
| bert_model | Bert pre-trained model selected in the list: bert-base-uncased, bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese. |
| from_pretrained_conceptBert | folder of the previous trained model. In this case, it's the first train, so the value is`None`  |
| from_pretrained  | pre-trained Bert model (VQA) |
| config_file  | 3 config files are available in `conceptBert/config/` |
| output_dir  | folder where the results are saved  |
| summary_writer  |  folder used to save tensorboard items. A sub-folder will be created with the date of the day |
| num_worker | Tells the data loader instance how many sub-processes to use for data loading. **Use your own value in
regard of your environment** |
| task  |  task = 0, we use VQA dataset |

## 2. Train with OK-VQA (fine-tuning)
Then we use OK-VQA dataset and the trained model from step 1 to train a model. Use the following command:

```bash
  python3 -u train_tasks.py --model_version 3 --bert_model=bert-base-uncased \
      --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin \
      --from_pretrained_conceptBert /nas-data/outputs/train1_vqa_trained_model/VQA_bert_base_6layer_6conect/pytorch_model_19.bin \
      --config_file config/bert_base_6layer_6conect.json \
      --output_dir=/nas-data/outputs/train2_okvqa_trained_model/ \
      --summary_writer /outputs/tensorboards/  \
      --num_workers 16 \
      --tasks 42
```

### Kubernetes Command

```
./deploy.sh deployment/train-model3-okvqa-mz.tpl  
```
    
#### Template command   
In the template the command is :
```
args: ["cd conceptBert && python3 -u train_tasks.py --model_version 3 --bert_model=bert-base-uncased --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin --from_pretrained_conceptBert /nas-data/vilbert/outputs/vilbert-job-0.1.dev752-g896be56.d20200807135547/VQA_bert_base_6layer_6conect/pytorch_model_19.bin --config_file config/bert_base_6layer_6conect.json --output_dir=/nas-data/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER --num_workers 16 --tasks 42"]
```
### Command description

The parameters are the same as above, but these values change:

| Parameter | Description |
|-----------|-------------|
| from_pretrained_conceptBert | The path of the model trained previously (step1 VQA). Corresponding of the last `pytorch_model_**.bin` file generated |
| from_pretrained  | pre-trained Bert model (OK-VQA) |
| task  |  task = 42 OKVQA dataset is used |

## 3. Validation with OK-VQA

To validate on held out validation split, we use the model trained in step 2 using following command:
VQA_bert_base_6layer_6conect

```bash
  python3 -u eval_tasks.py --model_version 3 --bert_model=bert-base-uncased \
      --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin  \
      --from_pretrained_conceptBert=/nas-data/outputs/train2_okvqa_trained_model/OK-VQA_bert_base_6layer_6conect/pytorch_model_99.bin \
      --config_file config/bert_base_6layer_6conect.json \
      --output_dir=/nas-data/outputs/validation_okvqa_trained_model/ \
      --num_workers 16 \
      --tasks 42 \
      --split val
```

### Kubernetes Command

```
./deploy.sh deployment/eval-model3-okvqa-mz.tpl
```

#### Template command
```
args: ["cd conceptBert && python3 -u eval_tasks.py --model_version 3 --bert_model=bert-base-uncased --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin  --from_pretrained_conceptBert=/nas-data/vilbert/outputs/vilbert-job-0.1.dev752-g896be56.d20200810140504/OK-VQA_bert_base_6layer_6conect/pytorch_model_99.bin --config_file config/bert_base_6layer_6conect.json --output_dir=/nas-data/vilbert/outputs/JOB_NAME_PLACEHOLDER-JOB_ID_PLACEHOLDER --num_workers 16 --tasks 42 --split val"]
```

Two files will be generated:

* `Val_other` give 8 top answers for each questions
* `val_result` used in the evaluation

### Command description

The parameters are the same as above, but theses values change:

| Parameter | Description |
|-----------|-------------|
| from_pretrained_conceptBert | The path of the model trained previously (step2 OKVQA). Corresponding of the last `pytorch_model_**.bin` file generated |
| from_pretrained  | same pre-trained Bert model (OK-VQA) as step2 |
| task  |  task = 42 OKVQA is used |



# :rocket: Evaluation

Run the evaluation :

## Start the training with:

```bash
  python3 PythonEvaluationTools/vqaEval_okvqa.py \
      --json_dir /nas-data/outputs/validation_okvqa_trained_model/ \
      --output_dir /nas-data/outputs/validation_okvqa_trained_model/
```

### Kubernetes Command
```console
./deploy.sh deployment/evaluation.tpl
```

#### Template command
```
args: ["python /app/conceptBert/PythonEvaluationTools/vqaEval_okvqa.py --json_dir /nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/ --output_dir /nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/"]
```

## Command description

* `json_dir`: path where is located the `val_result.json`
* `output_path`: folder where the accuracy will be saved
* `/nas-data/outputs/validation_okvqa_trained_model/`: is the final json. *You must change this by the path of
  the json you want to evaluate*.


# :bug: Known issues

* If `python-prctl` return `"python-prctl" Command "python setup.py egg_info" failed with error` error, use this
  command :

```bash
  sudo apt-get install libcap-dev python3-dev
```


# :bulb: Compare the results
## Step 1: Training with VQA
* 20 checkpoints must have been created (`last file name must be pytorch_model_19.bin`)

## Step 2: Training with OK-VQA
* 100 checkpoints must have been created (`last file name must be pytorch_model_99.bin`)

## Step 3: Validation with OK-VQA
* The validation generates two json file. `val_result.json` will be used in the evaluation.
* Open the logs in the output folder (`nas-data-`) to check the result of the `eval_score`:

```bash
08/12/2020 13:09:46 - INFO - utils -   Validation [OK-VQA]: loss 3.681 score 33.040
```

If you want to optimize your model the `loss` and `score` must be at least be the same as above.

## Evaluation

Compare your results in the `accuracy.json` file (results must be at least as good as the following ones).
```json
{
  "overall": 33.04,
  "perQuestionType": {
    "one": 30.82,
    "eight": 33.6,
    "other": 32.57,
    "seven": 30.61,
    "four": 36.79,
    "five": 33.66,
    "three": 31.73,
    "nine": 31.43,
    "ten": 45.58,
    "two": 30.23,
    "six": 30.07
  },
  "perAnswerType": {
    "other": 33.04
  }
}
```


# VQA Training
* [Documentation here](misc/training_vqa.md)
# OK-VQA Training
* [Documentation here](misc/training_okvqa.md)


# Troubleshooting

## CUDA out of memory
Try the following recommendation to resolve the problem:

* Change the value of `num_workers` in your training command (ex. `--num_workers 1`)
* Try one of the [improvements](#improvements) proposition bellow
* Reduce parameters in `vlbert_tasks.yml`:
    * max_seq_length
    * batch_size
    * eval_batch_size

Example:

```bash
  max_seq_length: 4 # DGX value : 16
  batch_size: 256 # DGX value : 1024
  eval_batch_size: 256 # DGX value : 1024
```

# Improvements

There are several areas for improvement:

* Search and replace the `to.device()` parameter in the code to be executed in the better position
* Load a part of the dataset (create a method to load a batch of the dataset). Dataset management is in `vqa_dataset.py`
  , method `_load_dataset`, variables `questions = questions_train + questions_val[:-3000]`
  and `answers = answers_train + answers_val[:-3000]`
* Train your own BERT (or find a lighter Bert)
* Initialise Bert once and load it after

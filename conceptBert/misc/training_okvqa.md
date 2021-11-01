# OK-VQA
* Training 1 : 
* Validation/Evaluation from Training:


## Training
In comparaison of the best results procedure (with 2 training, 1 validation, 1 evaluation), I had run just one training on the VQA dataset.


Update the template file `train-model3-vqa-mz.tpl`:
```console
python3 -u train_tasks.py --model_version 3 --bert_model=bert-base-uncased \
  --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin \
  --from_pretrained_conceptBert /nas-data/outputs/train1_vqa_trained_model/VQA_bert_base_6layer_6conect/pytorch_model_19.bin \
  --config_file config/bert_base_6layer_6conect.json \
  --output_dir=/nas-data/outputs/train2_okvqa_trained_model/ \
  --summary_writer /outputs/tensorboards/  \
  --num_workers 16 \
  --tasks 42
```
* tasks 42: run on the VQA dataset


## Validation
Validation is based on the first training results (using "pytorch_model_19.bin").


Update the template file `eval-model3_okvqa_MZ.tpl`:
```
python3 -u eval_tasks.py --model_version 3 --bert_model=bert-base-uncased \
  --from_pretrained=/nas-data/vilbert/data2/save_final/VQA_bert_base_6layer_6conect-beta_vilbert_vqa/pytorch_model_11.bin  \
  --from_pretrained_conceptBert=/nas-data/outputs/train2_okvqa_trained_model/OK-VQA_bert_base_6layer_6conect/pytorch_model_99.bin \
  --config_file config/bert_base_6layer_6conect.json \
  --output_dir=/nas-data/outputs/validation_okvqa_trained_model/ \
  --num_workers 16 \
  --tasks 42 \
  --split val
```
* from_pretrained_conceptBert: change the value with the result of the previous training path
* tasks 42: run on the VQA dataset


### Evaluation

Update the template file `evaluation.tpl`:
```console
python3 PythonEvaluationTools/vqaEval_okvqa.py \
  --json_dir /nas-data/outputs/validation_okvqa_trained_model/ \
  --output_dir /nas-data/outputs/validation_okvqa_trained_model/
```
* json_dir: change the value with the result of the validation path
* output_dir: change the value with the result of the validation path (same path as json_dir)


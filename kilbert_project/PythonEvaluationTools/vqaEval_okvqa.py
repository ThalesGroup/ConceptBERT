# coding: utf-8
import argparse
import json

from PythonEvaluationTools.vqaEval import VQAEval
from PythonEvaluationTools.vqa_helper import VQA

# set up file names and paths
taskType = 'OpenEnded'
dataType = 'mscoco'
dataSubType = 'val2014'
data_dir = '/nas-data/vilbert/data2/OK-VQA/'
annFile = "%s/%s_%s_annotations.json" % (data_dir, dataType, dataSubType)
quesFile = "%s/%s_%s_%s_questions.json" % (data_dir, taskType, dataType, dataSubType)

# Use these parameters to run the evaluation on VQA dataset only
#data_dir = '/nas-data/vilbert/data2/VQA' #VQA version
#annFile = "%s/v2_%s_%s_annotations.json" % (data_dir, dataType, dataSubType)
#quesFile = "%s/v2_%s_%s_%s_questions.json" % (data_dir, taskType, dataType, dataSubType)

#json_dir = '/nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/'
#output_dir = '/nas-data/vilbert/outputs/vilbert-job-0.1.dev460-g22e5d72.d20200810225318/'
res_name = 'val_result'
fileTypes = ['accuracy', 'evalQA', 'evalQuesType', 'evalAnsType']
# An example result json file has been provided in './Results' folder.  


parser = argparse.ArgumentParser()
parser.add_argument('--json_dir', type=str)
parser.add_argument('--output_dir', type=str)

args = parser.parse_args()
json_dir = args.json_dir
output_dir = args.output_dir
resFile = '%s/%s.json' % (json_dir, res_name)

[accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/%s.json' % (output_dir, fileType) for fileType in
                                                                 fileTypes]

# create vqa object and vqaRes object
vqa = VQA(annFile, quesFile)
vqaRes = vqa.loadRes(resFile, quesFile)

# create vqaEval object by taking vqa and vqaRes
vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

# evaluate results
"""
If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
By default it uses all the question ids in annotation file
"""
vqaEval.evaluate()

# print accuracies
print("\n")
print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
print("Per Question Type Accuracy is the following:")
for quesType in vqaEval.accuracy['perQuestionType']:
    print("%s : %.02f" % (quesType, vqaEval.accuracy['perQuestionType'][quesType]))
print("\n")
print("Per Answer Type Accuracy is the following:")
for ansType in vqaEval.accuracy['perAnswerType']:
    print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
print("\n")

# save evaluation results to ./Results folder
json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
json.dump(vqaEval.evalQA, open(evalQAFile, 'w'))
json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
json.dump(vqaEval.evalAnsType, open(evalAnsTypeFile, 'w'))

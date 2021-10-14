### LIBRARIES ###
# Global libraries
import os
import json

### FUNCTION DEFINITION ###

def get_txt_questions(split):
    """
        Returns the text of the questions
    """  
    import torch
    from pytorch_pretrained_bert.tokenization import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    
    question_path = "/nas-data/vilbert/data2/OK-VQA/OpenEnded_mscoco_" + str(split) + "2014_questions.json"
    questions = sorted(json.load(open(question_path))["questions"], key=lambda x: x["question_id"])
    questions_ordered = {}
    for question in questions:
        questions_ordered[question["question_id"]] = question["question"]
        
    dict_tokens = {}
    for _, question in questions_ordered.items():
        tokens = tokenizer.tokenize(question)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        for token in tokens:
            token_emb = tokenizer.vocab.get(token, tokenizer.vocab["[UNK]"])
            
            if token_emb not in dict_tokens:
                dict_tokens[token_emb] = token
     
    return dict_tokens
        

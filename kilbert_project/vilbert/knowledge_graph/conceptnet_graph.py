### LIBRARIES ###
# Global libraries
import os
import _pickle
from termcolor import colored
from tqdm import tqdm

import numpy as np
import torch

# Custom libraries
from kilbert_project.vilbert import get_txt_questions

### UTILS FUNCTIONS ###
def initiate_embedding():
    """
        Saves the embedding dictionary in a file
    """
    dict_embedding = {}
    with open("/nas-data/vilbert/data2/conceptnet/raw/embeddings.txt", "r") as raw_file:
        for entry in tqdm(raw_file, desc="Saving the node embeddings"):
            entry.strip()
            if entry:
                embedding_split = entry.replace(" \n", "").split(" ")
                word = embedding_split[0]
                embedding = np.asarray(embedding_split[1:])
                dict_embedding[word] = embedding
    # Save in JSON file
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/embeddings.pkl", "wb"
    ) as pkl_file:
        _pickle.dump(dict_embedding, pkl_file)


def load_embeddings():
    """
        Loads the embeddings from ConceptNet
    """
    if not os.path.exists(
        "/nas-data/vilbert/data2/conceptnet/processed/embeddings.pkl"
    ):
        initiate_embedding()

    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/embeddings.pkl", "rb"
    ) as pkl_file:
        dict_embedding = _pickle.load(pkl_file)

    return dict_embedding


### CLASS DEFINITION ###
class ConceptNet:
    """
        Class represents the ConceptNet graph
    """

    def __init__(self, split=None):
        """
        """
        # Loading the word embeddings
        self.dict_embedding = load_embeddings()
        self.dim_word = len(self.dict_embedding["the"])

        # Load the question texts
        if split != None:
            print(colored("Loading question texts...", "yellow"))
            self.token_dictionary = get_txt_questions(split)

    def test_has_embedding(self, node):
        """
            Tests if a given node has an embedding
        """
        try:
            node_embedding = self.get_node_embedding(node)
            if len(node_embedding) == 0:
                pass
            return node_embedding
        except:
            return False

    def get_node_embedding(self, word):
        """
            Given a node (word), return its embedding
        """
        try:
            return self.dict_embedding[word]
        except:
            pass

    def get_node_embedding_tensor(self, word):
        """
            Returns the node embedding (in a tensor) of the given node
        """
        try:
            return torch.from_numpy(self.dict_embedding[word])
        except Exception as e:
            return torch.zeros(self.dim_word).double()

    def get_kg_embedding_tokens(self, input_ids, dim1, dim2):
        """
            Given a list of tokens, returns the node embedding of each word
        """
        kg_embedding = []

        input_ids = input_ids.tolist()

        for question in input_ids:
            question_embedding = []
            for word_token in question:
                try:
                    word = self.token_dictionary[word_token]
                    word_kg_emb = self.get_node_embedding_tensor(word)
                except:
                    word_kg_emb = torch.zeros(self.dim_word).double()
                target_emb = torch.zeros((dim2))
                target_emb[: word_kg_emb.size(0)] = word_kg_emb
                question_embedding.append(word_kg_emb)

            question_kg_emb = torch.stack(question_embedding)
            target_kg = torch.zeros((dim1, dim2))
            target_kg[
                : question_kg_emb.size(0), : question_kg_emb.size(1)
            ] = question_kg_emb
            kg_embedding.append(target_kg)

        return torch.stack(kg_embedding)


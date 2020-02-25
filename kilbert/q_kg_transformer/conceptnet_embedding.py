### LIBRARIES ###
# Global libraries
from termcolor import colored

import torch
import torch.nn as nn

# Custom libraries
from q_kg_transformer.utils import load_embeddings, get_txt_questions

### CLASS DEFINITION ###
class KnowledgePooler(nn.Module):
    def __init__(self, final_dim):
        super(KnowledgePooler, self).__init__()
        self.dense = nn.Linear(200, final_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ConceptNetEmbedding:
    """

    """

    def __init__(self, config, split=None):
        # Loading the word embeddings
        self.dict_embedding = load_embeddings()
        # self.dim_word = len(self.dict_embedding["the"])

        # Load the question texts
        if split != None:
            print(colored("Loading question texts...", "yellow"))
            self.token_dictionary = get_txt_questions(split)

        self.model_version = config.model_version

        # Pooler to return a 1024-D embedding
        if self.model_version == 1:
            self.cn_pooler = KnowledgePooler(1024)
            self.dim_word = 1024
        elif self.model_version == 2:
            self.cn_pooler = KnowledgePooler(768)
            self.dim_word = 768

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
            Given a node (word), returns its embedding
        """
        try:
            return self.dict_embedding[word]
        except:
            pass

    def get_node_embedding_tensor(self, word):
        """
            Given a node (word), returns the node embedding in a tensor
        """
        if self.model_version == 1:
            try:
                conceptnet_emb = torch.from_numpy(self.dict_embedding[word])
                return self.cn_pooler(conceptnet_emb)
            except:
                return torch.zeros(1024).double()

        elif self.model_version == 2 or self.model_version == 3:
            try:
                conceptnet_emb = torch.from_numpy(self.dict_embedding[word])
                return self.cn_pooler(conceptnet_emb)
            except:
                return torch.zeros(768).double()

    def get_kg_embedding_tokens_from_bert(self, input_ids, dim1, dim2):
        """
            Given a list of tokens from the question (BERT), returns the node embedding of each word
        """
        kg_embedding = []

        input_ids = input_ids.tolist()

        for question in input_ids:
            question_embedding = []
            for word_token in question:
                try:
                    word = self.token_dictionary[word_token]
                    if word[:2] == "##":
                        word = word[2:]
                    word_kg_emb = self.get_node_embedding_tensor(word)
                except:
                    word_kg_emb = torch.zeros(self.dim_word).double()

                """
                target_emb = torch.zeros((dim2))
                target_emb[: word_kg_emb.size(0)] = word_kg_emb
                """
                question_embedding.append(word_kg_emb)

            question_kg_emb = torch.stack(question_embedding)
            """
            target_kg = torch.zeros((dim1, dim2))
            target_kg[
                : question_kg_emb.size(0), : question_kg_emb.size(1)
            ] = question_kg_emb
            """
            # kg_embedding.append(target_kg)
            kg_embedding.append(question_kg_emb)

        return torch.stack(kg_embedding)

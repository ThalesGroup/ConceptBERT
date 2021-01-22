### LIBRARIES ###
# Global libraries
import os
import json
from termcolor import colored
from tqdm import tqdm

import numpy as np

import torch

# Custom libraries
from .get_node_embedding import NumberbatchConverter, GloveConverter
from .extract_infos import extract_nodes, compute_node_dictionary, compute_neighborhoods
from kilbert_project.vilbert import get_txt_questions

### CLASS DEFINITION ###
class ConceptNet:
    """
        Class represents the ConceptNet graph

        Methods: 
            - clean_nodes(): removes nodes that don't have any embedding in the chosen embedding method
            - test_has_embedding(node): tests if a given node has an embedding; returns it if it has one
            - get_node_embedding(node): given a node (word), returns its embedding
            - compute_wl_embedding(n_iterations): computes the Weisfeiler-Lehman embedding of each node, with `n_iterations` updates
            - get_wl_embedding(word): returns the previously-computed Weisfeiler-Lehman embedding of the given node
    """

    def __init__(self, embedding_method, split=None):
        """
        """
        # Loading the nodes
        print(colored("Loading the list of nodes...", "yellow"))
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/en_nodes.json"
        ):
            extract_nodes()
        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/en_nodes.json", "r"
        ) as json_file:
            self.list_nodes = json.load(json_file)

        # Loading the node dictionary
        print(colored("Loading the dictionary of nodes...", "yellow"))
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/en_nodes_dictionary.json"
        ):
            compute_node_dictionary()
        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/en_nodes_dictionary.json",
            "r",
        ) as json_file:
            self.nodes_dictionary = json.load(json_file)

        # Loading the nodes' neighborhoods
        print(colored("Loading the node neighborhoods...", "yellow"))
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/en_dict_neighbors.json"
        ):
            compute_neighborhoods()
        with open(
           "/nas-data/vilbert/data2/conceptnet/processed/en_dict_neighbors.json",
            "r",
        ) as json_file:
            dict_neighbors = json.load(json_file)

        self.dict_neighbors = {}
        for node, list_neighbors in dict_neighbors.items():
            self.dict_neighbors[node] = []
            for neighbor in list_neighbors:
                self.dict_neighbors[node].append(
                    [self.list_nodes[neighbor[0]], neighbor[1]]
                )

        # Loading the embedding method
        print(colored("Loading the embedding method...", "yellow"))
        if embedding_method == "glove":
            self.embedding_method = GloveConverter()
        elif embedding_method == "numberbatch":
            self.embedding_method = NumberbatchConverter()

        self.dim_word = len(self.embedding_method.convert_word_to_embedding("the"))

        # Variables for statistics
        self.count = 0
        self.created = 0
        self.counter = 0
        self.ok_count = 0

        # Filter the nodes, so that every node has an embedding
        self.filtered_dict_neighbors = self.clean_nodes()

        
        # Load the question texts
        if split != None:
            print(colored("Loading question texts...", "yellow"))
            self.token_dictionary = get_txt_questions(split)
        
    def clean_nodes(self):
        """
            Removes nodes if they don't have an embedding and they have less than two neighbors with an embedding
        """
        filtered_dict_neighbors = {}
        memory_node_embeddings = {}

        progress_bar = tqdm(total=len(self.dict_neighbors), desc="Filtering the nodes")

        for node, list_neighbors in self.dict_neighbors.items():
            if node in memory_node_embeddings:
                node_embedding = memory_node_embeddings[node]
            else:
                node_embedding = self.test_has_embedding(node)
                memory_node_embeddings[node] = node_embedding

            # Check if node has an embedding
            if type(self.test_has_embedding(node)) != bool:
                self.ok_count += 1
                filtered_dict_neighbors[node] = node_embedding

            else:
                self.counter += 1
                # The node doesn't have an embedding: check if at least two nodes have an embedding
                count_embeddings = 0
                sum_vectors = np.zeros(self.dim_word)
                for neighbor in list_neighbors:
                    if neighbor[0] in memory_node_embeddings:
                        neighbor_embedding = memory_node_embeddings[neighbor[0]]
                    else:
                        neighbor_embedding = self.test_has_embedding(neighbor[0])
                        memory_node_embeddings[neighbor[0]] = neighbor_embedding

                    if type(neighbor_embedding) != bool:
                        count_embeddings += 1
                        sum_vectors += neighbor_embedding

                if count_embeddings >= 2:
                    # More than 2 neighbors with an embedding, so we can generate an embedding for this node
                    filtered_dict_neighbors[node] = np.true_divide(
                        sum_vectors, count_embeddings
                    )
                    self.created += 1

            progress_bar.update(1)
        progress_bar.close()

        return filtered_dict_neighbors

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
            return self.embedding_method.convert_word_to_embedding(word)
        except:
            self.count += 1

    def compute_wl_embedding(self, n_iterations):
        """
            Computes the Weisfeiler-Lehman continuous embedding for each node
        """
        self.wl_embedding = {}

        # Initialize with the previously computed embeddings
        for node in self.filtered_dict_neighbors:
            self.wl_embedding[node] = np.array(self.filtered_dict_neighbors[node])

        # Start the Weisfeiler-Lehman algorithm
        for _ in tqdm(range(n_iterations), desc="Iterations", position=0, leave=False):
            for node in tqdm(
                self.filtered_dict_neighbors,
                desc="Nodes updates",
                position=1,
                leave=False,
            ):
                neighbors_list = self.dict_neighbors[node]
                new_wl_embedding = 0
                total_weight = 0

                for neighbor in neighbors_list:
                    if neighbor[0] in self.filtered_dict_neighbors:
                        n_name, n_weight = neighbor
                        new_wl_embedding += n_weight * self.wl_embedding[n_name]
                        total_weight += n_weight

                # Normalizes update using the total weight
                if total_weight != 0:
                    new_wl_embedding = np.true_divide(new_wl_embedding, total_weight)

                new_wl_embedding = 0.5 * (
                    self.wl_embedding[node] + new_wl_embedding / len(neighbors_list)
                )

                self.wl_embedding[node] = new_wl_embedding

    def get_wl_embedding(self, word):
        """
            Returns the Weisfeiler-Lehman continuous embedding of the given node
        """
        try:
            return self.wl_embedding[word]
        except Exception as e:
            # print(colored("ERROR: {}".format(e), "red"))
            pass

    def get_wl_embedding_tensor(self, word):
        """
            Returns the Weisfeiler-Lehman continuous embedding (in a tensor) of the given node
        """
        try:
            return torch.from_numpy(self.wl_embedding[word]).double()
        except Exception as e:
            # print(colored("ERROR: {}".format(e), "red"))
            return torch.zeros(self.dim_word).double()
            
    def get_wl_embedding_from_q_id(self, list_q_ids, dim1, dim2):
        """
            Given a list of quetion IDs, returns the embeddings of each word in every question
        """
               
        # Go through the question IDs; for each question ID, load the words in it
        from copy import deepcopy
        question_txt = []
        question_id = deepcopy(list_q_ids)
        question_id = question_id.tolist()
        
        question_embedding = []

        for q_id in question_id:
            question_wl_embedding = []
            sentence = self.dict_txt_questions[q_id]
            print("QUESTION ID (q_id): ", q_id)
            print("SENTENCE: ", sentence)
            list_words = sentence.split(" ")
            for word in list_words:
                word_wl_embedding = self.get_wl_embedding_tensor(word)
                target_emb = torch.zeros((dim2))
                target_emb[:word_wl_embedding.size(0)] = word_wl_embedding
                question_wl_embedding.append(word_wl_embedding)
            
            
            q_emb = torch.stack(question_wl_embedding)
            target_wl = torch.zeros((dim1, dim2))
            target_wl[:q_emb.size(0), :q_emb.size(1)] = q_emb
            question_embedding.append(target_wl)

        return torch.stack(question_embedding)
        
    def get_wl_embedding_tokens(self, input_ids, dim1, dim2):
        """
            Given a list of tokens, returns the WL embedding of each word
        """
        wl_embedding = []
        
        # from copy import deepcopy
        # input_ids_copy = deepcopy(input_ids)
        # input_ids_copy = input_ids_copy.tolist()
        input_ids = input_ids.tolist()
        
        # for question in input_ids_copy:
        for question in input_ids:
            question_embedding = []
            for word_token in question:
                try:
                    word = self.token_dictionary[word_token]
                    word_wl_emb = self.get_wl_embedding_tensor(word)
                except:
                    word_wl_emb = torch.zeros(self.dim_word).double()
                target_emb = torch.zeros((dim2))
                target_emb[:word_wl_emb.size(0)] = word_wl_emb
                question_embedding.append(word_wl_emb)
        
            question_wl_emb = torch.stack(question_embedding)
            target_wl = torch.zeros((dim1, dim2))
            target_wl[:question_wl_emb.size(0), :question_wl_emb.size(1)] = question_wl_emb
            wl_embedding.append(target_wl)
        
        return torch.stack(wl_embedding)
        
    # def get_graph_embedding(self):
    #     """
    #         Returns a matrix, which is the embedding of the graph
    #     """
    #     try:
    #         return self.graph_embedding
    #     except:
    #         list_node_embeddings = []
    #         
    #         for node in self.list_nodes:
    #             node_embedding = self.get_wl_embedding_tensor(node).float()
    #             list_node_embeddings.append(node_embedding)
    #         self.graph_embedding = torch.stack(list_node_embeddings)
    #         
    #         return self.graph_embedding



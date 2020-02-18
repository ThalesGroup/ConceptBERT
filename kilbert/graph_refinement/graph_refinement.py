### LIBRARIES ###
# Global libraries
import os
import json
from copy import deepcopy

import torch
import torch.nn as nn

# Custom libraries
from graph_refinement.importance_index import ImportanceIndex
from graph_refinement.utils import (
    write_node_dictionary,
    write_neighbors_list,
    write_weight_edges,
)

### CLASS DEFINITION ###
class GraphRefinement(nn.Module):
    """
        Model "G1" 
    """

    def __init__(self, conceptnet_embedding):
        super(GraphRefinement, self).__init__()

        # Module to compute the importance index
        self.importance_index = ImportanceIndex()
        # Won't propagate if the weight is smaller than this value
        self.propagation_threshold = 0.5
        # Coefficient multiplied to the weight at each iteration
        self.attenuation_coef = 0.25

        # Load the dictionary of the node indexes
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes_dictionary.json"
        ):
            write_node_dictionary()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes_dictionary.json", "r"
        ) as json_file:
            self.index_nodes_dict = json.load(json_file)

        # Load the list of neighbors for the graph
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_list_neighbors.json"
        ):
            write_neighbors_list()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_list_neighbors.json", "r"
        ) as json_file:
            self.list_neighbors = json.load(json_file)

        # Load the weight edges of the initial graph
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_weight_edges.json"
        ):
            write_weight_edges()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_weight_edges.json", "r"
        ) as json_file:
            self.initial_weight_edges = json.load(json_file)

        # Write dictionary to have the equivalence "edge <-> index"
        index_edge = 0
        edge_to_idx_dict = {}
        for edge in self.initial_weight_edges:
            edge_to_idx_dict[edge] = index_edge
            index_edge += 1
        self.edge_to_idx_dict = edge_to_idx_dict

        # Write initialization tensor representing the graph
        list_weights = []
        for _, weight in self.initial_weight_edges.items():
            list_weights.append(weight)
        self.init_graph_tensor = torch.Tensor(list_weights)

        # Write initialization tensor to keep track of visited edges
        self.init_visited_edges_tensor = torch.Tensor(
            [False for _ in self.initial_weight_edges]
        )

        self.conceptnet_embedding = conceptnet_embedding

    def compute_importance_index(self, list_question_attention):
        """
            Given a sentence, computes the importance index of each word

            Input: 
                - `sentence` (str): sentence to process
            Output: 
                - `list_importance_indexes` (List[float]): list of the importance index of each word
        """
        # Given the word and its attention, computes the importance indexes of each one

        list_importance_indexes = []
        for question_attention in list_question_attention:
            importance_indexes = self.importance_index(question_attention)
            list_importance_indexes.append(importance_indexes)

        return torch.stack(list_importance_indexes)

    def compute_graph_representation(self, graph_tensor, num_max_nodes):
        main_entity_indexes = sorted(
            range(len(graph_tensor)), key=lambda i: graph_tensor[i], reverse=True
        )[:num_max_nodes]
        kg_embedding = []

        for entity_idx in main_entity_indexes:
            kg_embedding.append(
                self.conceptnet_embedding.get_node_embedding_tensor(entity_idx)
            )

        return torch.stack(kg_embedding)

    def forward(
        self, list_questions, attention_question, conceptnet_graph, num_max_nodes
    ):
        """
            For each question in `list_questions`, computes the importance index of each word
            using `attention_question`.
            Then, propagates the importance index through the given ConceptNet graph
            In order to have parallel computation, uses tensors instead of the graph
            At the end, updates the graph weights with a simple addition (graph already normalized)
        """
        ## Step 1: Compute the "constants" in this function
        # TODO: Check if the `self.` values are overwritten by the parallel modules
        # or if everything works as expected

        # Send initialization tensor representing the graph to the right GPU
        self.init_graph_tensor = (
            self.init_graph_tensor.cuda(list_questions.get_device())
            if list_questions.is_cuda
            else self.init_graph_tensor
        )

        # Send initialization tensor to keep track of visited edges to the right GPU
        self.init_visited_edges_tensor = (
            self.init_visited_edges_tensor.cuda(list_questions.get_device())
            if list_questions.is_cuda
            else self.init_visited_edges_tensor
        )

        ## Step 2: Compute the importance index
        importance_indexes = self.compute_importance_index(attention_question)

        ## Step 3: Propagate the weights in the "graph"
        list_kg_embeddings = []

        for i, question in enumerate(list_questions):
            print("New question (device: " + str(attention_question.get_device()) + ")")
            graph_tensor = deepcopy(self.init_graph_tensor)
            for j, entity_index in enumerate(question):
                # Initialize the edges
                visited_edges_tensor = deepcopy(self.init_visited_edges_tensor)
                # Propagate the weights for this entity
                graph_tensor = self.propagate_weights(
                    graph_tensor,
                    visited_edges_tensor,
                    [(entity_index, importance_indexes[i][j])],
                )

            ## Step 4: Build the graph embedding
            question_graph_embedding = self.compute_graph_representation(
                graph_tensor, num_max_nodes
            )
            list_kg_embeddings.append(question_graph_embedding)

        return torch.stack(list_kg_embeddings)

    def translate_question_to_kg(self, q_index):
        """
            Given an index from a question, gives the equivalent index in the knowledge
            graph, if it exists.
            If it doesn't exist, returns an error.
        """
        try:
            word = self.conceptnet_embedding.token_dictionary[q_index]
            kb_index = self.index_nodes_dict[word]

            return kb_index

        except Exception as e:
            print("ERROR: ", e)

    def propagate_weights(self, graph_tensor, visited_edges_tensor, waiting_list):
        """
            Given the index of an entity, propagates the weights around it
        """
        if len(waiting_list) == 0:
            return graph_tensor
        else:
            entity_in_question, importance_index = waiting_list.pop(0)

            if importance_index >= self.propagation_threshold:
                # Convert entity in question to entity in knowledge graph
                entity_kg = self.translate_question_to_kg(entity_in_question)
                list_neighbors = self.list_neighbors[entity_kg]

                for neighbor in list_neighbors:
                    edge = (
                        "["
                        + str(min(entity_kg, neighbor))
                        + ";"
                        + str(max(entity_kg, neighbor))
                        + "]"
                    )
                    edge_index = self.edge_to_idx_dict[edge]

                    if not visited_edges_tensor[edge_index]:
                        graph_tensor[edge_index] += importance_index
                        visited_edges_tensor[edge_index] = True

                        if (
                            importance_index * self.attenuation_coef
                            >= self.propagation_threshold
                        ):
                            new_list_neighbors = self.list_neighbors[neighbor]

                            for new_neighbor in new_list_neighbors:
                                waiting_list.append(
                                    (
                                        new_neighbor,
                                        importance_index * self.attenuation_coef,
                                    )
                                )

                return self.propagate_weights(
                    graph_tensor, visited_edges_tensor, waiting_list
                )


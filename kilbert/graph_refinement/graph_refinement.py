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
    extract_nodes,
    write_node_dictionary,
    write_neighbors_list,
    write_weight_edges,
    sort_initial_weight_edges_list,
)

### CLASS DEFINITION ###
class GraphRefinement(nn.Module):
    """
        Model "G1" 
    """

    def __init__(self, conceptnet_embedding, num_max_nodes):
        super(GraphRefinement, self).__init__()

        # Module to compute the importance index
        self.importance_index = ImportanceIndex()
        # Won't propagate if the weight is smaller than this value
        self.propagation_threshold = 0.5
        # Coefficient multiplied to the weight at each iteration
        self.attenuation_coef = 0.25

        # Load the list of nodes in ConceptNet
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes.json"
        ):
            extract_nodes()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes.json", "r"
        ) as json_file:
            self.list_nodes = json.load(json_file)

        # Load the dictionary of the node tokens in ConceptNet
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes_dictionary.json"
        ):
            write_node_dictionary()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes_dictionary.json", "r"
        ) as json_file:
            self.index_nodes_dict = json.load(json_file)

        # Load the list of neighbors for the graph (given a ConceptNet token)
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

        # Load the ordered list of weights
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_ordered_weights_list.json"
        ):
            sort_initial_weight_edges_list()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_ordered_weights_list.json",
            "r",
        ) as json_file:
            ordered_edge_weights_list = json.load(json_file)

        """
                We want to reach the limit `num_max_nodes`, but having `num_max_nodes` edges doesn't
            mean we have `num_max_nodes` different nodes.
            In a graph with k nodes, we can have a maximum of k(k-1)/2 edges (we assume a single edge
            can link two nodes).
            Thus, if we want to have at least k nodes, we need to consider at least 1 + (k-1)(k-2)/2 edges
        """
        num_edges = 1 + int((num_max_nodes - 1) * (num_max_nodes - 2) / 2)
        self.ordered_edge_weights_list = ordered_edge_weights_list[:num_edges]

        # Write dictionary to have the equivalence "edge -> edge index"
        index_edge = 0
        edge_to_idx_dict = {}
        for edge in self.initial_weight_edges:
            edge_to_idx_dict[edge] = index_edge
            index_edge += 1
        self.edge_to_idx_dict = edge_to_idx_dict

        # Write list to have the equivalence "edge index -> edge"
        idx_to_edge_list = []
        for edge, index in self.edge_to_idx_dict.items():
            idx_to_edge_list.append(edge)
        self.idx_to_edge_list = idx_to_edge_list

        # Write initialization tensor representing the graph
        list_weights = []
        for _, weight in self.initial_weight_edges.items():
            list_weights.append(weight["weight"])
        self.init_graph_tensor = torch.Tensor(list_weights)

        # Write initialization tensor to keep track of visited edges
        self.init_visited_edges_tensor = torch.BoolTensor(
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

    def compute_graph_representation(
        self, graph_tensor, list_max_weights, num_max_nodes
    ):
        """
            list_max_weights: [[index_edge, weight_edge]]
        """
        set_nodes = set()
        for entity in list_max_weights:
            index_edge = entity[0]
            # Convert index edge to the string value
            str_edge = self.idx_to_edge_list[index_edge]
            str_edge = str_edge.replace("[", "").replace("]", "")
            list_nodes = str_edge.split(";")
            start_node = int(list_nodes[0])
            end_node = int(list_nodes[1])

            set_nodes.add(start_node)
            if len(set_nodes) >= num_max_nodes:
                break
            set_nodes.add(end_node)
            if len(set_nodes) >= num_max_nodes:
                break

        list_main_entities = list(set_nodes)
        if str(graph_tensor.get_device()) == "0":
            print(
                "List main entities from device "
                + str(graph_tensor.get_device())
                + " : ",
                list_main_entities,
            )

        # Get the embedding of each word
        kg_embedding = []

        for entity_idx in list_main_entities:
            word = self.list_nodes[entity_idx]
            kg_embedding.append(
                self.conceptnet_embedding.get_node_embedding_tensor(word)
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
        device = list_questions.get_device() if list_questions.is_cuda else -1
        self.init_graph_tensor = self.init_graph_tensor.cuda(device)

        # Send initialization tensor to keep track of visited edges to the right GPU
        self.init_visited_edges_tensor = self.init_visited_edges_tensor.cuda(device)

        ## Step 2: Compute the importance index
        importance_indexes = self.compute_importance_index(attention_question)
        print(
            "Importance indexes on device " + str(device) + " : ",
            importance_indexes[:3],
        )

        ## Step 3: Propagate the weights in the "graph"
        list_kg_embeddings = []

        for i, question in enumerate(list_questions):
            # if device == 0:
            #     print("New question (device: " + str(device) + ")")
            graph_tensor = deepcopy(self.init_graph_tensor)
            list_max_weights = self.ordered_edge_weights_list

            for j, entity_index in enumerate(question):
                # Initialize the edges
                visited_edges_tensor = deepcopy(self.init_visited_edges_tensor)
                # Propagate the weights for this entity
                graph_tensor, list_max_weights = self.propagate_weights(
                    graph_tensor,
                    visited_edges_tensor,
                    list_max_weights,
                    [(entity_index, importance_indexes[i][j])],
                )
            # if device == 0:
            #     print("Building the graph embedding")
            ## Step 4: Build the graph embedding
            question_graph_embedding = self.compute_graph_representation(
                graph_tensor, list_max_weights, num_max_nodes
            )
            list_kg_embeddings.append(question_graph_embedding)
            # if device == 0:
            #     print("Question done, onto the next one")

        knowledge_graph_embedding = torch.stack(list_kg_embeddings).float()
        # Send `knowledge_graph_embedding` to the correct device
        knowledge_graph_embedding = knowledge_graph_embedding.cuda(device)

        return knowledge_graph_embedding

    # def translate_question_to_kg(self, q_index):
    #     """
    #         Given an index from a question, gives the equivalent index in the knowledge
    #         graph, if it exists.
    #         If it doesn't exist, returns an error.
    #     """
    #     try:
    #         word = self.conceptnet_embedding.token_dictionary[int(q_index.item())]
    #         kb_index = self.index_nodes_dict[word]
    #         # Convert kb_index to tensor and send it to the correct device
    #         kb_index = (
    #             torch.Tensor(kb_index).cuda(q_index.get_device())
    #             if q_index.is_cuda
    #             else kb_index
    #         )
    #         return kb_index
    #
    #     except Exception as e:
    #         if word not in ["[CLS]", "[SEP]", "'", "?"]:
    #             print("ERROR in `translate_question_to_kg`: ", e)

    def propagate_weights(
        self, graph_tensor, visited_edges_tensor, list_max_weights, waiting_list
    ):
        """
            Given the index of an entity, propagates the weights around it
        """
        if len(waiting_list) == 0:
            return graph_tensor, list_max_weights
        else:
            # entity_in_question, importance_index = waiting_list.pop(0)
            entity_kg, importance_index = waiting_list.pop(0)

            if entity_kg != -1 and importance_index >= self.propagation_threshold:
                # Convert entity in question to entity in knowledge graph
                try:
                    # entity_kg = self.translate_question_to_kg(entity_in_question)
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

                            # Check if the new weight is bigger than the smallest weight
                            # in `list_max_weights`
                            if graph_tensor[edge_index] > list_max_weights[-1][1]:
                                is_in_max_list = False
                                new_position = 0
                                # If the weight was already in the list_max_weights, just update its weight
                                for i, entity in enumerate(list_max_weights):
                                    if entity[1] < graph_tensor[edge_index]:
                                        new_position = i
                                    if entity[0] == edge_index:
                                        list_max_weights[i][
                                            1
                                        ] += importance_index.item()
                                        is_in_max_list = True
                                        # Sort the new list
                                        new_list = list_max_weights[: i + 1]
                                        new_list.sort(key=lambda x: x[1], reverse=True)
                                        list_max_weights[: i + 1] = new_list
                                        break

                                if not is_in_max_list:
                                    # Check where to add the new weight
                                    for i, entity in enumerate(list_max_weights):
                                        if entity[0] < graph_tensor[edge_index]:
                                            new_position = i
                                            break
                                    # Update `list_max_weights`, so that it is still sorted
                                    list_max_weights.pop()
                                    list_max_weights.insert(
                                        new_position,
                                        [edge_index, graph_tensor[edge_index].item()],
                                    )

                                """
                                # Sort the list
                                # TODO: Instead of a sort, just input the new weight at the correct place
                                # This can be done beforehand when checking if the new weight is already
                                # in the list, or be done again here
                                list_max_weights.sort(key=lambda x: x[1], reverse=True)
                                """
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
                except Exception as e:
                    print("ERROR in `propagate_weights`: ", e)
                    pass

            return self.propagate_weights(
                graph_tensor, visited_edges_tensor, list_max_weights, waiting_list
            )


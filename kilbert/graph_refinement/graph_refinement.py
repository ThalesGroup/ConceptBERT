### LIBRARIES ###
# Global libraries
import os
import sys
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
        self.attenuation_coef = 0.1667

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
        self, graph_tensor, tensor_max_weights, num_max_nodes
    ):
        """
            tensor_max_weights: [[index_edge, weight_edge]]
        """
        # if str(graph_tensor.get_device()) == "0":
        #     print(
        #         "First weights from `tensor_max_weights`: ", tensor_max_weights[5:]
        #     )

        set_nodes = set()
        # list_weights = []
        for entity in tensor_max_weights:
            index_edge = int(entity[0].item())

            # weight_edge = entity[1].item()
            # list_weights.append(weight_edge)

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

        # TODO: Convert `list_main_entities` to tensor
        list_main_entities = list(set_nodes)
        if str(graph_tensor.get_device()) == "0":
            # Have the equivalent words
            list_main_words = []
            for entity in list_main_entities:
                list_main_words.append(self.list_nodes[entity])
            print("List main entities from device 0: ", list_main_words)

        # Print the weight of the main entities too:
        # print("List weights main entities: ", list_weights)

        # Get the embedding of each word
        # TODO: Instead of adding to a list and sending the list, just
        # complete the existing tensor
        kg_embedding = []

        for entity_idx in list_main_entities:
            word = self.list_nodes[entity_idx]
            kg_embedding.append(
                self.conceptnet_embedding.get_node_embedding_tensor(word)
            )

        return torch.stack(kg_embedding)

    def forward(
        self, list_questions, attention_question, num_max_nodes,
    ):
        """
            For each question in `list_questions`, computes the importance index of each word
            using `attention_question`.
            Then, propagates the importance index through the given ConceptNet graph
            In order to have parallel computation, uses tensors instead of the graph
            At the end, updates the graph weights with a simple addition (graph already normalized)
        """
        ## Step 1: Compute the "constants" in this function

        # Send initialization tensor representing the graph to the right GPU
        device = list_questions.get_device() if list_questions.is_cuda else -1
        self.init_graph_tensor = self.init_graph_tensor.cuda(device)

        # Send initialization tensor to keep track of visited edges to the right GPU
        self.init_visited_edges_tensor = self.init_visited_edges_tensor.cuda(device)

        ## Step 2: Compute the importance index
        importance_indexes = self.compute_importance_index(attention_question)
        # if device == 0:
        #     print("Importance indexes on device 0: ", importance_indexes[:3])

        ## Step 3: Propagate the weights in the "graph"
        list_kg_embeddings = []

        for i, question in enumerate(list_questions):
            if device == 0:
                print("New question (device: " + str(device) + ")")
                print("Importance indexes: ", importance_indexes[i])
                # Read the question here
                list_index_words = []
                for entity_index in question:
                    if int(entity_index.item()) == -1:
                        list_index_words.append(-1)
                    else:
                        list_index_words.append(
                            self.list_nodes[int(entity_index.item())]
                        )
                print("QUESTION : ", list_index_words)
            graph_tensor = torch.zeros_like(self.init_graph_tensor)
            # graph_tensor = deepcopy(self.init_graph_tensor)
            # Convert the list of max_weights to a tensor
            # list_max_weights = self.ordered_edge_weights_list
            # tensor_max_weights = torch.Tensor(self.ordered_edge_weights_list).cuda(
            #     device
            # )
            tensor_max_weights = torch.zeros_like(
                torch.Tensor(self.ordered_edge_weights_list).cuda(device)
            )

            for j, entity_index in enumerate(question):
                # Initialize the edges
                visited_edges_tensor = torch.BoolTensor(
                    len(self.initial_weight_edges) * [False]
                )
                # Propagate the weights for this entity
                graph_tensor, tensor_max_weights = self.propagate_weights(
                    graph_tensor,
                    visited_edges_tensor,
                    # list_max_weights,
                    tensor_max_weights,
                    torch.Tensor([[entity_index, importance_indexes[i][j]]]),
                )
            if device == 0:
                print("Building the graph embedding")
            ## Step 4: Build the graph embedding
            question_graph_embedding = self.compute_graph_representation(
                # graph_tensor, list_max_weights, num_max_nodes
                graph_tensor,
                tensor_max_weights,
                num_max_nodes,
            )
            print("Updating `list_kg_embeddings`")
            list_kg_embeddings.append(question_graph_embedding)
            # if device == 0:
            #     print("Question done, onto the next one")

        knowledge_graph_embedding = torch.stack(list_kg_embeddings).float()
        # Send `knowledge_graph_embedding` to the correct device
        knowledge_graph_embedding = knowledge_graph_embedding.cuda(device)

        return knowledge_graph_embedding

    def update_sorted_list(self, original_tensor, index_modif, new_index):
        """
            Given a tensor and the index of the modified value,
            returns a new tensor which is correctly sorted.
        """
        new_value = original_tensor[index_modif]
        length_tensor = original_tensor.shape[0]

        # Create the new tensor
        sorted_tensor = torch.zeros_like(original_tensor)
        # Fill the new tensor
        for i in range(new_index):
            sorted_tensor[i] = original_tensor[i]
        sorted_tensor[new_index] = new_value
        if new_index + 1 == index_modif:
            sorted_tensor[new_index + 1] = original_tensor[new_index]
        else:
            for i in range(new_index + 1, index_modif):
                sorted_tensor[i] = original_tensor[i - 1]
        for i in range(index_modif + 1, length_tensor):
            sorted_tensor[i] = original_tensor[i]

        return sorted_tensor

    def add_and_update(self, original_tensor, new_position, new_entity):
        """
            The entity `new_entity` doesn't exist in `original_tensor`, so 
            add it at `new_position` and push the other weights
        """
        length_tensor = original_tensor.shape[0]
        # Create the new tensor
        sorted_tensor = torch.zeros_like(original_tensor)
        # Fill the new tensor
        for i in range(new_position):
            sorted_tensor[i] = original_tensor[i]
        sorted_tensor[new_position] = new_entity
        for i in range(new_position + 1, length_tensor):
            sorted_tensor[i] = original_tensor[i - 1]

        return sorted_tensor

    def propagate_weights(
        self, graph_tensor, visited_edges_tensor, tensor_max_weights, waiting_tensor
    ):
        """
            Given the index of an entity, propagates the weights around it
        """
        device = graph_tensor.get_device()
        while len(waiting_tensor) > 0:
            # print("WAITING TENSOR: ", waiting_tensor)
            entity_kg, importance_index = waiting_tensor[0]
            waiting_tensor = waiting_tensor[1:]

            if (
                entity_kg.item() != -1
                and importance_index >= self.propagation_threshold
            ):
                # Convert entity in question to entity in knowledge graph
                try:
                    list_neighbors = self.list_neighbors[int(entity_kg.item())]

                    for neighbor in list_neighbors:
                        edge = (
                            "["
                            + str(min(int(entity_kg.item()), neighbor))
                            + ";"
                            + str(max(int(entity_kg.item()), neighbor))
                            + "]"
                        )
                        edge_index = self.edge_to_idx_dict[edge]

                        if not visited_edges_tensor[edge_index]:
                            graph_tensor[edge_index] += importance_index
                            visited_edges_tensor[edge_index] = True

                            # Check if the new weight is bigger than the
                            # smallest weight in `tensor_max_weights`
                            if graph_tensor[edge_index] > tensor_max_weights[-1][1]:
                                # Check if the edge is already in the list of the
                                # heaviest weights and update it
                                is_in_max_list = False
                                check_position = True
                                for i, entity in enumerate(tensor_max_weights):
                                    if (
                                        check_position
                                        and entity[1] < graph_tensor[edge_index]
                                    ):
                                        new_position = i
                                        check_position = False
                                    if entity[0] == edge_index:
                                        if check_position:
                                            new_position = i
                                        tensor_max_weights[i][1] += importance_index
                                        is_in_max_list = True
                                        tensor_max_weights = self.update_sorted_list(
                                            tensor_max_weights, i, new_position
                                        )
                                        break

                                if not is_in_max_list:
                                    # Update `list_max_weights`, so that it
                                    # is still sorted
                                    tensor_max_weights = self.add_and_update(
                                        tensor_max_weights,
                                        new_position,
                                        torch.Tensor(
                                            [
                                                edge_index,
                                                graph_tensor[edge_index].item(),
                                            ]
                                        ),
                                    )

                            # Continue the propagation
                            if (
                                importance_index * self.attenuation_coef
                                >= self.propagation_threshold
                            ):
                                neighbor_tensor = torch.zeros_like(entity_kg) + neighbor
                                waiting_tensor = torch.cat(
                                    [
                                        waiting_tensor,
                                        torch.Tensor(
                                            [
                                                [
                                                    neighbor_tensor,
                                                    importance_index
                                                    * self.attenuation_coef,
                                                ]
                                            ]
                                        ),
                                    ]
                                )

                except Exception as e:
                    error_type, error_obj, error_tb = sys.exc_info()
                    # print("ERROR in `propagate_weights`: ", e)
                    print(error_type, error_obj, error_tb.tb_lineno)
                    # pass

        # return graph_tensor, list_max_weights
        return graph_tensor, tensor_max_weights

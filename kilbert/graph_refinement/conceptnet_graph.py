### LIBRARIES ###
# Global libraries
import os
import json

# Custom libraries
from graph_refinement.utils import (
    write_node_dictionary,
    write_neighbors_list,
    write_weight_edges,
)

### CLASS DEFINITION ###
class ConceptNet:
    """
        ConceptNet object
    """

    def __init__(self):
        # Load the dictionary of the node indexes
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes_dictionary.json"
        ):
            write_node_dictionary()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes_dictionary.json", "r"
        ) as json_file:
            self.index_nodes_dict = json.load(json_file)

        list_nodes = ["" for _ in range(len(self.index_nodes_dict))]
        for word, index in self.index_nodes_dict.items():
            list_nodes[self.index_nodes_dict[word]] = index
        self.list_nodes = list_nodes

        # Load the list of list of neighbors
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_list_neighbors.json"
        ):
            write_neighbors_list()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_list_neighbors.json", "r"
        ) as json_file:
            self.list_neighbors = json.load(json_file)

        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_weight_edges.json"
        ):
            write_weight_edges()

        with open(
            "/nas-data/vilbert/data2/conceptnet/processed/cn_weight_edges.json", "r"
        ) as json_file:
            self.weight_edges = json.load(json_file)

    def get_index(self, word):
        """
            Given a word, finds the corresponding index

            Input: 
                - `word` (str): word to find in the graph 
            Output: 
                - `self.index_nodes_dict[word]` (int): index of `word` in the graph
        """
        try:
            return self.index_nodes_dict[word]
        except Exception as e:
            print("ERROR: ", e)

    def get_word(self, index):
        """
            Given an index, finds the corresponding word
        """
        try:
            return self.list_nodes[index]
        except Exception as e:
            print("ERROR: ", e)

    def propagate_weights(self, waiting_list, propagation_threshold, attenuation_coef):
        """
            Given an entity, propagates the weights around it
        """
        if len(waiting_list) == 0:
            pass
        else:
            entity, importance_index = waiting_list.pop(0)

            if importance_index >= propagation_threshold:
                list_neighbors = self.list_neighbors[entity]

                for neighbor in list_neighbors:
                    edge = (
                        "["
                        + str(min(entity, neighbor))
                        + ";"
                        + str(max(entity, neighbor))
                        + "]"
                    )
                    if not self.weight_edges[edge]["updated"]:
                        self.weight_edges[edge]["weight"] += importance_index
                        self.weight_edges[edge]["updated"] = True

                        new_list_neighbors = self.list_neighbors[neighbor]
                        for new_neighbor in new_list_neighbors:
                            waiting_list.append(
                                (new_neighbor, importance_index * attenuation_coef)
                            )

    def normalize_weights(self):
        """
            Normalizes the weights of the edges
        """
        # First, find the maximum weight of all the edges
        maximum_weight = 0
        for edge in self.weight_edges:
            if self.weight_edges[edge]["weight"] > maximum_weight:
                maximum_weight = self.weight_edges[edge]["weight"]

        # Then, normalize all the weights
        for edge in self.weight_edges:
            self.weight_edges[edge]["weight"] = float(
                self.weight_edges[edge]["weight"] / maximum_weight
            )

    def select_top_edges(self, max_nodes):
        """
            Returns a vector with the indexes of the heaviest edges
        """
        set_nodes = set()

        # Sort weight edges in decreasing order
        self.weight_edges = {
            k: v
            for k, v in sorted(
                self.weight_edges.items(), key=lambda item: item[1], reverse=True
            )
        }

        # Add the nodes seen in `weight_edges`, until the limit `max_nodes` is reached
        for edge in self.weight_edges:
            str_edge = edge.replace("[", "").replace("]", "")
            list_nodes = str_edge.split(";")
            start_node = list_nodes[0]
            end_node = list_nodes[1]

            set_nodes.add(start_node)
            if len(set_nodes) >= max_nodes:
                break
            set_nodes.add(end_node)
            if len(set_nodes) >= max_nodes:
                break

        # Convert the node indexes to words
        list_main_entities = []
        for node in list(set_nodes):
            list_main_entities.append(self.get_word(int(node)))

        return list_main_entities

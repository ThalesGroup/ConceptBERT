### LIBRARIES ###
# Global libraries
import os
import csv
import json
import ast
from tqdm import tqdm
import codecs

# Custom libraries
from utils_functions_bis import download_file

### FUNCTION DEFINITIONS ###
def conceptnet_loader():
    """
        Reads the `assertions.csv` file line by line
    """
    # Checks if the file exists
    if not os.path.exists("/home/mziaeefard/nas/human-ai-dialog/vilbert/data2/conceptnet/raw/assertions.csv"):
        # Download the assertions.csv file
        download_file(
            "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
            "/home/mziaeefard/nas/human-ai-dialog/vilbert/data2/conceptnet/raw/assertions.csv",
        )

    with codecs.open("/home/mziaeefard/nas/human-ai-dialog/vilbert/data2/conceptnet/raw/assertions.csv", encoding="utf-8"
    ) as csv_file:
        datareader = csv.reader(csv_file, delimiter="\t",)
        for row in datareader:
            yield row


def extract_nodes():
    """
        Extracts the node (English language only) from ConceptNet and saves them
    """
    set_nodes = set()

    progress_bar = tqdm(
        range(34074917), desc="Lines processed extracting"
    )  # Number of lines in `assertions.csv`

    for row in conceptnet_loader():
        start_node = row[2]
        end_node = row[3]

        start_split = start_node.split("/")
        end_split = end_node.split("/")

        if start_split[2] == "en" and end_split[2] == "en":
            
            set_nodes.add(start_split[3])
            set_nodes.add(end_split[3])

        progress_bar.update(1)

    progress_bar.close()

    # Save the nodes in a file
    with open("/home/mziaeefard/nas/human-ai-dialog/vilbert/data2/conceptnet/processed/en_nodes.json", "w") as json_file:
        json.dump(list(set_nodes), json_file)
        print("`en_nodes.json` saved")


def compute_node_dictionary():
    """
        Given the list of nodes, compute the reverse.
        The result is a dict {word (str): index (int)}
    """
    with open(
       "/home/mziaeefard/nas/human-ai-dialog/vilbert/data2/conceptnet/processed/en_nodes.json", "r"
    ) as json_file:
        list_nodes = json.load(json_file)

    index = 0
    dict_nodes = {}
    progress_bar = tqdm(range(len(list_nodes)), desc="Creating the node dictionary")
    for node in list_nodes:
        dict_nodes[node] = index
        index += 1
        progress_bar.update(1)
    progress_bar.close()

    # Save the dictionary
    with open(
        "/home/mziaeefard/nas/human-ai-dialog/vilbert/data2/conceptnet/processed/en_nodes_dictionary.json", "w"
    ) as json_file:
        json.dump(dict_nodes, json_file)
        print("`en_nodes_dictionary.json` saved")


def compute_neighborhoods():
    """
        Given the nodes and the edges, computes the neighborhood of each node
    """
    # Load the dictionary
    with open(
        "/home/mziaeefard/nas/human-ai-dialog/vilbert/data2/conceptnet/processed/en_nodes_dictionary.json", "r"
    ) as json_file:
        node_dictionary = json.load(json_file)

    # Initialization of the dictionary
    dict_neighbors = {}
    for node in node_dictionary:
        dict_neighbors[node] = []

    progress_bar = tqdm(
        range(34074917), desc="Lines processed"
    )  # Number of lines in `assertions.csv`

    for row in conceptnet_loader():
        start_node = row[2]
        end_node = row[3]

        start_split = start_node.split("/")
        end_split = end_node.split("/")

        if start_split[2] == "en" and end_split[2] == "en":
            edge_weight = ast.literal_eval(row[4])["weight"]

            start_index = node_dictionary[start_split[3]]
            end_index = node_dictionary[end_split[3]]

            dict_neighbors[start_split[3]].append((end_index, edge_weight))
            dict_neighbors[end_split[3]].append((start_index, edge_weight))

        progress_bar.update(1)

    progress_bar.close()

    # Save the neighborhood dictionary in a file
    with open(
        os.path.join("/home/mziaeefard/nas/human-ai-dialog/vilbert/data2/conceptnet/processed/en_dict_neighbors.json"), "w"
    ) as json_file:
        json.dump(dict_neighbors, json_file)
        print("`en_dict_neighbors.json` saved")
        
        
extract_nodes()
compute_node_dictionary()
compute_neighborhoods()


### LIBRARIES ###
# Global libraries
import os
import codecs
import ast

import csv
import json
import _pickle

from termcolor import colored
from tqdm import tqdm

import requests
import io
import zipfile
import gzip
import shutil

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer

### FUNCTION DEFINITIONS ###


def download_file(source_url, dest_path, source_path=""):
    """
        Downloads the given archive and extracts it
        Currently works for: 
            - `zip` files
            - `tar.gz` files

        Inputs: 
            - `source_url` (str): URL to download the ZIP file
            - `source_path` (str): path of the file in the ZIP file
            - `dest_path` (str): path of the extracted file
    """
    # Initiate the request
    r = requests.get(source_url, stream=True)

    # Measure the total size of the ZIP file
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    t = tqdm(total=total_size, unit="iB", unit_scale=True)

    file_extension = source_url.split(".")[-1]

    if file_extension == "zip":
        # Save the ZIP file in a temporary ZIP file
        with open(os.path.join("data", "raw", "temp.zip"), "wb") as f:
            for data in r.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            print(
                colored(
                    "ERROR: Something went wrong while downloading the ZIP file", "red"
                )
            )

        z = zipfile.ZipFile(os.path.join("data", "raw", "temp.zip"))
        # Extract the file from the temporary file
        if source_path != "":
            z.extract(source_path, os.path.dirname(dest_path))
            os.rename(os.path.join(os.path.dirname(dest_path), source_path), dest_path)
        else:
            z.extractall(os.path.dirname(dest_path))
            # z.extractall(dest_path.split(os.path.sep)[:-1])

        # Remove the temporary file
        os.remove(os.path.join("data", "raw", "temp.zip"))

    elif file_extension == "gz":
        # Save the GZ file in a temporary GZ file
        with open(os.path.join("data", "raw", "temp.gz"), "wb") as temp_file:
            for data in r.iter_content(block_size):
                t.update(len(data))
                temp_file.write(data)
        t.close()

        if total_size != 0 and t.n != total_size:
            print(
                colored(
                    "ERROR: Something went wrong while downloading the GZ file", "red"
                )
            )

        with gzip.open(os.path.join("data", "raw", "temp.gz"), "rb") as file_in:
            with open(dest_path, "wb") as file_out:
                shutil.copyfileobj(file_in, file_out)

        # Remove the temporary file
        os.remove(os.path.join("data", "raw", "temp.gz"))


def download_and_save_file(downloaded_file, message_name, cfg, dataset):
    """
        Downloads the specified file and saves it to the given path.

        Inputs: 
            - `downloaded_file` (str): name of the downloaded file (cf. `configs/main.yml`)
            - `message_name` (str): name of the downloaded file that will be displayed
    """
    if not os.path.exists(cfg["paths"][dataset][downloaded_file]):
        try:
            print(colored("Downloading {}...".format(message_name), "yellow"))
            download_file(
                cfg["download_links"][dataset][downloaded_file],
                cfg["paths"][dataset][downloaded_file],
                cfg["paths"][dataset][downloaded_file + "_raw"],
            )
            print(colored("{} was successfully saved.".format(message_name), "green"))

        except Exception as e:
            print(colored("ERROR: {}".format(e), "red"))


def conceptnet_loader():
    """
        Reads the `assertions.csv` file line by line
    """
    # Checks if the file exists
    if not os.path.exists("/nas-data/vilbert/data2/conceptnet/raw/assertions.csv"):
        # Download the `assertions.csv` file
        download_file(
            "https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz",
            "/nas-data/vilbert/data2/conceptnet/raw/assertions.csv",
        )

    with codecs.open(
        "/nas-data/vilbert/data2/conceptnet/raw/assertions.csv", encoding="utf-8"
    ) as csv_file:
        datareader = csv.reader(csv_file, delimiter="\t")
        for row in datareader:
            yield row


def extract_nodes():
    """
        Extracts the nodes (English language only) from ConceptNet and saves them
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
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes.json", "w"
    ) as json_file:
        json.dump(list(set_nodes), json_file)
        print(colored("`cn_nodes.json` saved", "green"))


def write_node_dictionary():
    """
        Given the list of nodes, computes the reverse.
        The result is a dictionary like this: 
            {
                "word": int(index)
            }
    """
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes.json", "r"
    ) as json_file:
        list_nodes = json.load(json_file)

    index = 0
    dict_nodes = {}
    for node in tqdm(list_nodes, desc="Creating the node dictionary"):
        dict_nodes[node] = index
        index += 1

    # Save the dictionary
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/cn_nodes_dictionary.json", "w"
    ) as json_file:
        json.dump(dict_nodes, json_file)
        print(colored("`cn_nodes_dictionary.json` saved", "green"))


def write_neighbors_list():
    """
        Given the nodes and the edges, computes the neighborhood of each node
    """
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/en_nodes_dictionary.json", "r"
    ) as json_file:
        node_dictionary = json.load(json_file)

    # Initialization of the list
    list_neighbors = []
    for node in node_dictionary:
        list_neighbors.append([])

    progress_bar = tqdm(
        range(34074917), desc="Extracting neighbors list"
    )  # Number of lines in `assertions.csv`

    for row in conceptnet_loader():
        start_node = row[2]
        end_node = row[3]
        start_split = start_node.split("/")
        end_split = end_node.split("/")

        if start_split[2] == "en" and end_split[2] == "en":
            start_index = node_dictionary[start_split[3]]
            end_index = node_dictionary[end_split[3]]

            list_neighbors[start_index].append(end_index)
            list_neighbors[end_index].append(start_index)

        progress_bar.update(1)
    progress_bar.close()

    # Save the neighborhood dictionary in a file
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/cn_list_neighbors.json", "w"
    ) as json_file:
        json.dump(list_neighbors, json_file)
        print(colored("`cn_list_neighbors.json` saved", "green"))


def write_weight_edges():
    """
        weight_edges = {
            "[0, 1]": {
                "weight": float,
                "updated": bool,
            }
        }
    """
    """
        Given the nodes and the edges, computes the weights of each edge
    """
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/en_nodes_dictionary.json", "r"
    ) as json_file:
        node_dictionary = json.load(json_file)

    # Initialization of the dictionary
    weight_edges = {}

    progress_bar = tqdm(
        range(34074917), desc="Extracting edge weights"
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

            edge = (
                "["
                + str(min(start_index, end_index))
                + ";"
                + str(max(start_index, end_index))
                + "]"
            )

            weight_edges[edge] = {"weight": edge_weight, "updated": False}

        progress_bar.update(1)
    progress_bar.close()

    # Save the weight edges dictionary in a file
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/cn_weight_edges.json", "w"
    ) as json_file:
        json.dump(weight_edges, json_file)
        print(colored("`cn_weight_edges.json` saved", "green"))


def sort_initial_weight_edges_list():
    """
        Given the weights of the edges of a graph, sort the edges from the heaviest edge to the lightest one.
        Save this list in a file to be loaded afterwards.
    """
    # Load the weights of the edges
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/cn_weight_edges.json", "r"
    ) as json_file:
        weight_edges = json.load(json_file)

    list_idx_weights = []
    index_edge = 0
    for _, weight in weight_edges.items():
        list_idx_weights.append([index_edge, weight["weight"]])
        index_edge += 1

    print("Beginning sorting the list...")
    sorted_list_idx = sorted(
        range(len(list_idx_weights)), key=lambda i: list_idx_weights[i], reverse=True
    )

    # Add the weights in the file and normalize them
    max_weight = list_idx_weights[sorted_list_idx[0]]
    sorted_list_idx_weight = []
    for index in sorted_list_idx:
        sorted_list_idx_weight.append(
            [index, float(list_idx_weights[index]["weight"]) / max_weight]
        )

    # Save in the file
    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/cn_ordered_weights_list.json", "w"
    ) as json_file:
        json.dump(sorted_list_idx_weight, json_file)
        print(colored("`cn_ordered_weights_list.json` saved", "green"))

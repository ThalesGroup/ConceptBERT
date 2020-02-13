### LIBRARIES ###
# Global libraries
import os
import codecs
import ast

import _pickle
import json

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


def initiate_embeddings():
    """
        Saves the embedding dictionary in a file
    """
    # Check if the raw file exists
    if not os.path.exists("/nas-data/vilbert/data2/conceptnet/raw/embeddings.txt"):
        # Download the embeddings file
        download_file(
            "https://ttic.uchicago.edu/~kgimpel/comsense_resources/embeddings.txt.gz",
            "/nas-data/vilbert/data2/conceptnet/raw/embeddings.txt",
        )

    dict_embedding = {}
    with open("/nas-data/vilbert/data2/conceptnet/raw/embeddings.txt", "r") as raw_file:
        for entry in tqdm(raw_file, desc="Saving the node embeddings"):
            entry.strip()
            if entry:
                embedding_split = entry.split(" ")
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
        initiate_embeddings()

    with open(
        "/nas-data/vilbert/data2/conceptnet/processed/embeddings.pkl", "rb"
    ) as pkl_file:
        dict_embedding = _pickle.load(pkl_file)

    return dict_embedding


def get_txt_questions(split):
    """
        Returns the text of the questions
    """

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

    question_path = (
        "/nas-data/vilbert/data2/OK-VQA/OpenEnded_mscoco_"
        + str(split)
        + "2014_questions.json"
    )
    questions = sorted(
        json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
    )
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


def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_BERT_CACHE
    if sys.version_info[0] == 3 and isinstance(url_or_filename, Path):
        url_or_filename = str(url_or_filename)
    if sys.version_info[0] == 3 and isinstance(cache_dir, Path):
        cache_dir = str(cache_dir)

    parsed = urlparse(url_or_filename)

    if parsed.scheme in ("http", "https", "s3"):
        # URL, so get it from the cache (downloading if necessary)
        return get_from_cache(url_or_filename, cache_dir)
    elif os.path.exists(url_or_filename):
        # File, and it exists.
        return url_or_filename
    elif parsed.scheme == "":
        # File, but it doesn't exist.
        raise EnvironmentError("file {} not found".format(url_or_filename))
    else:
        # Something unknown
        raise ValueError(
            "unable to parse {} as a URL or as a local path".format(url_or_filename)
        )


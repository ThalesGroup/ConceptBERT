### LIBRARIES ###
# Global libraries
import os
from termcolor import colored

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

# Custom libraries
from .utils_functions_bis import download_file

### CLASS DEFINITION ###
class GloveConverter:
    """
        Class to get the GloVe embeddings of words
    """

    def __init__(self):
        """
            Loads the GloVe model
        """
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/glove.gensim"
        ):
            print(
                colored(
                    "Processing the `GloVe` dataset for the first time...", "yellow"
                )
            )

            # Downloads the model if needed
            if not os.path.exists(
                "/nas-data/vilbert/data2/conceptnet/processed/glove.840B.300d.txt"
            ):
                download_file(
                    "http://nlp.stanford.edu/data/glove.840B.300d.zip",
                    "/nas-data/vilbert/data2/conceptnet/raw/glove.840B.300d.txt",
                )

            print(
                colored(
                    "Converting the GloVe dataset to a `word2vec` format...", "yellow",
                )
            )

            _ = glove2word2vec(
                "/nas-data/vilbert/data2/conceptnet/raw/glove.840B.300d.txt",
                "/nas-data/vilbert/data2/conceptnet/raw/glove_word2vec.txt",
            )

            custom_embedding = KeyedVectors.load_word2vec_format(
                "/nas-data/vilbert/data2/conceptnet/raw/glove_word2vec.txt",
                binary=False,
            )

            # Saves the model
            print(colored("Saving the embedding model...", "yellow"))
            custom_embedding.save(
                "/nas-data/vilbert/data2/conceptnet/processed/glove.gensim"
            )

        # Loads the gensim model
        self.embedding_model = KeyedVectors.load(
            "/nas-data/vilbert/data2/conceptnet/processed/glove.gensim"
        )

    def convert_word_to_embedding(self, word):
        """
            Given a word, returns its GloVe embedding
        """
        try:
            return self.embedding_model[word]
        except Exception as e:
            # print(colored("ERROR: {}".format(e), "red"))
            pass


class NumberbatchConverter:
    """
        Class to get the Numberbatch embeddings of words
    """

    def __init__(self):
        """
            Loads the Numberbatch model
        """
        if not os.path.exists(
            "/nas-data/vilbert/data2/conceptnet/processed/numberbatch_en.gensim"
        ):
            print(
                colored(
                    "Processing the `Numberbatch` dataset for the first time...",
                    "yellow",
                )
            )

            # Downloads the model if needed
            if not os.path.exists(
                "/nas-data/vilbert/data2/conceptnet/raw/numberbatch_en.txt"
            ):
                download_file(
                    "https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz",
                    "/nas-data/vilbert/data2/conceptnet/raw/numberbatch_en.txt",
                )

            custom_embedding = KeyedVectors.load_word2vec_format(
                "/nas-data/vilbert/data2/conceptnet/raw/numberbatch_en.txt",
                binary=False,
            )

            # Saves the model
            print(colored("Saving the embedding model...", "yellow"))
            custom_embedding.save(
                "/nas-data/vilbert/data2/conceptnet/processed/numberbatch_en.gensim"
            )

        # Loads the gensim model
        self.embedding_model = KeyedVectors.load(
            "/nas-data/vilbert/data2/conceptnet/processed/numberbatch_en.gensim"
        )

    def convert_word_to_embedding(self, word):
        """
            Given a word, returns its GloVe embedding
        """
        try:
            return self.embedding_model[word]
        except Exception as e:
            # print(colored("ERROR: {}".format(e), "red"))
            pass

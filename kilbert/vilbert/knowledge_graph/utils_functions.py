### LIBRARIES ###
# Global libraries
import os
import requests
from tqdm import tqdm
from termcolor import colored

import io
import zipfile
import gzip
import shutil

### FUNCTION DEFINITIONS ###


def download_file(source_url, dest_path, source_path=""):
    """
        Downloads the given archive and extracts it
        Currently works for: 
            - `zip` files
            - `tar.gz` files

        Inputs: 
            - source_url (str): URL to download the ZIP file
            - source_path (str): Path of the file in the ZIP file
            - dest_path (str): Path of the extracted file 
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
        with open("/nas-data/vilbert/data2/conceptnet/raw/temp.zip", "wb") as f:
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

        z = zipfile.ZipFile("/nas-data/vilbert/data2/conceptnet/raw/temp.zip")
        # Extract the file from the temporary file
        if source_path != "":
            z.extract(source_path, os.path.dirname(dest_path))
            os.rename(os.path.join(os.path.dirname(dest_path), source_path), dest_path)
        else:
            z.extractall(os.path.dirname(dest_path))
            # z.extractall(dest_path.split(os.path.sep)[:-1])

        # Remove the temporary file
        os.remove("/nas-data/vilbert/data2/conceptnet/raw/temp.zip")

    elif file_extension == "gz":
        # Save the GZ file in a temporary GZ file
        with open(
            "/nas-data/vilbert/data2/conceptnet/raw/temp.zip", "wb"
        ) as temp_file:
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

        with gzip.open(
            "/nas-data/vilbert/data2/conceptnet/raw/temp.zip", "rb"
        ) as file_in:
            with open(dest_path, "wb") as file_out:
                shutil.copyfileobj(file_in, file_out)

        # Remove the temporary file
        os.remove("/nas-data/vilbert/data2/conceptnet/raw/temp.zip")



# 1.0.0 (2021-03-07)
## Updated
* Documentation (README.md):
* Update `vlbert_tasks.yml` to reset the value to the Kubernetes one. An example of modifications is added on the README.md
* Update Validation and Evaluation code (missing import)
* Update the setup.py
* Update the project for the packaging

## Added
* MANIFEST.in to add missing files in the packaging (some of them have specific extensions).
* missing `__init__.py` everywhere in the project

# 0.2.1 (2021-02-18)
## Updated
* Documentation (README.md):
    * specify the minimum requirement to run the project
    * update the training commands
    * add information about training parameters
    * update Docker documentation
* Update `vlbert_tasks.yml` to run the project on the GPU Server (4x `GeForce RTX 2080 Ti`)
* Add trace to debug the GPU usage

# 0.2.0 (2021-01-22)
## Added
* change multiple import path from multiple files (FCnet, TCnet, etc..)
* update the documentation to add the tools compilation
* update the documentation of the Docker and Local run


# 0.1.1 (2021-01-21)
## Added
* working pipenv environment
* update the documentation
* add Docker documentation to start the training


## Added
* add version file
* add CHANGELOG.md file

## Updated
* update setup file
* README to add training/validation/eval commands

## Removed
* kubernetes template
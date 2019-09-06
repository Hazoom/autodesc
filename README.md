# Have you ever dreamed about an AI that could understand a code snippet and tells you what it does?
**Datahack 2019 Project**

This is an auto code description generation and code search engine project.

It's based on StackOverFlow dataset, focusing on Data Science and Data Structures fields only.


# Directory Structure
* `src`
    Main source code directory.
* `rsources`
   External resources necessary for running this project, like the BERT's vocabulary `txt` file.
* `scripts`
   Helper scripts such that:
   * SQL query for fetching the StackOverFlow data from Google's BigQuery service.
   * `shell` scripts for running each step in the process.


# Prerequisites
1. Make sure to have `Python 3.6`
2. Install `pipenv` by `pip install pipenv`
3. In your terminal, create a new virtual environment inside a new shell, using the command `pipenv shell` (make sure to run all commands inside this shell to not affect your global environment settings).
    This should create a `.venv` folder inside the project's root folder.
4. Install all the requirements using the `Pipfile` and `Pipfile.lock` files by running the following command: `pipenv sync`.
    **Note**: If using a GPU machine (recommended) one needs to change `tensorflow` to `tensorflow-gpu`,


# Getting Started
1. Fetch the data from Google's [BigQuery](https://github.com/hamelsmu/) service using the script `scripts/bigquery_stackoverflow.sql`.
They supply free trail of 300$ which is more than enough for this task.
2. Clone Google's [BERT](https://github.com/google-research/bert) code into `src/bertcode/bert` (the folder is in `.gitignore`).
3. Download BERT's base uncased model for English into `bert/models/uncased_L-12_H-768_A-12` (the folder is in `.gitignore`)

# Running Examples
Please follow the scripts in `resources` folder for all running examples.

**Acknowledgements**
* [Amenity Analytics](https://www.amenityanalytics.com/) for the credit and resources. Thanks!
* Main idea derived from [hamelsmu](https://github.com/hamelsmu/code_search) with some modifications to fit to the problem of generating comment from code, mainly in the data, pre-processing, cleaning and sentence embedding mechanism.
# You know, for searching code..
**Datahack 2019 Project**

This is a semantic code search engine and auto code comment generation.

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
3. Install all the requirements using the `Pipfile` and `Pipfile.lock` files by running the following command: `pipenv sync`.
   **Note**: If using a GPU machine (recommended) one needs to change `tensorflow` to `tensorflow-gpu`,

# Getting Started
1. Fetch the data from Google's [BigQuery](https://github.com/hamelsmu/) service using the script `scripts/bigquery_stackoverflow.sql`.
They supply free trail of 300$ which is more than enough for this task.
2. Clone Google's [BERT](https://github.com/google-research/bert) code into `src/bertcode/bert` (the folder is in `.gitignore`).
3. Download BERT's base uncased model for English into `bert/models/uncased_L-12_H-768_A-12` (the folder is in `.gitignore`)

# Running Examples
Please follow the scripts in `resources` folder for all running examples.

**Acknowledgements**

Some ideas derived from [hamelsmu](https://github.com/hamelsmu/).
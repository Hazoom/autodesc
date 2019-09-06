import os
import pickle
import argparse
import argcomplete
import numpy as np
import pandas as pd
import nmslib

from bertcode import bertvectors


def create_nmslib_search_index(numpy_vectors):
    """
    Copyright (c) 2018 Hamel Husain
    :param numpy_vectors:
    :return:
    """
    search_index = nmslib.init(method='hnsw', space='cosinesimil')
    search_index.addDataPointBatch(numpy_vectors)
    search_index.createIndex({'post': 2}, print_progress=True)
    return search_index


def train_search_engine(titles_bert_vectors_file: str,
                        train_file: str,
                        output_dir: str):
    vectors_map = bertvectors.load_title_vectors(titles_bert_vectors_file)
    train_df = pd.read_csv(train_file)
    index_to_id = {i: index for i, index in enumerate(train_df['index'].to_list())}
    vectors = np.array([value for _, value in vectors_map.items()])
    dim768_tfidf_search_index = create_nmslib_search_index(vectors)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    with open(os.path.join(output_dir, 'index_to_id.pkl'), 'wb') as out_fp:
        pickle.dump(index_to_id, out_fp)

    dim768_tfidf_search_index.saveIndex(os.path.join(output_dir, 'dim768_tfidf_searchindex.nmslib'))


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--titles-bert-vectors-file", type=str,
                                 help='Title BERT vectors file', required=True)
    argument_parser.add_argument("--train-file", type=str,
                                 help='Train CSV file', required=True)
    argument_parser.add_argument("--output-dir", type=str,
                                 help='Output dir', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    train_search_engine(args.titles_bert_vectors_file,
                        args.train_file,
                        args.output_dir)


if __name__ == '__main__':
    main()

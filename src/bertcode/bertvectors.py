import os
import argparse
import argcomplete
import pandas as pd
from typing import List
import pickle

from bert_serving.client import BertClient

from scores.attentivepooling import AttentivePooler
from preprocessing.cleaning import clean_title


def load_title_vectors(file_path: str):
    with open(file_path, 'rb') as in_fp:
        return pickle.load(in_fp)


class BertVectorsService:
    def __init__(self,
                 score_model_path: str,
                 ip: str = '3.248.147.179',
                 port: int = 5555,
                 port_out: int = 5556):
        self.ip = ip
        self.port = port
        self.port_out = port_out
        self.pooller = AttentivePooler(score_model_path)

    def get_vectors(self,
                    sentence_list: List[str]):
        bc = BertClient(self.ip, self.port, self.port_out, show_server_config=False, check_length=False)
        cleaned_sentences = [clean_title(sentence) for sentence in sentence_list]
        vectors = bc.encode(cleaned_sentences)
        return self.pooller.pool(vectors, cleaned_sentences).tolist()


def test():
    bert_service = BertVectorsService('data/score_model/score_model.pkl')
    vectors_demo = bert_service.get_vectors(
        sentence_list=['pandas dataframe merge two columns and merge two columns']
    )
    print(vectors_demo[0])


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def create_vectors(titles_file: str,
                   output_file: str,
                   score_model_path: str):
    bert_service = BertVectorsService(score_model_path)
    titles_df = pd.read_csv(titles_file)
    indexes = titles_df['index'].to_list()
    titles = titles_df['title'].to_list()
    index_and_titles_list = [(index, title) for index, title in zip(indexes, titles)]
    results = {}
    finished_index = 0
    index_and_titles_chunks = list(chunks(index_and_titles_list, 32))
    print('Split the {} rows into {} chunks'.format(
        str(len(index_and_titles_list)), str(len(index_and_titles_chunks)))
    )
    for chunk in index_and_titles_chunks:
        chunk_titles = [item[1] for item in chunk]
        chunk_indexes = [item[0] for item in chunk]
        vectors = bert_service.get_vectors(chunk_titles)
        for index, vector in zip(chunk_indexes, vectors):
            results[index] = vector

        finished_index += 1
        if finished_index % 10 == 0 and finished_index > 0:
            print('Finished {} out of {} chunks'.format(str(finished_index), str(len(index_and_titles_chunks))))
    print('Finished {} out of {}'.format(str(finished_index), str(len(index_and_titles_chunks))))

    dir_path = os.path.dirname(output_file)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    with open(output_file, 'wb') as out_fp:
        pickle.dump(results, out_fp)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--titles-file", type=str, help='CSV file', required=True)
    argument_parser.add_argument("--output-file", type=str, help='Output file path', required=True)
    argument_parser.add_argument("--score-model", type=str, help='Score model path', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    create_vectors(args.titles_file, args.output_file, args.score_model)


if __name__ == "__main__":
    main()

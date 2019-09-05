import pickle

import nmslib
import pandas as pd

from bertcode import bertvectors


class TitleSearchEngine:
    def __init__(self,
                 nmslib_index: str = 'data/bert_vectors/dim768_tfidf_searchindex.nmslib',
                 index_to_id_file: str = 'data/bert_vectors/index_to_id.pkl',
                 score_model: str = 'data/score_model/score_model.pkl',
                 train_csv: str = 'data/processed/train.csv'):

        print(f'Loading nmslib index {nmslib_index}')
        self.search_index = nmslib.init(method='hnsw', space='cosinesimil')
        self.search_index.loadIndex(nmslib_index)
        print(f'Finished loading nmslib index {nmslib_index}')

        with open(index_to_id_file, 'rb') as in_fp:
            self.index_to_id = pickle.load(in_fp)
        self.bert_vectors_service = bertvectors.BertVectorsService(score_model_path=score_model)
        titles_df = pd.read_csv(train_csv)
        indexes = titles_df['index'].to_list()
        titles = titles_df['title'].to_list()
        self.id_to_tile = {index: title for index, title in zip(indexes, titles)}

    def search(self,
               title: str,
               top_k: int = 3):
        query = self.bert_vectors_service.get_vectors([title])
        idxs, dists = self.search_index.knnQuery(query, k=top_k)

        print(f'\nSample: {title}')
        for idx, dist in zip(idxs, dists):
            print(f'Cosine similarity:{1.0 - dist:.4f}\n---------------\n', self.id_to_tile[self.index_to_id[idx]])


def test():
    title_search_engine = TitleSearchEngine()
    title_search_engine.search('load csv or json into pandas dataframe')
    title_search_engine.search('sort pandas dataframe')
    title_search_engine.search('sort dictionary by values')


if __name__ == '__main__':
    test()

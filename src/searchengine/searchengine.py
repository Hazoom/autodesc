import pickle
from flask import Flask, request, Response
from flask_restplus import Resource, Api
import pandas as pd
import nmslib

from preprocessing import textpreprocess
from bertcode import bertvectors
from sharedvectors.sharedvectors import load_shared_vector_space_model

app = Flask(__name__)
api = Api(app, version='1.0', title='Code Semantic Search Engine (Datahack 2019)')


def main():

    # initialize token to id map
    with open('data/bert_vectors/index_to_id.pkl', 'rb') as in_fp:
        index_to_id = pickle.load(in_fp)

    # initialize shared vector space model
    shared_vector_model = \
        load_shared_vector_space_model('data/models/shared_space/shared_vector_space_model_weights_best.h5')

    # initialize bert vector service
    bert_vectors_service = bertvectors.BertVectorsService(score_model_path='data/score_model/score_model.pkl')

    # initialize id to title and code map
    titles_df = pd.read_csv('data/processed/train.csv')
    indexes = titles_df['index'].to_list()
    titles = titles_df['title'].astype(str).to_list()
    codes = titles_df['answer_code'].astype(str).to_list()
    titles_and_codes = [(title, code) for title, code in zip(titles, codes)]
    id_to_title_and_code = {index: {'title': title_and_code[0], 'code': title_and_code[1]}
                            for index, title_and_code in zip(indexes, titles_and_codes)}
    print('Loaded {} pairs of title and code'.format(str(len(id_to_title_and_code))))

    # initialize code pre processor
    code_pre_processor, _ = textpreprocess.load_text_preprocessor('data/vectors/code_pre_processor.dpkl')

    # initialize nmslib title search index
    title_nmslib_index = 'data/vectors/predicted_titles/dim768_predicted_titles_index.nmslib'
    print(f'Loading nmslib index {title_nmslib_index}')
    title_search_index = nmslib.init(method='hnsw', space='cosinesimil')
    title_search_index.loadIndex(title_nmslib_index)
    print(f'Finished loading nmslib index {title_nmslib_index}')

    # set all necessary resources
    app.index_to_id = index_to_id
    app.bert_vectors_service = bert_vectors_service
    app.id_to_title_and_code = id_to_title_and_code
    app.code_pre_processor = code_pre_processor
    app.shared_vector_model = shared_vector_model
    app.title_search_index = title_search_index

    app.run(host='0.0.0.0', debug=False)
    app.logger.disabled = False


if __name__ == "__main__":
    main()

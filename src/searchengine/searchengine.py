import pickle
from datetime import datetime
from flask import Flask, request, jsonify
from flask_restplus import Resource, Api
import pandas as pd
import nmslib
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences

from preprocessing import textpreprocess
from bertcode import bertvectors
from sharedvectors.sharedvectors import load_shared_vector_space_model

app = Flask(__name__)
api = Api(app, version='1.0', title='Code Semantic Search Engine (Datahack 2019)')

global graph
graph = tf.get_default_graph()


def _get_param_or_default(args, param_name, default):
    return args.get(param_name) if args.get(param_name) else default


def _get_elapsed_time_in_ms(start_time):
    return int((datetime.now() - start_time).total_seconds() * 1000)


@api.route('/datahack2019/code')
class ClustersSearchAPI(Resource):
    @api.doc(params={'query': 'A query text',
                     'k': 'Top k. Default: 3'},
             responses={200: 'Success'},
             description='Generated top k best code snippets for given text')
    def get(self):
        start_time = datetime.now()
        query = request.args.get("query")
        top_k = int(_get_param_or_default(request.args, "k", '3'))
        print(f'Call to /code api with query: "{query}" and k: "{top_k}"')

        query_vector = app.bert_vectors_service.get_vectors([query])[0]
        indexes, dists = app.title_search_index.knnQuery(query_vector, k=top_k)
        codes = []
        for idx, dist in zip(indexes, dists):
            codes.append({'code': app.id_to_title_and_code[app.index_to_id[idx]]['code'],
                          'similarity': 1.0 - dist})
        results = {'codes': codes}

        print('Total process time {} milliseconds'.format(_get_elapsed_time_in_ms(start_time)))
        return jsonify(results)


@api.route('/datahack2019/comment')
class ClustersSearchAPI(Resource):
    @api.doc(params={'code': 'Python code',
                     'k': 'Top k. Default: 3'},
             responses={200: 'Success'},
             description='Generated top k best comments for a given Python code')
    def get(self):
        start_time = datetime.now()
        code = request.args.get("code")
        top_k = int(_get_param_or_default(request.args, "k", '3'))
        print(f'Call to /comment api with query: "{code}" and k: "{top_k}"')

        query_preprocessed = app.code_pre_processor.transform([code])

        with graph.as_default():
            query_vectors = app.shared_vector_model.predict(query_preprocessed, batch_size=1)

        query_vector = query_vectors[0]
        indexes, dists = app.title_search_index.knnQuery(query_vector, k=top_k)
        comments = []
        for idx, dist in zip(indexes, dists):
            comments.append({'comment': app.id_to_title_and_code[app.index_to_id[idx]]['title'],
                             'similarity': 1.0 - dist})
        results = {'comments': comments}

        print('Total process time {} milliseconds'.format(_get_elapsed_time_in_ms(start_time)))
        return jsonify(results)


def _get_id_to_title_and_code():
    titles_df = pd.read_csv('data/processed/train.csv')
    indexes = titles_df['index'].to_list()
    titles = titles_df['raw_title'].astype(str).to_list()
    codes = titles_df['answer_code_raw'].astype(str).to_list()
    titles_and_codes = [(title, code) for title, code in zip(titles, codes)]
    id_to_title_and_code_map = {index: {'title': title_and_code[0], 'code': title_and_code[1]}
                                for index, title_and_code in zip(indexes, titles_and_codes)}
    print('Loaded {} pairs of title and code'.format(str(len(id_to_title_and_code_map))))
    return id_to_title_and_code_map


def _get_title_nmslib_index():
    title_nmslib_index_name = 'data/vectors/predicted_titles/dim768_predicted_titles_index.nmslib'
    print(f'Loading nmslib index {title_nmslib_index_name}')
    title_search_index_nmslib = nmslib.init(method='hnsw', space='cosinesimil')
    title_search_index_nmslib.loadIndex(title_nmslib_index_name)
    print(f'Finished loading nmslib index {title_nmslib_index_name}')
    return title_search_index_nmslib


if __name__ == "__main__":
    # initialize token to id map
    with open('data/bert_vectors/index_to_id.pkl', 'rb') as in_fp:
        index_to_id = pickle.load(in_fp)

    # initialize shared vector space model
    shared_vector_model = \
        load_shared_vector_space_model('data/models/shared_space/shared_vector_space_model_best.h5')
    shared_vector_model.summary()

    # initialize bert vector service
    bert_vectors_service = bertvectors.BertVectorsService(score_model_path='data/score_model/score_model.pkl')

    # initialize id to title and code map
    id_to_title_and_code = _get_id_to_title_and_code()

    # initialize code pre processor
    code_pre_processor, _ = textpreprocess.load_text_preprocessor('data/vectors/code_pre_processor.dpkl')

    # initialize nmslib title search index
    title_search_index = _get_title_nmslib_index()

    # set all necessary resources
    app.index_to_id = index_to_id
    app.bert_vectors_service = bert_vectors_service
    app.id_to_title_and_code = id_to_title_and_code
    app.code_pre_processor = code_pre_processor
    app.shared_vector_model = shared_vector_model
    app.title_search_index = title_search_index

    app.run(host='0.0.0.0', debug=False)
    app.logger.disabled = False

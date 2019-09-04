import argparse
import os
import pickle
from math import log
from typing import List
import argcomplete

import numpy as np

from bertcode.tokenization import FullTokenizer


def pad_scores(sentence, max_len):
    if len(sentence) > max_len:
        sentence = sentence[0:max_len]

    pad_len = max_len - len(sentence)
    sentence += [('[PAD]', 0.0)] * pad_len

    return sentence


def normalize(x):
    norm = np.sqrt(np.sum(x ** 2))
    return x / norm


def simple_normalization(x_data, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    x_data: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis: int (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(x_data)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(x_data.shape) == 1:
        p = p.flatten()

    return p


def get_score_for_token(token, token_count, score_model):
    return token_count * (log(score_model['sentence_count'] / score_model['tokens'][token]['sentence_count']) + 1.0)


class SentenceWeightModel:
    def __init__(self):
        self.tokenizer = FullTokenizer(vocab_file='resources/vocab.txt')
        self.tokenize = self.tokenizer.tokenize
        self.score_model = None
        self.unknown_word_score = 0.1

    def load_model(self, model_name):
        print(f'Loading pre-trained COUNT scores model with name: {model_name}')

        if not os.path.exists(model_name):
            raise Exception(f'Model {model_name} not found!')

        with open(model_name, 'rb') as input_file_pointer:
            self.score_model = pickle.load(input_file_pointer)

    def train(self,
              titles_file: str,
              output_model_name: str):
        print('Calculating scores for each token...')

        model = {'tokens': {}}
        sentence_count = 0
        with open(titles_file, 'r') as corpus_fp:
            lines = corpus_fp.readlines()

        for index, sentence in enumerate(lines):
            tokens_added = set()
            for feature in self.tokenize(sentence):
                if feature in model['tokens']:
                    model['tokens'][feature]['count'] += 1.0
                else:
                    model['tokens'][feature] = {'count': 1.0, 'sentence_count': 1.0}
                    tokens_added.add(feature)
                if feature not in tokens_added:
                    model['tokens'][feature]['sentence_count'] += 1
                    tokens_added.add(feature)

            sentence_count += 1

        print('Finished {} sentences out of {}'.format(str(sentence_count), str(sentence_count)))

        model['sentence_count'] = float(sentence_count)

        self.score_model = model

        print('Finished calculating scores for each token...')

        print(f'Saving scores model with name: {output_model_name}')

        model_path = os.path.dirname(output_model_name)
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        pickle.dump(self.score_model, open(output_model_name, "wb"))

    def sentences_to_score(self,
                           sentences: List[str],
                           max_len: int = None,
                           normalize_scores: bool = True):
        if self.score_model is None:
            raise Exception('Model was not trained or not loaded!')

        sentences = [self.tokenize(sentence) for sentence in sentences]

        cls_tuple = ('[CLS]', 0.0)
        sep_tuple = ('[SEP]', 0.0)
        results = []
        for i, sentence in enumerate(sentences):
            tokens_to_count = {}
            for token in sentence:
                if token in tokens_to_count:
                    tokens_to_count[token] += 1
                else:
                    tokens_to_count[token] = 1

            tokens = [token for token, _ in tokens_to_count.items()]
            scores = [get_score_for_token(token, token_count, self.score_model) if token in self.score_model[
                'tokens'] else self.unknown_word_score
                      for token, token_count in tokens_to_count.items()]
            if normalize_scores:
                tokens_and_scores_normalized = [(token, normalized_score) for token, normalized_score in
                                                zip(tokens, normalize(np.array(scores)))]

                results.append([cls_tuple] + tokens_and_scores_normalized + [sep_tuple])
            else:
                results.append([cls_tuple] + [(token, score) for token, score in zip(tokens, scores)] + [sep_tuple])

        if max_len is None:
            return results
        else:
            padded_sentences = [pad_scores(sentence, max_len) for sentence in results]
            return padded_sentences


def train(corpus, output_model_path, perform_train):
    score_model = SentenceWeightModel()

    if perform_train:
        score_model.train(corpus, output_model_path)
    else:
        score_model.load_model(output_model_path)

    sentences = ['pandas dataframe merge two columns and merge two columns']
    tokens_and_scores = score_model.sentences_to_score(sentences)[0]
    print("\nScores and Tokens:")
    print(tokens_and_scores)
    scores = np.array([score[1] for score in tokens_and_scores])
    print("\nAfter Normalization:")
    print(simple_normalization(scores))


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('--corpus', type=str, help='Corpus txt file', required=True)
    argument_parser.add_argument('--output', type=str, help='Output model path', required=True)
    argument_parser.add_argument('--train', help='Perform training. Default: False',
                                 action='store_true', default=False)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    train(args.corpus, args.output, args.train)

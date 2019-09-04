from typing import List

import numpy as np

from scores.scoreutils import simple_normalization
from scores import weightsmodel


def _attentive_pool_with_weights(encoded, weights):
    weights = simple_normalization(np.expand_dims(weights, axis=-1), axis=1)  # [B,L,1]
    weight_encoded = weights * encoded  # [B,L,D]
    pooled = np.sum(weight_encoded, axis=1)  # [B, D]
    return pooled


class AttentivePooler:
    def __init__(self,
                 score_model_path: str):
        self.sentence_weights_model = weightsmodel.SentenceWeightModel()
        self.sentence_weights_model.load_model(score_model_path)

    def get_weights(self,
                    sentences: List[str],
                    seq_length: int = 64):
        tokens_and_weights = self.sentence_weights_model.sentences_to_score(sentences,
                                                                            max_len=seq_length)
        weights = np.array([[weight for _, weight in sentence] for sentence in tokens_and_weights], dtype=np.float64)
        return weights

    def pool(self,
             matrix_list,
             sentences: List[str],
             seq_length: int = 64):
        weights = self.get_weights(sentences, seq_length)
        vectors = _attentive_pool_with_weights(matrix_list, weights)
        return vectors


if __name__ == '__main__':
    attentive_pooler = AttentivePooler('data/score_model/score_model.pkl')
    weights_sample = attentive_pooler.get_weights(['pandas dataframe merge two columns and merge two columns'],
                                                  seq_length=64)
    print(weights_sample)

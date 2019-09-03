import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Embedding, Activation, Permute
from keras.layers import Input, Flatten, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional

from seq2seq.layers.custum_recurrents import AttentionDecoder


def simpleNMT(pad_length=100,
              n_chars=105,
              n_labels=6,
              embedding_learnable=False,
              encoder_units=256,
              decoder_units=256,
              trainable=True,
              return_probabilities=False):
    """
    Builds a Neural Machine Translator that has alignment attention
    :param pad_length: the size of the input sequence
    :param n_chars: the number of characters in the vocabulary
    :param n_labels: the number of possible labelings for each character
    :param embedding_learnable: decides if the one hot embedding should be refinable.
    :return: keras.models.Model that can be compiled and fit'ed

    *** REFERENCES ***
    Lee, Jason, Kyunghyun Cho, and Thomas Hofmann.
    "Neural Machine Translation By Jointly Learning To Align and Translate"
    """
    input_ = Input(shape=(pad_length,), dtype='float32')
    input_embed = Embedding(n_chars, n_chars,
                            input_length=pad_length,
                            trainable=embedding_learnable,
                            weights=[np.eye(n_chars)],
                            name='OneHot')(input_)

    rnn_encoded, enc_forward_h, enc_forward_c, enc_backward_h, enc_backward_c = Bidirectional(
        LSTM(encoder_units, return_sequences=True, return_state=True),
        name='bidirectional_1',
        merge_mode='concat',
        trainable=trainable)(input_embed)

    # Encapsulate the encoder as a separate entity so we can just encode without decoding if we want to
    # encoder_model = Model(inputs=input_, outputs=K.concatenate([enc_forward_h + enc_backward_h]), name='Encoder-Model')
    encoder_model = Model(inputs=input_, outputs=enc_forward_h, name='Encoder-Model')
    encoder_model.summary()

    seq2seq_encoder_out = encoder_model(input_)

    y_hat = AttentionDecoder(decoder_units,
                             name='attention_decoder_1',
                             output_dim=n_labels,
                             return_probabilities=return_probabilities,
                             trainable=trainable)(rnn_encoded)

    model = Model(inputs=input_, outputs=y_hat)

    return model


if __name__ == '__main__':
    model_nmt = simpleNMT()
    model_nmt.summary()

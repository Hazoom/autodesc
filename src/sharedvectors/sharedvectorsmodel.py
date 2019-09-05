import os
import argparse
import argcomplete
import numpy as np
import pickle
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Input, GRU, Dense, Embedding, BatchNormalization
from keras.models import Model

from bertcode import bertvectors
from seq2seq.model import load_encoder_inputs, extract_encoder_model
from seq2seq.modelevaluation import load_model
from preprocessing import textpreprocess


def build_model(word_emb_dim: int,
                hidden_state_dim: int,
                encoder_seq_len: int,
                n_encoder_tokens: int,
                n_decoder_tokens: int,
                learning_rate: float = 0.00005):
    # Encoder Model
    encoder_inputs = Input(shape=(encoder_seq_len,), name='Encoder-Input')

    # Word embedding for encoder (i.e. code)
    encoder_embeddings = Embedding(n_encoder_tokens, word_emb_dim,
                                   name='Code-Embedding', mask_zero=False)(encoder_inputs)
    encoder_bn = BatchNormalization(name='Encoder-Batchnorm-1')(encoder_embeddings)

    # We do not need the `encoder_output` just the hidden state.
    encoder_out, encoder_state = GRU(hidden_state_dim, return_sequences=True, return_state=True,
                                     name='Encoder-Last-GRU', dropout=.5, recurrent_dropout=.4)(encoder_bn)

    # Encapsulate the encoder as a separate entity so we can just encode without decoding if we want to
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_state, name='Encoder-Model')

    seq2seq_encoder_out = encoder_model(encoder_inputs)

    # Decoder Model
    decoder_inputs = Input(shape=(None,), name='Decoder-Input')  # for teacher forcing

    # Word Embedding For Decoder (i.e. titles)
    dec_emb = Embedding(n_decoder_tokens, word_emb_dim, name='Title-Embedding', mask_zero=False)(decoder_inputs)
    dec_bn = BatchNormalization(name='Decoder-Batchnorm-1')(dec_emb)

    # Set up the decoder, using `decoder_state_input` as initial state.
    decoder_gru = GRU(hidden_state_dim, return_state=True, return_sequences=True, name='Decoder-GRU',
                      dropout=.5, recurrent_dropout=.4)
    decoder_gru_output, _ = decoder_gru(dec_bn, initial_state=seq2seq_encoder_out)
    decoder_bn = BatchNormalization(name='Decoder-Batchnorm-2')(decoder_gru_output)

    # # add attention layer
    # attn_layer = AttentionLayer(name='attention_layer')
    # attn_out, attn_states = attn_layer([encoder_out, decoder_gru_output])
    #
    # # concatenate the attn_out and decoder_out as an input to the softmax layer
    # decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_gru_output, attn_out])

    # Define TimeDistributed softmax layer
    decoder_dense = Dense(n_decoder_tokens, activation='softmax', name='Final-Output-Dense')
    decoder_outputs = decoder_dense(decoder_bn)

    seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    seq2seq_model.compile(optimizer=optimizers.Nadam(lr=learning_rate),
                          loss='sparse_categorical_crossentropy')

    return seq2seq_model


def train_shared_vector_space_model(train_code_vectors_file: str,
                                    train_code_preprocessor_file: str,
                                    train_title_preprocessor_file: str,
                                    titles_bert_vectors_file: str,
                                    model_weights_file: str,
                                    output_dir: str,
                                    epochs: int,
                                    batch_size: int,
                                    validation_split: float,
                                    learning_rate: float):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    _, n_decoder_tokens = textpreprocess.load_text_preprocessor(train_title_preprocessor_file)
    _, n_encoder_tokens = textpreprocess.load_text_preprocessor(train_code_preprocessor_file)

    encoder_vectors = load_encoder_inputs(train_code_vectors_file)

    vectors_map = bertvectors.load_title_vectors(titles_bert_vectors_file)
    title_vectors = np.array([value for _, value in vectors_map.items()])

    assert len(title_vectors) == len(encoder_vectors)

    seq2seq_model = load_model(encoder_vectors.shape[1],
                               n_encoder_tokens,
                               n_decoder_tokens,
                               model_weights_file)

    encoder_model = extract_encoder_model(seq2seq_model)
    encoder_model.summary()


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--code-vectors-file", type=str,
                                 help='Train code vectors file path', required=True)
    argument_parser.add_argument("--code-preprocessor-file", type=str,
                                 help='Train code pre processor file path', required=True)
    argument_parser.add_argument("--title-preprocessor-file", type=str,
                                 help='Train title pre processor file path', required=True)
    argument_parser.add_argument("--titles-bert-vectors-file", type=str,
                                 help='Title BERT vectors file', required=True)
    argument_parser.add_argument("--model-weights-file", type=str,
                                 help='Model weights file path', required=True)
    argument_parser.add_argument("--output-dir", type=str, help='Output directory for model', required=True)
    argument_parser.add_argument("--epochs", type=int, help='Number of epochs. Default: 16', required=False,
                                 default=16)
    argument_parser.add_argument("--batch-size", type=int, help='Batch size. Default: 32', required=False,
                                 default=32)
    argument_parser.add_argument("--validation-split", type=float, help='Validation size. Default: 0.1', required=False,
                                 default=0.1)
    argument_parser.add_argument("--learning-rate", type=float, help='Learning rate. Default: 0.00005',
                                 required=False, default=0.00005)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    train_shared_vector_space_model(args.code_vectors_file,
                                    args.code_preprocessor_file,
                                    args.title_preprocessor_file,
                                    args.titles_bert_vectors_file,
                                    args.model_weights_file,
                                    args.output_dir,
                                    args.epochs,
                                    args.batch_size,
                                    args.validation_split,
                                    args.learning_rate)


if __name__ == '__main__':
    main()

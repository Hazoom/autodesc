"""
MIT License

Copyright (c) 2018 Hamel Husain

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import os
import argcomplete
import numpy as np
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Input, GRU, Dense, Embedding, BatchNormalization, TimeDistributed, Concatenate
from keras.models import Model

from preprocessing import textpreprocess
from seq2seq.layers.attention import AttentionLayer


def _load_encoder_inputs(file_path):
    code_vectors = np.load(file_path)
    print(f'Shape of encoder vectors: {code_vectors.shape}')
    return code_vectors


def _load_decoder_inputs(file_path):
    title_vectors = np.load(file_path)
    decoder_input_vectors = title_vectors[:, :-1]
    decoder_target_vectors = title_vectors[:, 1:]

    print(f'Shape of decoder input: {decoder_input_vectors.shape}')
    print(f'Shape of decoder target: {decoder_target_vectors.shape}')
    return decoder_input_vectors, decoder_target_vectors


def train(train_code_vectors_file: str,
          train_code_preprocessor_file: str,
          train_title_vectors_file: str,
          train_title_preprocessor_file: str,
          output_dir: str,
          epochs: int,
          word_embedding_dim: int = 300,
          hidden_state_dim: int = 768):
    # Load vectors and title/code pre processors
    decoder_input_vectors, decoder_target_vectors = _load_decoder_inputs(train_code_vectors_file)
    encoder_vectors = _load_encoder_inputs(train_title_vectors_file)
    title_pre_processor, n_decoder_tokens = textpreprocess.load_text_preprocessor(train_title_preprocessor_file)
    code_pre_processor, n_encoder_tokens = textpreprocess.load_text_preprocessor(train_code_preprocessor_file)

    model = build_model(word_embedding_dim,
                        hidden_state_dim,
                        encoder_vectors.shape[1],
                        n_decoder_tokens,
                        n_encoder_tokens)

    model.summary()

    model.compile(optimizer=optimizers.Nadam(lr=0.00005),
                  loss='sparse_categorical_crossentropy')

    csv_logger = CSVLogger('model.log')

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'model.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'),
        save_best_only=True)

    batch_size = 32
    model.fit([encoder_vectors, decoder_input_vectors], np.expand_dims(decoder_target_vectors, -1),
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.07,
              callbacks=[csv_logger, model_checkpoint])

    return model


def build_model(word_emb_dim,
                hidden_state_dim,
                encoder_seq_len,
                n_encoder_tokens,
                n_decoder_tokens):
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

    # Define TimeDistributed Softmax layer and provide decoder_concat_input as the input
    decoder_dense = Dense(n_decoder_tokens, activation='softmax', name='Final-Output-Dense')
    dense_time = TimeDistributed(decoder_dense, name='time_distributed_layer')
    decoder_outputs = dense_time(decoder_bn)

    seq2seq_model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    return seq2seq_model


def train_seq2seq(code_vectors_file: str,
                  code_pre_processor_file: str,
                  title_vectors_file: str,
                  title_pre_processor_file: str,
                  output_dir: str,
                  epochs: int):

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = train(code_vectors_file, code_pre_processor_file,
                  title_vectors_file, title_pre_processor_file,
                  output_dir, epochs)

    # save model
    model.save(os.path.join(output_dir, 'code_title_seq2seq_model.h5'))


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--code-vectors-file", type=str, help='Train code vectors file path', required=True)
    argument_parser.add_argument("--code-preprocessor-file", type=str,
                                 help='Train code pre processor file path', required=True)
    argument_parser.add_argument("--title-vectors-file", type=str, help='Train title vectors file path', required=True)
    argument_parser.add_argument("--title-preprocessor-file", type=str,
                                 help='Train title pre processor file path', required=True)
    argument_parser.add_argument("--output-dir", type=str, help='Output directory for model', required=True)
    argument_parser.add_argument("--epochs", type=int, help='Number of epochs. Default: 16', required=False,
                                 default=16)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    train_seq2seq(args.code_vectors_file, args.code_preprocessor_file,
                  args.title_vectors_file, args.title_preprocessor_file,
                  args.output_dir, args.epochs)


if __name__ == '__main__':
    main()

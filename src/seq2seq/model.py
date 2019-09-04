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


# fix random seed for reproducibility
seed = 42
np.random.seed(seed)


def load_encoder_inputs(file_path: str):
    code_vectors = np.load(file_path)
    print(f'Shape of encoder vectors: {code_vectors.shape}')
    return code_vectors


def load_decoder_inputs(file_path: str):
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
          batch_size: int,
          validation_split: float,
          learning_rate: float,
          word_embedding_dim: int = 300,
          hidden_state_dim: int = 768):
    # Load vectors and title/code pre processors
    decoder_input_vectors, decoder_target_vectors = load_decoder_inputs(train_title_vectors_file)
    encoder_vectors = load_encoder_inputs(train_code_vectors_file)
    title_pre_processor, n_decoder_tokens = textpreprocess.load_text_preprocessor(train_title_preprocessor_file)
    code_pre_processor, n_encoder_tokens = textpreprocess.load_text_preprocessor(train_code_preprocessor_file)

    model = build_model(word_embedding_dim,
                        hidden_state_dim,
                        encoder_vectors.shape[1],
                        n_encoder_tokens,
                        n_decoder_tokens,
                        learning_rate=learning_rate)

    model.summary()

    csv_logger = CSVLogger(os.path.join(output_dir, 'model.log'))

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'model_weights_best.hdf5'),
        save_best_only=True)

    model.fit([encoder_vectors, decoder_input_vectors], np.expand_dims(decoder_target_vectors, -1),
              batch_size=batch_size,
              epochs=epochs,
              validation_split=validation_split,
              callbacks=[csv_logger, model_checkpoint])

    return model


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


def train_seq2seq(code_vectors_file: str,
                  code_pre_processor_file: str,
                  title_vectors_file: str,
                  title_pre_processor_file: str,
                  output_dir: str,
                  epochs: int,
                  batch_size: int,
                  validation_split: float,
                  learning_rate: float):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    model = train(code_vectors_file, code_pre_processor_file,
                  title_vectors_file, title_pre_processor_file,
                  output_dir,
                  epochs, batch_size, validation_split, learning_rate)

    # save model
    model.save(os.path.join(output_dir, 'code_title_seq2seq_model.h5'))


def extract_encoder_model(model):
    encoder_model = model.get_layer('Encoder-Model')
    return encoder_model


def extract_decoder_model(model):
    # the latent dimension is the dimension of the hidden state passed from the encoder to the decoder.
    latent_dim = model.get_layer('Encoder-Model').output_shape[-1]

    # Reconstruct the input into the decoder
    decoder_inputs = model.get_layer('Decoder-Input').input
    dec_emb = model.get_layer('Title-Embedding')(decoder_inputs)
    dec_bn = model.get_layer('Decoder-Batchnorm-1')(dec_emb)

    # Instead of setting the initial state from the encoder and forgetting about it, during inference
    # we are not doing teacher forcing, so we will have to have a feedback loop from predictions back into
    # the GRU, thus we define this input layer for the state so we can add this capability
    gru_inference_state_input = Input(shape=(latent_dim,), name='hidden_state_input')

    # we need to reuse the weights. That's why we are getting this
    # If you inspect the decoder GRU that we created for training, it will take as input
    # 2 tensors -> (1) is the embedding layer output for the teacher forcing
    #                  (which will now be the last step's prediction, and will be <START> on the first time step)
    #              (2) is the state, which we will initialize with the encoder on the first time step, but then
    #                   grab the state after the first prediction and feed that back in again.
    gru_out, gru_state_out = model.get_layer('Decoder-GRU')([dec_bn, gru_inference_state_input])

    # Reconstruct dense layers
    dec_bn2 = model.get_layer('Decoder-Batchnorm-2')(gru_out)
    decoder_dense = model.get_layer('Final-Output-Dense')(dec_bn2)
    decoder_outputs = decoder_dense(dec_bn2)

    decoder_model = Model([decoder_inputs, gru_inference_state_input],
                          [decoder_outputs, gru_state_out])
    return decoder_model


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
    argument_parser.add_argument("--batch-size", type=int, help='Batch size. Default: 32', required=False,
                                 default=32)
    argument_parser.add_argument("--validation-split", type=float, help='Validation size. Default: 0.1', required=False,
                                 default=0.1)
    argument_parser.add_argument("--learning-rate", type=float, help='Learning rate. Default: 0.00005',
                                 required=False, default=0.00005)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    train_seq2seq(args.code_vectors_file, args.code_preprocessor_file,
                  args.title_vectors_file, args.title_preprocessor_file,
                  args.output_dir,
                  args.epochs, args.batch_size, args.validation_split, args.learning_rate)


if __name__ == '__main__':
    main()

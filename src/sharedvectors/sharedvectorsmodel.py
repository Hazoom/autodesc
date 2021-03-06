"""
MIT License

Original work: Copyright (c) 2018 Hamel Husain
Modified work: Copyright (c) 2019 Moshe Hazoom

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
import os
import argparse
import argcomplete
import numpy as np
from keras import optimizers
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Input, Dense, BatchNormalization
from keras.models import Model

from bertcode import bertvectors
from seq2seq.model import load_encoder_inputs, extract_encoder_model
from seq2seq.modelevaluation import load_model
from preprocessing import textpreprocess


def build_shared_vector_model(encoder_model,
                              n_encoder_tokens: int,
                              vector_shape: int,
                              learning_rate: float):
    shared_vector_input = Input(shape=(n_encoder_tokens,), name='shared-vector-input')
    enc_out = encoder_model(shared_vector_input)

    shared_vector_dense = Dense(units=300, activation='relu')(enc_out)
    shared_vector_bn = BatchNormalization(name='Batch-Normalization')(shared_vector_dense)

    # The output of the shared vector space model should be in the same dimension as the title vectors
    shared_vector_out = Dense(units=vector_shape)(shared_vector_bn)
    shared_vector_model = Model(inputs=[shared_vector_input], outputs=shared_vector_out,
                                name='Shared-Vector-Space-Model')
    shared_vector_model.compile(optimizer=optimizers.Nadam(lr=learning_rate), loss='cosine_proximity')
    return shared_vector_model


def _train(shared_vector_model,
           encoder_vectors,
           title_vectors,
           output_dir: str,
           epochs: int,
           batch_size: int,
           validation_split: float,
           initial_epoch: int = None):
    csv_logger = CSVLogger(os.path.join(output_dir, 'shared_vector_space_model.log'))

    model_checkpoint = ModelCheckpoint(
        os.path.join(output_dir, 'shared_vector_space_model_weights_best.hdf5'),
        save_best_only=True)

    if initial_epoch:
        shared_vector_model.fit([encoder_vectors], title_vectors,
                                batch_size=batch_size,
                                epochs=epochs,
                                initial_epoch=initial_epoch,
                                validation_split=validation_split,
                                callbacks=[csv_logger, model_checkpoint])
    else:
        shared_vector_model.fit([encoder_vectors], title_vectors,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_split=validation_split,
                                callbacks=[csv_logger, model_checkpoint])
    return shared_vector_model


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

    # Freeze Encoder Model
    for layer in encoder_model.layers:
        layer.trainable = False
        print(f'Layer: {layer} trainable: {layer.trainable}')

    shared_vector_model = build_shared_vector_model(encoder_model,
                                                    encoder_vectors.shape[1],
                                                    title_vectors.shape[1],
                                                    learning_rate)
    print('\n\n')
    shared_vector_model.summary()

    shared_vector_model = _train(shared_vector_model,
                                 encoder_vectors,
                                 title_vectors,
                                 output_dir,
                                 epochs,
                                 batch_size,
                                 validation_split)

    # UnFreeze Shared Vector Model
    for layer in shared_vector_model.layers:
        layer.trainable = True
        print(f'Layer: {layer} trainable: {layer.trainable}')

    # train it again for several epochs with unfreezed layers to fit on the task
    shared_vector_model = _train(shared_vector_model,
                                 encoder_vectors,
                                 title_vectors,
                                 output_dir,
                                 epochs + 5,
                                 batch_size,
                                 validation_split / 2.0,
                                 initial_epoch=epochs)

    # save model
    shared_vector_model.save(os.path.join(output_dir, 'shared_vector_space_model_best.h5'))


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
                                 default=256)
    argument_parser.add_argument("--validation-split", type=float, help='Validation size. Default: 0.1', required=False,
                                 default=0.1)
    argument_parser.add_argument("--learning-rate", type=float, help='Learning rate. Default: 0.00005',
                                 required=False, default=0.002)
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

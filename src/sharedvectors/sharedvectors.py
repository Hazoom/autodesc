import argparse
import os
import pickle

import argcomplete
from keras.models import load_model

from preprocessing import textpreprocess
from seq2seq.model import load_encoder_inputs


def load_shared_vector_space_model(shared_vector_model_path: str):
    return load_model(shared_vector_model_path)


def create_shared_vectors(train_code_vectors_file: str,
                          train_code_preprocessor_file: str,
                          shared_vector_model_file: str,
                          output_dir: str,
                          batch_size: int):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    shared_vector_model = load_shared_vector_space_model(shared_vector_model_file)
    code_pre_processor, _ = textpreprocess.load_text_preprocessor(train_code_preprocessor_file)
    encoder_vectors = load_encoder_inputs(train_code_vectors_file)

    print('Start predicting titles vectors...')
    titles_vectors = shared_vector_model.predict(encoder_vectors, batch_size=batch_size)
    print('Finished predicting titles vectors')

    results = {index: vector for index, vector in enumerate(titles_vectors)}
    with open(output_dir, 'wb') as out_fp:
        pickle.dump(results, out_fp)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--code-vectors-file", type=str,
                                 help='Train code vectors file path', required=True)
    argument_parser.add_argument("--code-preprocessor-file", type=str,
                                 help='Train code pre processor file path', required=True)
    argument_parser.add_argument("--shared-vector-model-file", type=str,
                                 help='Shared vector space model weights file path', required=True)
    argument_parser.add_argument("--output-dir", type=str, help='Output directory for vectors', required=True)
    argument_parser.add_argument("--batch-size", type=int, help='Batch size. Default: 32', required=False,
                                 default=256)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    create_shared_vectors(args.code_vectors_file,
                          args.code_preprocessor_file,
                          args.shared_vector_model_file,
                          args.output_dir,
                          args.batch_size)


if __name__ == '__main__':
    main()

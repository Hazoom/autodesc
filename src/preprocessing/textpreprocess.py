import argparse
import os

import argcomplete
import pandas as pd
import numpy as np
import dill
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


class TextPreprocessor:
    def __init__(self,
                 append_borders: bool = False,
                 n_vocab: int = 30000,
                 max_length: int = 128,
                 truncating: str = 'post',
                 padding: str = 'pre'):
        self.padding = padding
        self.truncating = truncating
        self.max_length = max_length
        self.n_vocab = n_vocab
        self.append_borders = append_borders
        self.tokenizer = Tokenizer(n_vocab, oov_token='<OOV>')
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.token_to_id = None
        self.id_to_token = None
        self.n_tokens = None
        self.padding_value = 0

    def process_text(self, text: str):
        if self.append_borders:
            return self.start_token + ' ' + text + ' ' + self.end_token
        return text

    def transform(self, texts):
        if self.token_to_id is None:
            raise Exception('Model is not fitted yet!')

        processed_texts = self.process_texts(texts)

        assert len(processed_texts) == len(texts)

        return self._transform_processed_texts(processed_texts)

    def _transform_processed_texts(self, processed_texts):
        print('Converting processed texts to integer vectors...')
        vectors = self.tokenizer.texts_to_sequences(processed_texts)
        print('Done converting processed texts to integer vectors')

        assert len(vectors) == len(processed_texts)

        print('Padding sequences...')
        padded_vectors = self._pad_sequences(vectors)
        print('Done padding sequences')

        assert len(padded_vectors) == len(vectors)

        return padded_vectors

    def fit_transform(self, texts):
        processed_texts = self.fit(texts)

        return self._transform_processed_texts(processed_texts)

    def fit(self, texts):
        print('Processing texts...')
        processed_texts = self.process_texts(texts)

        assert len(processed_texts) == len(texts)

        self.tokenizer.fit_on_texts(processed_texts)
        self.token_to_id = self.tokenizer.word_index
        self.id_to_token = {value: key for key, value in self.token_to_id.items()}
        self.n_tokens = max(self.tokenizer.word_index.values())
        print('Finished processing texts')

        return processed_texts

    def process_texts(self, texts):
        return [self.process_text(text) for text in texts]

    def _pad_sequences(self, sequences):
        return pad_sequences(sequences,
                             maxlen=self.max_length,
                             padding=self.padding,
                             truncating=self.truncating,
                             value=self.padding_value)


def _save_pre_processor(pre_processor: TextPreprocessor, output_dir: str, file_name: str):
    with open(os.path.join(output_dir, file_name), 'wb+') as out_fp:
        dill.dump(pre_processor, out_fp)


def _save_vectors(train_title_vectors, output_dir: str, file_name: str):
    np.save(os.path.join(output_dir, file_name), train_title_vectors)


def parse_data(input_file, output_dir):
    train_df = pd.read_csv(input_file)
    title_pre_processor = TextPreprocessor(append_borders=True, n_vocab=10000, max_length=128,
                                           truncating='post', padding='post')

    print('Fitting pre-processor on titles... (1/2)')
    train_title_vectors = title_pre_processor.fit_transform(train_df['title'].tolist())
    print('Finished fitting pre-processor on titles (1/2)')

    code_pre_processor = TextPreprocessor(append_borders=False, n_vocab=20000, max_length=128)

    print('Fitting pre-processor on codes... (2/2)')
    train_code_vectors = code_pre_processor.fit_transform(train_df['answer_code'].astype(str).tolist())
    print('Finished fitting pre-processor on codes (2/2)')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    print('Saving pre processors and vectors...')

    # save the pre processors for later use
    _save_pre_processor(title_pre_processor, output_dir, 'title_pre_processor.dpkl')
    _save_pre_processor(code_pre_processor, output_dir, 'code_pre_processor.dpkl')

    # save the vectors of title and code
    _save_vectors(train_title_vectors, output_dir, 'title_vectors.npy')
    _save_vectors(train_code_vectors, output_dir, 'code_vectors.npy')

    print('Done.')


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input-file", type=str, help='Input CSV file after cleaning', required=True)
    argument_parser.add_argument("--output-dir", type=str, help='Output directory', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    parse_data(args.input_file, args.output_dir)


if __name__ == '__main__':
    main()

import argparse
import os
from collections import OrderedDict

import argcomplete
import dill
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer, text_to_word_sequence

from preprocessing import cleaning
from bert.tokenization import FullTokenizer


def load_text_preprocessor(file_path):
    """
    Load TextPreprocessor dpickle file from disk.
    :param file_path: str
        File path on disk
    :return:
    text_pre_processor: TextPreprocessor
    n_tokens: int
    """
    with open(file_path, 'rb') as in_fp:
        text_pre_processor = dill.load(in_fp)
    n_tokens = text_pre_processor.n_tokens + 1  # + 1 because of padding token
    print(f'Loaded model: {file_path}. Number of tokens: {n_tokens}')
    return text_pre_processor, n_tokens


class TextPreprocessor:
    def __init__(self,
                 mode: str,  # title/code
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
        self.bert_tokenizer = FullTokenizer(vocab_file='resources/vocab.txt')
        if mode == 'title':
            self.tokenizer = self.bert_tokenizer.tokenize
            self.cleaner = cleaning.clean_code
        elif mode == 'code':
            self.tokenizer = text_to_word_sequence  # Tokenizer(num_words=n_vocab, oov_token='<OOV>')
            self.cleaner = cleaning.clean_title
        else:
            raise Exception(f'Unknown mode {mode}')
        self.start_token = '<START>'
        self.end_token = '<END>'
        self.token_to_id = None
        self.id_to_token = None
        self.n_tokens = None
        self.padding_value = 0
        self.indexer = None
        self.mode = mode

    def process_texts(self,
                      texts: str):
        if self.append_borders:
            return [[self.start_token] + self.tokenizer(self.cleaner(doc)) + [self.end_token] for doc in texts]
        return [self.tokenizer(self.cleaner(doc)) for doc in texts]

    def transform(self, texts):
        if self.token_to_id is None:
            raise Exception('Model is not fitted yet!')

        tokenized_data = self.process_texts(texts)

        print(f'Transforming {self.mode}...')
        indexed_data = self.indexer.tokenized_texts_to_sequences(tokenized_data)
        print(f'Done transforming {self.mode}')

        print(f'Padding {self.mode}...')
        padded_sequences = self._pad_sequences(indexed_data)
        print(f'Done padding {self.mode}')

        return padded_sequences

    def _transform_tokenized_texts(self, tokenized_data):
        print(f'Transforming {self.mode}...')
        indexed_data = self.indexer.tokenized_texts_to_sequences(tokenized_data)
        print(f'Done transforming {self.mode}')

        print(f'Padding {self.mode}...')
        padded_sequences = self._pad_sequences(indexed_data)
        print(f'Done padding {self.mode}')

        return padded_sequences

    def fit_transform(self, texts):
        tokenized_data = self.fit(texts)
        return self._transform_tokenized_texts(tokenized_data)

    def fit(self, texts):
        print(f'Tokenize {self.mode}...')
        tokenized_texts = self.process_texts(texts)
        assert len(tokenized_texts) == len(texts)
        print(f'Done tokenize {self.mode}')

        self.indexer = CustomIndexer(num_words=self.n_vocab)

        print(f'Fitting {self.mode}...')
        self.indexer.fit_on_tokenized_texts(tokenized_texts)
        print(f'Done fitting {self.mode}...')

        # Build Dictionary accounting For 0 padding, and reserve 1 for unknown and rare Words
        self.token_to_id = self.indexer.word_index
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.n_tokens = max(self.indexer.word_index.values())

        return tokenized_texts

    def _pad_sequences(self, sequences):
        return pad_sequences(sequences,
                             maxlen=self.max_length,
                             padding=self.padding,
                             truncating=self.truncating,
                             value=self.padding_value)


class CustomIndexer(Tokenizer):
    """
    Text vectorization utility class.
    This class inherits keras.preprocess.text.Tokenizer but adds methods
    to fit and transform on already tokenized text.
    Parameters
    ----------
    num_words : int
        the maximum number of words to keep, based
        on word frequency. Only the most common `num_words` words will
        be kept.
    """

    def __init__(self, num_words, **kwargs):
        # super().__init__(num_words, **kwargs)
        self.num_words = num_words
        self.word_counts = OrderedDict()
        self.word_docs = {}
        self.document_count = 0

    def fit_on_tokenized_texts(self, tokenized_texts):
        self.document_count = 0
        for seq in tokenized_texts:
            self.document_count += 1

            for w in seq:
                if w in self.word_counts:
                    self.word_counts[w] += 1
                else:
                    self.word_counts[w] = 1
            for w in set(seq):
                if w in self.word_docs:
                    self.word_docs[w] += 1
                else:
                    self.word_docs[w] = 1

        word_counts = list(self.word_counts.items())
        word_counts.sort(key=lambda x: x[1], reverse=True)
        sorted_voc = [wc[0] for wc in word_counts][:self.num_words]
        # note that index 0 and 1 are reserved, never assigned to an existing word
        self.word_index = dict(list(zip(sorted_voc, list(range(2, len(sorted_voc) + 2)))))

    def tokenized_texts_to_sequences(self, tok_texts):
        """Transforms tokenized text to a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            tokenized texts:  List[List[str]]
        # Returns
            A list of integers.
        """
        res = []
        for vector in self.tokenized_texts_to_sequences_generator(tok_texts):
            res.append(vector)
        return res

    def tokenized_texts_to_sequences_generator(self, tok_texts):
        """Transforms tokenized text to a sequence of integers.
        Only top "num_words" most frequent words will be taken into account.
        Only words known by the tokenizer will be taken into account.
        # Arguments
            tokenized texts:  List[List[str]]
        # Yields
            Yields individual sequences.
        """
        for seq in tok_texts:
            vector = []
            for w in seq:
                # if the word is missing you get oov_index
                i = self.word_index.get(w, 1)
                vector.append(i)
            yield vector


def _save_pre_processor(pre_processor: TextPreprocessor, output_dir: str, file_name: str):
    with open(os.path.join(output_dir, file_name), 'wb+') as out_fp:
        dill.dump(pre_processor, out_fp)


def _save_vectors(train_title_vectors, output_dir: str, file_name: str):
    np.save(os.path.join(output_dir, file_name), train_title_vectors)


def parse_data(input_file, output_dir):
    train_df = pd.read_csv(input_file)
    title_pre_processor = TextPreprocessor(mode='title', append_borders=True, n_vocab=10000, max_length=64,
                                           truncating='post', padding='post')

    print('Fitting pre-processor on titles... (1/2)')
    train_title_vectors = title_pre_processor.fit_transform(train_df['title'].tolist())
    print('Finished fitting pre-processor on titles (1/2)')

    code_pre_processor = TextPreprocessor(mode='code', append_borders=False, n_vocab=20000, max_length=30)

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

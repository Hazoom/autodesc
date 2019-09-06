"""
Copyright (c) 2018 Hamel Husain
"""
import argparse
import argcomplete
import numpy as np
import pandas as pd
from typing import List
from keras.preprocessing.sequence import pad_sequences
from pathos.multiprocessing import cpu_count
from pypeln import thread as th
from nltk.translate.bleu_score import corpus_bleu

from seq2seq.model import build_model, extract_encoder_model, extract_decoder_model
from preprocessing import textpreprocess


def load_model(encoder_seq_len: int,
               n_encoder_tokens: int,
               n_decoder_tokens: int,
               model_weights_path: str,
               word_embedding_dim: int = 300,
               hidden_state_dim: int = 768):
    # Load vectors and title/code pre processors
    seq2seq_model = build_model(word_embedding_dim,
                                hidden_state_dim,
                                encoder_seq_len,
                                n_encoder_tokens,
                                n_decoder_tokens)
    seq2seq_model.load_weights(model_weights_path)
    return seq2seq_model


def evaluate_seq2seq(code_preprocessor_file: str,
                     title_preprocessor_file: str,
                     test_file: str,
                     model_weights_path: str):
    # Load vectors and title/code pre processors
    title_pre_processor, n_decoder_tokens = textpreprocess.load_text_preprocessor(title_preprocessor_file)
    code_pre_processor, n_encoder_tokens = textpreprocess.load_text_preprocessor(code_preprocessor_file)

    # load model
    model = load_model(code_pre_processor.max_length,
                       n_encoder_tokens,
                       n_decoder_tokens,
                       model_weights_path)

    seq2seq_inference = Seq2SeqInference(encoder_preprocessor=code_pre_processor,
                                         decoder_preprocessor=title_pre_processor,
                                         seq2seq_model=model)

    # load test set
    test_df = pd.read_csv(test_file)
    test_codes = test_df['answer_code'].astype(str).to_list()
    test_titles = test_df['title'].astype(str).to_list()

    # show random predictions
    seq2seq_inference.demo_model_predictions(n_samples=15, input_texts=test_codes, output_text=test_titles)

    # evaluate model with BLEU score
    blue_score = seq2seq_inference.evaluate_model(code_strings=test_codes,
                                                  title_strings=test_titles,
                                                  max_len=title_pre_processor.max_length)
    print(f'BLEU score on test set: {blue_score}')


class Seq2SeqInference(object):
    def __init__(self,
                 encoder_preprocessor: textpreprocess.TextPreprocessor,
                 decoder_preprocessor: textpreprocess.TextPreprocessor,
                 seq2seq_model):
        self.enc_pp = encoder_preprocessor
        self.dec_pp = decoder_preprocessor
        self.seq2seq_model = seq2seq_model
        self.encoder_model = extract_encoder_model(seq2seq_model)
        self.decoder_model = extract_decoder_model(seq2seq_model)
        self.default_max_len = self.dec_pp.max_length
        self.nn = None
        self.rec_df = None

    def predict(self,
                raw_input_text: str,
                max_len: int = None):
        if max_len is None:
            max_len = self.default_max_len

        # get the encoder's features for the decoder
        raw_tokenized = self.enc_pp.transform([raw_input_text])
        encoding = self.encoder_model.predict(raw_tokenized)

        # we want to save the encoder's embedding before its updated by decoder
        # because we can use that as an embedding for other tasks.
        original_encoding = encoding

        state_value = np.array(self.dec_pp.token_to_id[self.dec_pp.start_token]).reshape(1, 1)

        decoded_sentence = []
        while True:
            predictions, state = self.decoder_model.predict([state_value, encoding])

            # We are going to ignore indices 0 (padding) and indices 1 (unknown)
            # Argmax will return the integer index corresponding to the prediction + 2 because we chopped off first two
            pred_idx = np.argmax(predictions[:, :, 2:]) + 2

            # retrieve word from index prediction
            pred_word_str = self.dec_pp.id_to_token[pred_idx]

            if pred_word_str == self.dec_pp.end_token or len(decoded_sentence) >= max_len:
                break

            decoded_sentence.append(pred_word_str)

            # update the decoder for the next word
            encoding = state
            state_value = np.array(pred_idx).reshape(1, 1)

        return original_encoding, ' '.join(decoded_sentence)

    def print_example(self,
                      index: int,
                      input_text: str,
                      output_text: str):
        """
        Prints an example of the model's prediction for manual inspection.
        """
        if index:
            print('\n\n==============================================')
            print(f'============== Example # {index} =================\n')

        print(f"Original Input:\n {input_text} \n")

        if output_text:
            print(f"Original Output:\n {output_text}")

        _, gen_title = self.predict(input_text)
        print(f"\n****** Predicted Output ******:\n {gen_title}")

    def demo_model_predictions(self,
                               n_samples: int,
                               input_texts,
                               output_text):
        demo_list = np.random.randint(low=1, high=len(input_texts), size=n_samples)
        for index in demo_list:
            self.print_example(index,
                               input_text=input_texts[index],
                               output_text=output_text[index])

    def evaluate_model(self,
                       code_strings: List[str],
                       title_strings: List[str],
                       max_len: int):
        assert len(code_strings) == len(title_strings)
        num_examples = len(code_strings)

        print('\nGenerating predictions.')
        # step over the whole set

        total_length_str = str(num_examples)
        cpu_cores = cpu_count()
        results = []
        (range(num_examples)
         | th.each(lambda index: self._get_prediction_and_actual(code_strings,
                                                                 index,
                                                                 max_len,
                                                                 title_strings,
                                                                 total_length_str,
                                                                 results),
                   workers=cpu_cores, maxsize=0)
         | list)
        print(f'Finished {total_length_str} out of {total_length_str}')
        results = sorted(results, key=lambda x: x[0])
        actual = [result[1] for result in results]
        predicted = [result[2] for result in results]

        print('Calculating BLEU...')
        bleu_score = corpus_bleu([[actual_item] for actual_item in actual], predicted)
        return bleu_score

    def _get_prediction_and_actual(self,
                                   code_strings,
                                   index,
                                   max_len,
                                   title_strings,
                                   total_length_str,
                                   results):
        _, y_hat = self.predict(raw_input_text=code_strings[index], max_len=max_len)
        actual_value = self.dec_pp.process_texts([title_strings[index]])[0]
        predicted_value = self.dec_pp.process_texts([y_hat])[0]

        if index % 100 == 0 and index > 0:
            print('Finished {} out of {}'.format(str(index), total_length_str))

        results.append((index, actual_value, predicted_value))


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--code-preprocessor-file", type=str,
                                 help='Train code pre processor file path', required=True)
    argument_parser.add_argument("--title-preprocessor-file", type=str,
                                 help='Train title pre processor file path', required=True)
    argument_parser.add_argument("--test-file", type=str,
                                 help='Test CSV file', required=True)
    argument_parser.add_argument("--model-weights-file", type=str,
                                 help='Model weights file path', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    evaluate_seq2seq(args.code_preprocessor_file,
                     args.title_preprocessor_file, args.test_file,
                     args.model_weights_file)


if __name__ == '__main__':
    main()

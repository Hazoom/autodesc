import argparse
import argcomplete
import numpy as np
from nltk.translate.bleu_score import corpus_bleu

from seq2seq.model import build_model, load_encoder_inputs, extract_encoder_model, extract_decoder_model
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


def evaluate_seq2seq(test_code_vectors_file: str,
                     code_preprocessor_file: str,
                     title_preprocessor_file: str,
                     model_weights_path: str):
    # Load vectors and title/code pre processors
    title_pre_processor, n_decoder_tokens = textpreprocess.load_text_preprocessor(title_preprocessor_file)
    code_pre_processor, n_encoder_tokens = textpreprocess.load_text_preprocessor(code_preprocessor_file)
    encoder_vectors = load_encoder_inputs(test_code_vectors_file)

    # load model
    model = load_model(encoder_vectors.shape[1],
                       n_encoder_tokens,
                       n_decoder_tokens,
                       model_weights_path)

    seq2seq_inf = Seq2SeqInference(encoder_preprocessor=code_pre_processor,
                                   decoder_preprocessor=title_pre_processor,
                                   seq2seq_model=model)

    seq2seq_inf.demo_model_predictions(n_samples=15, input_texts=test_codes, output_text=test_titles)


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
                max_len=None):
        """
        Use the seq2seq model to generate an output given the input.
        Inputs
        ------
        raw_input: str
            The body of what is to be summarized or translated.
        max_len: int (optional)
            The maximum length of the output
        """
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
                      index,
                      input_text,
                      output_text):
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
                       input_strings,
                       output_strings,
                       max_len):
        """
        Method for calculating BLEU Score.
        Parameters
        ----------
        input_strings : List[str]
            These are the issue bodies that we want to summarize
        output_strings : List[str]
            This is the ground truth we are trying to predict --> issue titles
        max_len: int
            Maximum sequence length
        Returns
        -------
        bleu : float
            The BLEU Score
        """
        actual, predicted = list(), list()
        assert len(input_strings) == len(output_strings)
        num_examples = len(input_strings)

        print('Generating predictions.')
        # step over the whole set

        for i in range(num_examples):
            _, y_hat = self.predict(raw_input_text=input_strings[i], max_len=max_len)
            actual.append(self.dec_pp.process_texts([output_strings[i]])[0])
            predicted.append(self.dec_pp.process_texts([y_hat])[0])

        print('Calculating BLEU...')
        bleu_score = corpus_bleu([[actual_item] for actual_item in actual], predicted)
        return bleu_score


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--test-code-vectors-file", type=str, help='Test code vectors file path',
                                 required=True)
    argument_parser.add_argument("--code-preprocessor-file", type=str,
                                 help='Train code pre processor file path', required=True)
    argument_parser.add_argument("--title-preprocessor-file", type=str,
                                 help='Train title pre processor file path', required=True)
    argument_parser.add_argument("--model-weights", type=str,
                                 help='Model weights file path', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    evaluate_seq2seq(args.test_code_vectors_file, args.code_preprocessor_file,
                     args.title_preprocessor_file,
                     args.model_weights_file)


if __name__ == '__main__':
    main()

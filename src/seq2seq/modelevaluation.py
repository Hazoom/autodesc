import argparse
import argcomplete

from seq2seq.model import build_model, load_decoder_inputs, load_encoder_inputs
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
                     test_title_vectors_file: str,
                     title_preprocessor_file: str,
                     model_weights_path: str):
    # Load vectors and title/code pre processors
    title_pre_processor, n_decoder_tokens = textpreprocess.load_text_preprocessor(title_preprocessor_file)
    code_pre_processor, n_encoder_tokens = textpreprocess.load_text_preprocessor(code_preprocessor_file)
    decoder_input_vectors, decoder_target_vectors = load_decoder_inputs(test_title_vectors_file)
    encoder_vectors = load_encoder_inputs(test_code_vectors_file)

    # load model
    model = load_model(encoder_vectors.shape[1],
                       n_encoder_tokens,
                       n_decoder_tokens,
                       model_weights_path)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--test-code-vectors-file", type=str, help='Test code vectors file path',
                                 required=True)
    argument_parser.add_argument("--code-preprocessor-file", type=str,
                                 help='Train code pre processor file path', required=True)
    argument_parser.add_argument("--test-title-vectors-file", type=str, help='Test title vectors file path',
                                 required=True)
    argument_parser.add_argument("--title-preprocessor-file", type=str,
                                 help='Train title pre processor file path', required=True)
    argument_parser.add_argument("--model-weights", type=str,
                                 help='Model weights file path', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    evaluate_seq2seq(args.test_code_vectors_file, args.code_preprocessor_file,
                     args.test_title_vectors_file, args.title_preprocessor_file,
                     args.model_weights_file)


if __name__ == '__main__':
    main()

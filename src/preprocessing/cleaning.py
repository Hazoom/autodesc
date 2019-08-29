import re
import argparse
import argcomplete
import pandas as pd
import textacy

CODE_RE = re.compile(r'<pre><code>(.+?)</code></pre>', re.DOTALL)


def _extract_code(code_text):
    matches = [match.group(1) for match in re.finditer(CODE_RE, code_text)]
    if matches:
        return matches[0].strip()  # take the first code snippet in the code answer
    return ''


def _clean_title(title):
    return textacy.preprocess_text(title,
                                   fix_unicode=True,
                                   lowercase=True,
                                   transliterate=True,
                                   no_urls=True,
                                   no_emails=True,
                                   no_phone_numbers=True,
                                   no_numbers=True,
                                   no_currency_symbols=True,
                                   no_punct=True,
                                   no_contractions=False,
                                   no_accents=True)


def clean_data(input_file, output_file):
    code_df = pd.read_csv(input_file)
    code_df['answer_code'] = code_df['answer_body'].apply(_extract_code)
    code_df['title'] = code_df['title'].apply(_clean_title)

    # filter relevant columns
    code_df = code_df.filter(['title', 'answer_code'], axis=1)
    code_df.to_csv(output_file)


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input-file", type=str, help='Input CSV file', required=True)
    argument_parser.add_argument("--output-file", type=str, help='Output CSV files', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    clean_data(args.input_file, args.output_file)


if __name__ == '__main__':
    main()

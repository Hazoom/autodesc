import re
import argparse
import argcomplete
import pandas as pd


CODE_RE = re.compile(r'<pre><code>(.+?)</code></pre>', re.DOTALL)


def _extract_code(code_text):
    matches = [match.group(1) for match in re.finditer(CODE_RE, code_text)]
    if matches:
        return matches[0].strip()  # take the first code snippet in the code answer
    return ''


def clean_data(input_file, output_file):
    code_df = pd.read_csv(input_file)
    code_df['answer_code'] = code_df['answer_body'].apply(_extract_code)

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

import re
import argparse
import argcomplete
import pandas as pd

CODE_RE = re.compile(r'<pre><code>(.+?)</code></pre>', re.DOTALL)


def _extract_code(code_text):
    matches = [match.group(1) for match in re.finditer(CODE_RE, code_text)]

    if not matches:
        return ''

    return matches[0].strip()  # take the first code snippet in the code answer


def clean_code(code):
    code = code.lower()
    code = re.sub(r'\t+', ' ', code)
    code = re.sub(r'\n+', ' ', code)
    code = re.sub(r' +', ' ', code)
    return code


def clean_title(title: str):
    title = title.lower()
    title = re.sub(r'\t+', ' ', title)
    title = re.sub(r'\n+', ' ', title)
    title = re.sub(r'[^a-z0-9]', ' ', title)
    title = re.sub(r' +', ' ', title)

    return title


def clean_data(input_file: str,
               output_file: str,
               bert_titles_file: str):
    code_df = pd.read_csv(input_file)
    code_df['answer_code_raw'] = code_df['answer_body'].apply(_extract_code)
    code_df['answer_code'] = code_df['answer_code_raw'].apply(clean_code)
    code_df['raw_title'] = code_df['title'].to_list()
    code_df['title'] = code_df['title'].apply(clean_title)

    # filter relevant columns
    code_df = code_df.filter(['title', 'answer_code', 'raw_title', 'answer_code_raw'], axis=1)
    code_df.to_csv(output_file, index_label='index', index=True)

    with open(bert_titles_file, 'w+') as out_fp:
        lines = code_df['title'].to_list()
        for line in lines:
            out_fp.write(line + '\n')


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input-file", type=str, help='Input CSV file', required=True)
    argument_parser.add_argument("--output-file", type=str, help='Output CSV files', required=True)
    argument_parser.add_argument("--bert-titles-file", type=str, help='Output titles txt', required=True)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    clean_data(args.input_file, args.output_file, args.bert_titles_file)


if __name__ == '__main__':
    main()

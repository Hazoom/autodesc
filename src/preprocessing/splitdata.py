import os
import argparse
import argcomplete
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def _write_to_csv(df_code, output_dir, file_name):
    df_code.to_csv(os.path.join(output_dir, file_name), index=True, index_label='index')


def split_data(input_file, output_dir, split_ratio):
    code_df = pd.read_csv(input_file)

    train, test = train_test_split(code_df, train_size=split_ratio, shuffle=True, random_state=RANDOM_STATE)

    print(f'Train set rows: {train.shape[0]:,}')
    print(f'Test set rows: {test.shape[0]:,}')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    _write_to_csv(train, output_dir, 'train.csv')
    _write_to_csv(test, output_dir, 'test.csv')


def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--input-file", type=str, help='Input CSV file', required=True)
    argument_parser.add_argument("--output-dir", type=str, help='Output directory', required=True)
    argument_parser.add_argument("--split-ratio", type=float, help='Split ratio. Default: 0.85', required=False,
                                 default=0.85)
    argcomplete.autocomplete(argument_parser)
    args = argument_parser.parse_args()
    split_data(args.input_file, args.output_dir, args.split_ratio)


if __name__ == '__main__':
    main()

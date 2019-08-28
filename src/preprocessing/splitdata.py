import os
import argparse
import argcomplete
import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def _reset_index(code_df):
    code_df.reset_index(drop=True, inplace=True)
    code_df = code_df.drop(code_df.columns[0], axis=1).copy()
    return code_df


def _write_to_csv(df_code, output_dir, file_name):
    df_code.to_csv(os.path.join(output_dir, file_name), index=False)


def split_data(input_file, output_dir, split_ratio):
    code_df = pd.read_csv(input_file)

    train, test = train_test_split(code_df, train_size=split_ratio, shuffle=True, random_state=RANDOM_STATE)
    train, validation = train_test_split(train, train_size=0.9, random_state=RANDOM_STATE)

    train = _reset_index(train)
    test = _reset_index(test)
    validation = _reset_index(validation)

    print(f'Train set rows: {train.shape[0]:,}')
    print(f'Validation set rows: {validation.shape[0]:,}')
    print(f'Test set rows: {test.shape[0]:,}')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    _write_to_csv(train, output_dir, 'train.csv')
    _write_to_csv(test, output_dir, 'test.csv')
    _write_to_csv(validation, output_dir, 'validation.csv')


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

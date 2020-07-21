import argparse


def init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", default='data/train.csv')
    parser.add_argument("--valid_file", default='data/valid.csv')
    parser.add_argument("--test_file", default='data/test.csv')
    parser.add_argument("--stopwords_file", default='data/stopwords.txt')

    parser.add_argument('--learning_rate', type=float, nargs='?', default=1.0)
    parser.add_argument('--epoch', type=int, nargs='?', default=3)
    parser.add_argument('--word_ngrams', type=int, nargs='?', default=1)
    parser.add_argument('--min_count', type=int, nargs='?', default=1)
    parser.add_argument('--model_name', type=str, nargs='?', default='output/model.pkl')
    parser.add_argument('--output_file', type=str, nargs='?', default='output/test_predict.csv')

    args = parser.parse_args()

    return args

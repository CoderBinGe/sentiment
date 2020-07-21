import pandas as pd
import jieba
from sklearn.metrics import f1_score

pd.set_option('display.max_columns', None)


def load_data(file_csv):
    df = pd.read_csv(file_csv, usecols=None, encoding='utf-8', nrows=15000, low_memory=False)
    return df


def load_train(args):
    train_file = args.train_file
    train_df = load_data(train_file)
    print(f'train data size: {train_df.shape}')
    print(f'train data:\n {train_df.head()}')
    return train_df


def load_valid(args):
    valid_file = args.valid_file
    valid_df = load_data(valid_file)
    print(f'valid data size: {valid_df.shape}')
    return valid_df


def load_test(args):
    test_file = args.test_file
    test_df = load_data(test_file)
    print(f'test data size: {test_df.shape}')
    return test_df


def load_stopwords(args):
    stopwords_file = args.stopwords_file
    stopwords = []
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            stopwords.append(line)
    return stopwords


def seg_words(args, contents):
    stopwords = load_stopwords(args)
    contents_segs = list()
    for content in contents:
        rcontent = content.replace("\r\n", " ").replace("\n", " ")
        segs = [word for word in jieba.cut(rcontent) if word not in stopwords]
        contents_segs.append(" ".join(segs))
    return contents_segs


def get_f1_score(y_true, y_pred):
    return f1_score(y_true, y_pred, labels=[1, 0, -1, -2], average='macro')

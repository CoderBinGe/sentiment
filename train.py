import logging
import numpy as np
import joblib
import utils.args_util as args_util
import utils.data_util as data_util
from skift import FirstColFtClassifier

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = args_util.init_args()

    logger.info("start load data...")
    train_df = data_util.load_train(args)
    valid_df = data_util.load_valid(args)

    logger.info("start seg train data...")
    content_train = train_df.iloc[:, 1]
    content_train = data_util.seg_words(args, content_train)

    logger.info("prepare train format...")
    train_data_format = np.asarray([content_train]).T  # array([[第三次 参加 大众],[同行 点 小吃  榴莲 酥],...])

    columns = train_df.columns.values.tolist()

    logger.info("start train model...")
    classifier_dict = dict()
    for column in columns[2:]:  # 标签
        train_label = train_df[column]
        logger.info("start train %s model" % column)
        sk_clf = FirstColFtClassifier(lr=args.learning_rate, epoch=args.epoch,
                                      wordNgrams=args.word_ngrams,
                                      minCount=args.min_count, verbose=2)
        sk_clf.fit(train_data_format, train_label)
        logger.info("complete train %s model" % column)
        classifier_dict[column] = sk_clf

    logger.info("start save train model...")
    model_name = args.model_name
    joblib.dump(classifier_dict, model_name)

    logger.info("start seg valid data...")
    content_valid = valid_df.iloc[:, 1]
    content_valid = data_util.seg_words(args, content_valid)

    logger.info("prepare valid format")
    valid_data_format = np.asarray([content_valid]).T

    logger.info("start compute f1 score for valid model...")
    f1_score_dict = dict()
    for column in columns[2:]:
        true_label = np.asarray(valid_df[column])
        classifier = classifier_dict[column]
        pred_label = classifier.predict(valid_data_format).astype(int)
        f1_score = data_util.get_f1_score(true_label, pred_label)
        f1_score_dict[column] = f1_score

    f1_score = np.mean(list(f1_score_dict.values()))
    str_score = "\n"
    for column in columns[2:]:
        str_score += column + ":" + str(f1_score_dict[column]) + "\n"

    logger.info("f1_scores: %s\n" % str_score)
    logger.info("f1_score: %s" % f1_score)



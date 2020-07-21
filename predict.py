import logging
import joblib
import numpy as np
import utils.args_util as args_util
import utils.data_util as data_util
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    args = args_util.init_args()

    logger.info("start load test data...")
    test_df = data_util.load_test(args)

    logger.info("start load model...")
    model_name = args.model_name
    classifier_dict = joblib.load(model_name)

    logger.info("start seg test data...")
    content_test = test_df['content']
    content_test = data_util.seg_words(args, content_test)

    logger.info("prepare test format...")
    test_data_format = np.asarray([content_test]).T

    columns = test_df.columns.values.tolist()

    logger.info("start predict test data...")
    for column in columns[2:]:
        test_df[column] = classifier_dict[column].predict(
            test_data_format).astype(int)
        logger.info("complete %s predict" % column)

    test_df.to_csv(args.output_file, encoding="utf-8", index=False)

# -*- coding: utf-8 -*-
import click
import logging
import joblib
import pandas as pd
import os
import csv

from src.config import model_catboost_path, model_xgboost_path, TARGET_COL
from src.data.preprocess import pre_process_val
from src.features.build_features import feature_generation


def csv_write(path, data):
    with open(path, mode='w', encoding='utf-8') as w_file:
        column_names = ['Index'] + TARGET_COL
        file_writer = csv.DictWriter(w_file, delimiter=",",
                                     lineterminator="\r", fieldnames=column_names)
        file_writer.writeheader()
        for i in range(len(data)):
            file_writer.writerow({"Index": i, TARGET_COL[0]: data[i]})


@click.command()
@click.argument('data_filepath', type=click.Path())
def main(data_filepath):
    logger = logging.getLogger(__name__)
    logger.info('Model inference for ' + data_filepath + '')

    data_filepath = data_filepath[14:]
    test = pd.read_csv(data_filepath)
    test = pre_process_val(test)
    test = feature_generation(test)

    model_catboost = joblib.load(model_catboost_path)
    model_xgboost = joblib.load(model_xgboost_path)

    y_predicted_catboost = model_catboost.predict(test)
    logger.log(logging.INFO, '\n' + str(y_predicted_catboost))
    csv_write('reports/' + data_filepath[:-4].replace('/', '_') + '_result_catboost.csv', y_predicted_catboost)

    y_predicted_xgboost = model_xgboost.predict(test)
    logger.log(logging.INFO, '\n' + str(y_predicted_xgboost))
    csv_write('reports/' + data_filepath[:-4].replace('/', '_') + '_result_xgboost.csv', y_predicted_xgboost)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
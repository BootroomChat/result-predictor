import json

import numpy as np
import tensorflow as tf

TRAINING_SET_FRACTION = 0.95
HIDDEN_UNITS = [44]
OPTIMIZER = "Adam"

num_columns = [
    'DispossessedInD3rdPerAttack',
    'DispossessedPerAttack',
    'DribblePerAttack',
    'ForwardPassMeterPerAttack',
    'ForwardPassPerAttack',
    'FouledPerAttack',
    'KeyPassPerAttack',
    'PassCrossAccuratePerAttack',
    'PassLeadingKpPerAttack',
    'PassPerAttack',
    'ShotPerAttack',
    'SuccessPassPerAttack',
    'SuccessfulPassToA3rdPerAttack',
    'TurnoverPerAttack',
    'UnSuccessPassPerAttack',
    'AerialLossPerDefense',
    'AerialWonPerDefense',
    'ClearPerDefense',
    'FoulPerDefense',
    'InterceptionsPerDefense',
    'SavePerDefense',
    'TacklesPerDefense',
    'GoalConcededTotal',
    'GoalPerAttack'
]
category_columns = []


def map_results(results):
    features = {}

    for result in results:
        for key in result.keys():
            if isinstance(result[key], dict) or isinstance(result[key], list):
                continue
            if key not in features:
                features[key] = []

            features[key].append(result[key])

    for key in features.keys():
        features[key] = np.array(features[key])

    return features, features['result']


def parse_data(data):
    return data


def feature_columns():
    feature_columns = []
    for key in num_columns:
        feature_columns.append(tf.feature_column.numeric_column(key + '0'))
        feature_columns.append(tf.feature_column.numeric_column(key + '1'))

    for key in category_columns:
        category_column = tf.feature_column.categorical_column_with_vocabulary_list(key, ['0', '1'])
        feature_columns.append(tf.feature_column.indicator_column(category_column))
    return feature_columns


def load_json(file_name):
    try:
        with open(file_name, encoding="utf-8") as json_data:
            d = json.load(json_data)
            return d
    except Exception as e:
        print(e)
        return {}


def write_json(file_name, json_data):
    with open(file_name, 'w') as outfile:
        json.dump(json_data, outfile, ensure_ascii=False)
        return json_data

import numpy as np

import tensorflow as tf

TRAINING_SET_FRACTION = 0.95
HIDDEN_UNITS = [36]
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

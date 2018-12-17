import csv
import json

import numpy as np
import pandas as pd
import tensorflow as tf

TRAINING_SET_FRACTION = 0.95
num_columns = ['DispossessedInD3rdPerAttack',
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
               'TacklesPerDefense']
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


def predict():
    origin_test_data = load_json('test.json')
    test_data = parse_data(origin_test_data)
    test_results = test_data
    test_features, test_labels = map_results(test_results)

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_features,
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )
    model = tf.estimator.DNNClassifier(
        model_dir='model/',
        hidden_units=[10,10],
        feature_columns=feature_columns(),
        n_classes=3,
        label_vocabulary=['0', '1', '2'],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.1,
            l1_regularization_strength=0.001
        ))
    predictions = list(model.predict(input_fn=test_input_fn))
    print(origin_test_data)
    for i, prediction in enumerate(predictions):
        origin_test_data[str(i)]['expected'] = prediction['probabilities'][1]
    df = pd.DataFrame(origin_test_data)
    print(df[['match', 'expected', 'result', 'diff']])
    # print(df[['player_name', 'is_goal', 'xG']])
    # print(df.groupby('player_name').sum())


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


def main(argv):
    origin_data = load_json('data.json')
    data = parse_data(origin_data)
    train_results_len = int(TRAINING_SET_FRACTION * len(data))
    train_results = data[:train_results_len]
    test_results = data[train_results_len:]

    train_features, train_labels = map_results(train_results)
    test_features, test_labels = map_results(test_results)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=train_features,
        y=train_labels,
        batch_size=500,
        num_epochs=None,
        shuffle=True
    )

    test_input_fn = tf.estimator.inputs.numpy_input_fn(
        x=test_features,
        y=test_labels,
        num_epochs=1,
        shuffle=False
    )

    model = tf.estimator.DNNClassifier(
        model_dir='model/',
        hidden_units=[10,10],
        feature_columns=feature_columns(),
        n_classes=3,
        label_vocabulary=['0', '1', '2'],
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.01,
            l1_regularization_strength=0.001
        ))

    with open('training-log.csv', 'w') as stream:
        csvwriter = csv.writer(stream)

        for i in range(0, 200):
            model.train(input_fn=train_input_fn, steps=100)
            evaluation_result = model.evaluate(input_fn=test_input_fn)

            predictions = list(model.predict(input_fn=test_input_fn))

            csvwriter.writerow([(i + 1) * 100, evaluation_result['accuracy'], evaluation_result['average_loss'],])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
    predict()

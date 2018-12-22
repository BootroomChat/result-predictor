import csv
import json
import sys

import pandas as pd

from predict_config import *


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
        hidden_units=HIDDEN_UNITS,
        feature_columns=feature_columns(),
        n_classes=3,
        label_vocabulary=['0', '1', '2'],
        optimizer=OPTIMIZER
    )
    predictions = list(model.predict(input_fn=test_input_fn))
    # print(predictions)
    correct, total = (0.0, 0.0)
    for i, prediction in enumerate(predictions):
        origin_test_data[i]['expected'] = int(prediction['classes'][0])
        origin_test_data[i]['prob'] = max(list(prediction['probabilities']))
        if origin_test_data[i]['expected'] == int(origin_test_data[i]['result']) and origin_test_data[i]['prob'] > 0:
            correct += 1
        total += 1
    df = pd.DataFrame(origin_test_data)
    print(correct, total, float(correct) / total)
    # print(df[['match', 'prob', 'expected', 'result', 'diff']])
    # print(df[['player_name', 'is_goal', 'xG']])
    # print(df.groupby('player_name').sum())


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
        hidden_units=HIDDEN_UNITS,
        feature_columns=feature_columns(),
        n_classes=3,
        label_vocabulary=['0', '1', '2'],
        optimizer=OPTIMIZER
    )

    with open('training-log.csv', 'w') as stream:
        csvwriter = csv.writer(stream)

        for i in range(0, 200):
            model.train(input_fn=train_input_fn, steps=100)
            evaluation_result = model.evaluate(input_fn=test_input_fn)

            predictions = list(model.predict(input_fn=test_input_fn))

            csvwriter.writerow([(i + 1) * 100, evaluation_result['accuracy'], evaluation_result['average_loss'], ])


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    if len(sys.argv) < 2 or sys.argv[1] == 'predict':
        predict()
    else:
        tf.app.run(main=main)

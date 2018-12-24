import pandas as pd
from dateutil.parser import parse
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV

from predict_config import *

keys = [y for x in [[k + i for k in num_columns] for i in ['0', '1']] for y in x]
lr: LogisticRegressionCV = joblib.load('LogisticRegression.pkl')


def predict():
    data = load_json('future.json')
    for i, item in enumerate(data):
        if num_columns[0] + '1' not in item or num_columns[0] + '0' not in item:
            continue
        print(i, item['match'])
        data[i] = predict_by_lr(item)
        data[i] = predict_by_nn(data[i])
    write_json('future.json', data)


def predict_by_nn(item):
    origin_test_data = [item]
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
    prediction = list(model.predict(input_fn=test_input_fn))[0]
    item['nn_prob_win'], item['nn_prob_draw'], item['nn_prob_lose'] = [round(float(prob) * 1000)/1000.0 for prob in
                                                                       prediction['probabilities']]
    item['nn_predict'] = str(int(prediction['classes'][0]))
    return item


def predict_by_lr(item):
    item_df = pd.DataFrame([item])
    item['lr_prob_win'], item['lr_prob_draw'], item['lr_prob_lose'] = [round(float(prob) * 1000)/1000.0 for prob in
                                                                       lr.predict_proba(item_df[keys])[0]]
    item['lr_predict'] = str(int(lr.predict(item_df[keys])[0]))
    return item


def reformat():
    data = load_json('future.json')
    result = []
    for i, item in enumerate(data):
        new_item = {'homePreStats': {}, 'awayPreStats': {},
                    'timestamp': int(parse("{0} {1} +0000".format(item['date'], item['time'])).timestamp())}
        for key, value in item.items():
            if key in keys:
                new_item['homePreStats' if key.endswith('0') else 'awayPreStats'][key[:-1]] = value
            else:
                new_item[key] = value
        result.append(new_item)
    write_json('schedule.json', result)


if __name__ == '__main__':
    predict()
    reformat()

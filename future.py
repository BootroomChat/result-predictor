from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV

from predict_config import *
from ..xt_util import *

keys = [y for x in [[k + i for k in num_columns] for i in ['0', '1']] for y in x]
lr: LogisticRegressionCV = joblib.load('LogisticRegression.pkl')


def predict():
    data = load_json('future.json')
    for i, item in enumerate(data):
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
    item['lr_prob_win'], item['lr_prob_draw'], item['lr_prob_lose'] = prediction['probabilities']
    item['lr_predict'] = prediction['classes'][0]
    return item


def predict_by_lr(item):
    item_df = pd.DataFrame([item])
    item['lr_prob_win'], item['lr_prob_draw'], item['lr_prob_lose'] = lr.predict_proba(item_df[keys])[0]
    item['lr_predict'] = lr.predict(item_df[keys])[0]
    return item


if __name__ == '__main__':
    predict()
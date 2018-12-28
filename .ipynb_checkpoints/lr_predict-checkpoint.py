from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegressionCV

from predict_config import *

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

keys = [y for x in [[k + i for k in num_columns] for i in ['0', '1']] for y in x]
data = load_json('data.json')
df = pd.DataFrame(data)
x = df[keys]
y = df['result']
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=20)

lr = LogisticRegressionCV(solver='newton-cg', cv=12)
lr.fit(x, y)

joblib.dump(lr, 'LogisticRegression.pkl')
# lr: LogisticRegressionCV = joblib.load('LogisticRegression.pkl')
test_data = load_json('test.json')
correct, total = (0.0, 0.0)
for item in test_data:
    item_df = pd.DataFrame([item])
    print(lr.predict_proba(item_df[keys]))
    result = lr.predict(item_df[keys])
    if result[0] == item_df['result'][0]:
        correct += 1
    total += 1
print(correct, total, float(correct) / total)

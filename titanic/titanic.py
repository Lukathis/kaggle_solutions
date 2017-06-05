import pandas as pd

train = pd.read_csv('/Users/Chi/PycharmProjects/kaggle_solutions/titanic/data/train.csv')
test = pd.read_csv('/Users/Chi/PycharmProjects/kaggle_solutions/titanic/data/test.csv')

# print(train.info())
# print(test.info())

select_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']

X_train = train[select_features]
X_test = test[select_features]

y_train = train['Survived']

X_train['Embarked'].fillna('S', inplace=True)
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)

X_test['Embarked'].fillna('S', inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

X_train.info()
X_test.info()

from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
X_train=dict_vec.fit_transform(X_train.to_dict(orient='record'))
print(dict_vec.feature_names_)


X_test = dict_vec.transform(X_test.to_dict(orient='record'))

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
from xgboost import XGBClassifier
xgbc = XGBClassifier()

from sklearn.cross_validation import cross_val_score
print(cross_val_score(rfc, X_train, y_train, cv=5).mean())
print(cross_val_score(xgbc, X_train, y_train, cv=5).mean())

rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)
rfc_submission = pd.DataFrame({'PassengerId':test['PassengerId'], 'Survived':rfc_y_predict})
rfc_submission.to_csv('/Users/Chi/PycharmProjects/kaggle_solutions/titanic/data/rfc_submission.csv', index=False)

xgbc.fit(X_train, y_train)
XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=1,
              gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3,
              min_child_weight=1, missing=None, n_estimators=100, n_jobs=-1,
              objective='binary:logistic', reg_alpha=0, reg_lambda=1,
              scale_pos_weight=1, seed=0, silent=True, subsample=1)
xgbc_y_predict=xgbc.predict(X_test)
xgbc_submission=pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived':xgbc_y_predict})
xgbc_submission.to_csv('/Users/Chi/PycharmProjects/kaggle_solutions/titanic/data/xgbc_submission.csv', index=False)

# 超参数估计未进行
# from sklearn.grid_search import GridSearchCV
# params={'max_depth':range(2,7), 'n_estimators':range(100, 1100, 200), 'learning_rate':[0.05, 0.1, 0.25, 0.5, 1.0]}
#
# xgbc_best=XGBClassifier()
# gs=GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=1)
# gs.fit(X_train, y_train)
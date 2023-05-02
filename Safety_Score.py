import numpy as np
from sklearn import impute
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def test_model(train, test, descriptors, target):
    preds = []

    X_train = train[descriptors]
    y_train = train[target]
    X_test = test[descriptors]
    y_test = test[target]

    imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train.values.ravel())
    predictions = model.predict(X_test)
    preds.extend(predictions)

    print(predictions)
    preds = [int(i) for i in preds]
    print([int(i) for i in preds])


train_data = pd.read_csv("Safety.csv")
test_data = pd.read_csv("Safety_New_Dataset.csv")
descriptors = ["Flash Point", "Peroxide formation", "Resistivitylog10", "AIT", "CGPlog10", "CLP"]
test_model(train_data, test_data, descriptors, "Safety_Score")

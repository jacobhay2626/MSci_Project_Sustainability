import numpy as np
from sklearn import impute
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor


def test_model(train, test, descriptors, target):

    preds = []

    # DEFINE SPLITS
    X_train = train[descriptors]
    y_train = train[target]
    X_test = test[descriptors]
    y_test = test[target]

    # MEAN FOR MISSING VALUES IN X TRAIN
    imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    # SCALE AND TRANSFORM X TRAIN
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    # DEFINE MODEL AND FIT X TRAIN/Y TRAIN
    model = HistGradientBoostingRegressor()
    model.fit(X_train, y_train.values.ravel())
    predictions = model.predict(X_test)
    preds.extend(predictions)

    print(predictions)
    preds = [int(i) for i in preds]
    print([int(i) for i in preds])


train_data = pd.read_csv("../Datasets/Environmental.csv")
test_data = pd.read_csv("../Datasets/Environmental_New_Dataset.csv")
descriptors = ["Boiling Point", "Aquatic Toxicity (mg/L)log10", "OH radical RC (cm/molecule s)",
               "Vapour Pressure (mmHg)", "EC50 Daphnia Magna (48 hour) (mg/L)"]
test_model(train_data, test_data, descriptors, "Environmental_Score")

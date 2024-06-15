import numpy as np
from sklearn import impute
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def test_model(train, test, descriptors, target):
    X_train = train[descriptors]
    y_train = train[target]
    X_test = test[descriptors]
    y_test = test[target]

    imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    model = HistGradientBoostingClassifier()
    model.fit(X_train, y_train.values.ravel())
    predictions = model.predict(X_test)

    print(predictions)


train_data = pd.read_csv("../Datasets/Dataset.csv")
test_data = pd.read_csv("../Datasets/NewDataset.csv")
descriptors = ["HTP(ingestion)log10", "HTP(Inhalation)log10", "XVP", "Flash Point",
               "Peroxide formation", "AIT", "CGPlog10", "CLP", "Aquatic Toxicity (mg/L)log10",
               "Vapour Pressure (mmHg)", "OH radical RC (cm/molecule s)", "EC50 daphnia magna (mg/L)"]

test_model(train_data, test_data, descriptors, "Sustainability")

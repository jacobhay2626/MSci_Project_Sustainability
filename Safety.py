import numpy as np
import pandas as pd
from sklearn import impute, svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.ensemble import RandomForestRegressor

def Safety_score(descriptors, ml_method, validation):

    data = pd.read_csv('Safety.csv')

    data_descriptors = data[descriptors]

    columns = list(data_descriptors)

    y = data['Safety_Score']

    imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')

    imp.fit(data_descriptors)

    data_df_impute = imp.transform(data_descriptors)

    scaler = preprocessing.StandardScaler().fit(data_df_impute)
    data_df_impute = scaler.transform(data_df_impute)

    data_df_impute = pd.DataFrame(data_df_impute, columns=columns)

    #  Want four regression methods:
    #   SDG regressor?
    #  Support Vector Regression
    #  Linear
    #  RF
    #  Decision Trees? WOuld need to speak to Sam/Emma about using PCA to reduce dimensionality.

    if ml_method == "LR":
        model = LinearRegression()
    elif ml_method == "RF":
        model = RandomForestRegressor()
    elif ml_method == "SVR":
        model = svm.SVR()
    elif ml_method == "SGD":
        model = SGDRegressor()

    if validation == 10:
        k = 10
    else:
        k = len(y)

    kf = KFold(n_splits=k, random_state=None, shuffle=True)

    preds = []
    real = []
    name = []
    # add back name
    data_df_impute.insert(loc=0,
                          column='CHEM21 Solvents',
                          value=data["CHEM21 Solvents"])

    for train_index, test_index in kf.split(data_df_impute):
        X_train, X_test = data_df_impute.iloc[train_index, :], data_df_impute.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        # name to list
        name += X_test["CHEM21 Solvents"].tolist()
        X_train = X_train.drop(columns=["CHEM21 Solvents"])
        X_test = X_test.drop(columns=["CHEM21 Solvents"])
        model.fit(X_train, y_train.values.ravel())
        # print(model.coef_)
        y_pred = model.predict(X_test)
        preds.extend(y_pred)
        real.extend(y_test)

    # R2
    preds = [int(i) for i in preds]
    r_squared = r2_score(real, preds)
    print(r_squared)
    print(mean_absolute_error(real, preds))
    print([int(i) for i in preds])
    print(np.std(real))

    # plt.scatter(X_test, y_test, color="red")
    # plt.plot(X_test, preds, color="blue", linewidth=3)
    #
    # plt.xticks(())
    # plt.yticks(())
    #
    # plt.show()

    print(X_test)
    print(y_test)


descriptors = ['Boiling Point', 'Resistivitylog10', 'Peroxide formation', 'AIT', 'CGPlog10', 'CLP']
Safety_score(descriptors, "SGD", "LOO")
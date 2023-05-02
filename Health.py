import numpy as np
import pandas as pd
from sklearn import impute, svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px


def importance_method(importance, descriptors):
    importance = np.array(importance).T
    imp = [[np.mean(x), np.std(x)] for x in importance]
    for i in range(len(descriptors)):
        imp[i].insert(0, descriptors[i])
    importance = pd.DataFrame(data=imp, columns=["Descriptor", "Mean", "SD"])
    fig_x = px.bar(importance, x="Descriptor", y="Mean", error_y="SD", title="Health Descriptors")
    fig_x.update_layout(font=dict(family="Courier New, monospace", size=15))
    fig_x.show()
    return importance


def Health_Score(descriptors, ml_method, validation):
    data = pd.read_csv('Health.csv')

    data_descriptors = data[descriptors]

    columns = list(data_descriptors)

    y = data['Health_Score']

    imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')

    imp.fit(data_descriptors)

    data_df_impute = imp.transform(data_descriptors)

    scaler = preprocessing.StandardScaler().fit(data_df_impute)
    data_df_impute = scaler.transform(data_df_impute)

    data_df_impute = pd.DataFrame(data_df_impute, columns=columns)

    if ml_method == "LR":
        model = LinearRegression()
    elif ml_method == "RF":
        model = RandomForestRegressor()
    elif ml_method == "SVR":
        model = svm.SVR()

    if validation == 10:
        k = 10
    else:
        k = len(y)

    kf = KFold(n_splits=k, random_state=None, shuffle=True)

    h_preds = []
    h_real = []
    importance = []

    for train_index, test_index in kf.split(data_df_impute):
        X_train, X_test = data_df_impute.iloc[train_index, :], data_df_impute.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.drop(columns=["CHEM21 Solvents"])
        X_test = X_test.drop(columns=["CHEM21 Solvents"])
        model.fit(X_train, y_train.values.ravel())
        if ml_method == "LR":
            print(model.coef_)
            print(model.intercept_)
        elif ml_method == "RF":
            importance.append(model.feature_importances_)
        y_pred = model.predict(X_test)
        h_preds.extend(y_pred)
        h_real.extend(y_test)

    h_preds = [int(i) for i in h_preds]
    r_squared = r2_score(h_real, h_preds)
    print(r_squared)
    print(mean_absolute_error(h_real, h_preds))
    print([int(i) for i in h_preds])
    print(np.std(h_real))

    if ml_method == "RF":
        importance_method(importance, descriptors)

    return h_preds


descriptors = ['INH', 'SDP', 'ING', 'XVP']
Health_Score(descriptors, "RF", 10)


import numpy as np
import pandas as pd
from sklearn import impute, svm
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px

# WANT TO VISUALISE THE AVERAGE IMPORTANCE OF EACH FEATURE IN EACH SPLIT FOR RF.
def importance_method(importance, descriptors):
    importance = np.array(importance).T
    imp = [[np.mean(x), np.std(x)] for x in importance]
    for i in range(len(descriptors)):
        imp[i].insert(0, descriptors[i])
    importance = pd.DataFrame(data=imp, columns=["Descriptor", "Mean", "SD"])
    fig_x = px.bar(importance, x="Descriptor", y="Mean", error_y="SD", title="Safety Descriptors")
    fig_x.update_layout(font=dict(family="Courier New, monospace", size=18))
    fig_x.show()
    return importance


def Safety_score(descriptors, ml_method, validation):
    data = pd.read_csv('../../Datasets/Safety.csv')

    data_descriptors = data[descriptors]

    columns = list(data_descriptors)

    y = data['Safety_Score']

    # DATA IMPUTING
    imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')

    imp.fit(data_descriptors)

    data_df_impute = imp.transform(data_descriptors)

    # DATA SCALING
    scaler = preprocessing.StandardScaler().fit(data_df_impute)
    data_df_impute = scaler.transform(data_df_impute)

    data_df_impute = pd.DataFrame(data_df_impute, columns=columns)

    # METHOD
    if ml_method == "LR":
        model = LinearRegression()
    elif ml_method == "RF":
        model = RandomForestRegressor()
    elif ml_method == "SVR":
        model = svm.SVR()

    # VALIDATION SPLITS
    if validation == 10:
        k = 10
    else:
        k = len(y)

    kf = KFold(n_splits=k, random_state=None, shuffle=True)

    s_preds = []
    s_real = []
    importance = []

    for train_index, test_index in kf.split(data_df_impute):
        X_train, X_test = data_df_impute.iloc[train_index, :], data_df_impute.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.drop(columns=["CHEM21 Solvents"])
        X_test = X_test.drop(columns=["CHEM21 Solvents"])
        model.fit(X_train, y_train.values.ravel())
        # LR ANALYSIS = COEFFS AND INTERCEPTS
        if ml_method == "LR":
            print(model.coef_)
            print(model.intercept_)
        # RF ANALYSIS = FEATURE IMPORTANCE
        elif ml_method == "RF":
            importance.append(model.feature_importances_)
        y_pred = model.predict(X_test)
        s_preds.extend(y_pred)
        s_real.extend(y_test)

    # R2
    s_preds = [int(i) for i in s_preds]
    r_squared = r2_score(s_real, s_preds)
    print(r_squared)
    print(mean_absolute_error(s_real, s_preds))
    print([int(i) for i in s_preds])
    print(np.std(s_real))

    if ml_method == "RF":
        importance_method(importance, descriptors)
    return s_preds, s_real


descriptors = ['FP', 'AIT', 'CGP', 'RES', 'CLP', 'PER']
Safety_score(descriptors, "RF", 10)

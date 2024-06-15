import numpy as np
import pandas as pd
from sklearn import impute, preprocessing
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
from sklearn.model_selection import KFold
from sklearn import svm

# WANT TO VISUALISE THE AVERAGE IMPORTANCE OF EACH FEATURE IN EACH SPLIT FOR RF.
def importance_method(importance, descriptors):
    importance = np.array(importance).T
    imp = [[np.mean(x), np.std(x)] for x in importance]
    for i in range(len(descriptors)):
        imp[i].insert(0, descriptors[i])
    importance = pd.DataFrame(data=imp, columns=["Descriptor", "Mean", "SD"])
    fig_x = px.bar(importance, x="Descriptor", y="Mean", error_y="SD", title="Environment Descriptors")
    fig_x.update_layout(font=dict(family="Courier New, monospace", size=15))
    fig_x.show()
    return importance


def Environmental_Score(descriptors, ml_method, validation):
    data = pd.read_csv('../../Datasets/Environmental.csv')

    data_descriptors = data[descriptors]

    columns = list(data_descriptors)

    y = data['Environmental_Score']

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

    # CROSS VALIDATION SPLITS
    if validation == 10:
        k = 10
    else:
        k = len(y)

    kf = KFold(n_splits=k, random_state=None, shuffle=True)

    # EMPTY ARRAYS FOR PREDS AND REAL VALUES
    e_preds = []
    e_real = []
    importance = []

    #
    for train_index, test_index in kf.split(data_df_impute):
        X_train, X_test = data_df_impute.iloc[train_index, :], data_df_impute.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.drop(columns=["CHEM21 Solvents"])
        X_test = X_test.drop(columns=["CHEM21 Solvents"])
        model.fit(X_train, y_train.values.ravel())
        # LR ANALYSIS: COEFFICIENTS AND INTERCEPT
        if ml_method == "LR":
            print(model.coef_)
            print(model.intercept_)
        # RF ANALYSIS: FEATURE IMPORTANCES
        elif ml_method == "RF":
            importance.append(model.feature_importances_)
        y_pred = model.predict(X_test)
        e_preds.extend(y_pred)
        e_real.extend(y_test)

    # SET PREDICTIONS TO INTEGERS
    e_preds = [int(i) for i in e_preds]
    r_squared = r2_score(e_real, e_preds)
    # R SQUARED AND MAE VALUES
    print(r_squared)
    print(mean_absolute_error(e_real, e_preds))
    print([int(i) for i in e_preds])
    print(np.std(e_real))

    # IMPORTANCE IF USING RF
    if ml_method == "RF":
        importance_method(importance, descriptors)
    return e_preds


descriptors = ['BP', 'VP', 'RC', 'DM', 'FM']
Environmental_Score(descriptors, "RF", 10)

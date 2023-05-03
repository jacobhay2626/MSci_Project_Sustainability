import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import impute
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, GridSearchCV



def classification(descriptors, ml_method, validation):
    #  ////////////////////////////////////////////////////////////////////////////////////////////////////

    #  READ CSV FILE
    data = pd.read_csv("Dataset.csv")

    #  ////////////////////////////////////////////////////////////////////////////////////////////////////

    #  DEFINE DESCRIPTORS
    data_descriptors = data[descriptors]

    y = data['Sustainability']

    columns = list(data_descriptors)

    # columns_b = list(string_d_data)

    #  ////////////////////////////////////////////////////////////////////////////////////////////////////

    #  MEAN, FIT, TRANSFORM AND SCALE

    # All missing values were given the mean value for the column
    imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    # fit to data
    imp.fit(data_descriptors)
    # transform data
    data_df_impute = imp.transform(data_descriptors)

    # scaling

    scaler = preprocessing.StandardScaler().fit(data_df_impute)
    data_df_impute = scaler.transform(data_df_impute)

    #  ////////////////////////////////////////////////////////////////////////////////////////////////////

    # CREATE DATAFRAME

    data_df_impute = pd.DataFrame(data_df_impute, columns=columns)

    #  ////////////////////////////////////////////////////////////////////////////////////////////////////

    #  METHOD, VALIDATION METHOD

    if ml_method == "RF":
        model = RandomForestClassifier(max_depth=9)
    elif ml_method == "SVM":
        model = SVC()

    if validation == 10:
        k = 10
    else:
        k = len(y)

    kf = KFold(n_splits=k, random_state=None, shuffle=True)

    #  ////////////////////////////////////////////////////////////////////////////////////////////////////

    #  MODEL

    preds = []
    real = []

    for train_index, test_index in kf.split(data_df_impute):
        X_train, X_test = data_df_impute.iloc[train_index, :], data_df_impute.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        X_train = X_train.drop(columns=["CHEM21 Solvents"])
        X_test = X_test.drop(columns=["CHEM21 Solvents"])
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        preds.extend(y_pred)
        real.extend(y_test)

    #  ////////////////////////////////////////////////////////////////////////////////////////////////////

    #  CONFUSION MATRIX, CLASSIFICATION REPORT

    print("=== Confusion Matrix ===")
    print(confusion_matrix(real, preds))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(real, preds))

    # param_grid = {
    #     'n_estimators': [25, 50, 100, 150],
    #     'max_features': ['sqrt', 'log2', None],
    #     'max_depth': [3, 6, 9]
    # }
    #
    # random_search = GridSearchCV(RandomForestClassifier(),
    #                              param_grid=param_grid)
    #
    # random_search.fit(X_train, y_train)
    # print(random_search.best_estimator_)

    return preds, real, confusion_matrix(real, preds), classification_report(real, preds)

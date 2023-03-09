import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn import impute
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier


def classification(descriptors, ml_method, validation):
    data = pd.read_csv("Dataset.csv")
    data_descriptors = data[descriptors]
    y = data['Sustainability']

    columns = list(data_descriptors)

    # All missing values were given the mean value for the column
    imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
    # fit to data
    imp.fit(data_descriptors)
    # transform data
    data_df_impute = imp.transform(data_descriptors)

    # scaling

    scaler = preprocessing.StandardScaler().fit(data_df_impute)
    data_df_impute = scaler.transform(data_df_impute)

    data_df_impute = pd.DataFrame(data_df_impute, columns=columns)

    # X_train, X_test, y_train, y_test = train_test_split(data_df_impute, y, test_size=0.2)
    if ml_method == "RF":
        model = RandomForestClassifier()
    elif ml_method == "SVM":
        model = SVC(kernel='linear')

    if validation == 10:
        k = 10
    else:
        k = len(y)

    kf = KFold(n_splits=k, random_state=None, shuffle=False)

    preds = []
    real = []

    for train_index, test_index in kf.split(data_df_impute):
        X_train, X_test = data_df_impute.iloc[train_index, :], data_df_impute.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train.values.ravel())
        y_pred = model.predict(X_test)
        preds.extend(y_pred)
        real.extend(y_test)

    print(preds)
    print(real)
    print("=== Confusion Matrix ===")
    print(confusion_matrix(real, preds))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(real, preds))
    return preds, real, confusion_matrix(real, preds), classification_report(real, preds)

# ---------------------------------------------------------------------------------------------
# RANDOM FORREST METHOD

# randomforrest = RandomForestClassifier()
# randomforrest.fit(X_train, y_train)
# predictions = randomforrest.predict(X_test)
# print(predictions)
# print(y_test)

# ---------------------------------------------------------------------------------------------

# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, predictions))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(y_test, predictions))

# ---------------------------------------------------------------------------------------------

# HOLDOUT CROSS VALIDATION:
# x_train, x_test, y_train, y_test = train_test_split()

# K FOLD METHOD
# NUM_SPLITS = X
# kfold = KFold(n_splits=NUM_SPLITS)
# split_data = kfold.split(data)

# LEAVE p OUT Method
#  P_VAL = X
# lpocv = LeavePOut(p=p_VAL)
# split_lpocv.split(data)

# LEAVE ONE OUT METHOD
# loocv = LeaveOneOut()
# split_loocv = loocv.split(data)

# ---------------------------------------------------------------------------------------------

# SVM METHOD
#
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
#
# clf = SVC(kernel='linear')
# clf.fit(x_train,y_train)
# y_pred = clf.predict(x_test)
# print(accuracy_score(y_test,y_pred))

# ---------------------------------------------------------------------------------------------

# Cross Validation (Leave one out) 2nd
# Plot the graphs for descriptors (correlation) 1st
# Hyper parameters (change # of neighbours) 4th
# Try SVM and random forrest 3rd compare methods

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import preprocessing
from sklearn import impute
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("Dataset.csv")
data_descriptors = data.drop(columns=['CHEM21 Solvents'])
data_descriptors = data_descriptors.drop(['Sustainability'], axis=1).values
y = data['Sustainability'].values

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

data_df_impute = pd.DataFrame(data_df_impute, columns=columns, )

for x in columns:
    for y in columns:
        if x == y:
            continue
        if x != y:
            fig = px.scatter(data_df_impute, x=x, y=y)
            if y != x:
                break

X_train, X_test, y_train, y_test = train_test_split(data_df_impute, y, random_state=0)

# randomforrest = RandomForestClassifier()
# randomforrest.fit(X_train, y_train)
# predictions = randomforrest.predict(X_test)
# print(predictions)
# print(y_test)
#
# print("=== Confusion Matrix ===")
# print(confusion_matrix(y_test, predictions))
# print('\n')
# print("=== Classification Report ===")
# print(classification_report(y_test, predictions))

# Cross Validation (Leave one out) 2nd
# Plot the graphs for descriptors (correlation) 1st
# Hyper parameters (change # of neighbours) 4th
# Try SVM and random forrest 3rd compare methods

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn import impute
from sklearn import preprocessing
from itertools import combinations
import os

if not os.path.exists("Correlations"):
    os.mkdir("Correlations")

data = pd.read_csv('Dataset.csv')
data_descriptors = data.drop(columns=['CHEM21 Solvents', 'Sustainability'])

columns = list(data_descriptors)
imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data_descriptors)
data_df_impute = imp.transform(data_descriptors)

scaler = preprocessing.StandardScaler().fit(data_df_impute)
data_df_impute = scaler.transform(data_df_impute)

data_df_impute = pd.DataFrame(data_df_impute, columns=columns)

res = list(combinations(data_descriptors, 2))


for i in res:
    x = i[0]
    y = i[1]
    fig = px.scatter(data_df_impute, x=i[0], y=i[1])





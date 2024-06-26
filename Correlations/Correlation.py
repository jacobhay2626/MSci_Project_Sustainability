import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn import impute
from sklearn import preprocessing
from itertools import combinations


#
# IDENTIFYING ANY CORRELATION BETWEEN VARIABLES

data = pd.read_csv('../Datasets/Dataset.csv')
data_descriptors = data.drop(columns=['CHEM21 Solvents', 'Sustainability'])


columns = list(data_descriptors)

#Data Imputation
imp = impute.SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(data_descriptors)
data_df_impute = imp.transform(data_descriptors)

# Data Scaling
scaler = preprocessing.StandardScaler().fit(data_df_impute)
data_df_impute = scaler.transform(data_df_impute)

data_df_impute = pd.DataFrame(data_df_impute, columns=columns)

# List combinations of pairs of the descriptors
res = list(combinations(data_descriptors, 2))
fig_list = []

if not os.path.exists(""):
    os.mkdir("")

# Plot any correlations
for i in res:
    x = i[0]
    y = i[1]
    fig = px.scatter(data_df_impute, x=i[0], y=i[1])
    fig_list.append(fig)

pio.write_image(fig_list[0], "Correlations/fig0.png")
pio.write_image(fig_list[1], "fig1.png")
pio.write_image(fig_list[2], "fig2.png")
pio.write_image(fig_list[3], "fig3.png")
pio.write_image(fig_list[4], "Correlations/fig4.png")
pio.write_image(fig_list[5], "Correlations/fig5.png")
pio.write_image(fig_list[6], "fig6.png")
pio.write_image(fig_list[7], "Correlations/fig7.png")
pio.write_image(fig_list[8], "fig8.png")
pio.write_image(fig_list[9], "Correlations/fig9.png")
pio.write_image(fig_list[10], "Correlations/fig10.png")
pio.write_image(fig_list[11], "Correlations/fig11.png")
pio.write_image(fig_list[12], "Correlations/fig12.png")
pio.write_image(fig_list[13], "Correlations/fig13.png")
pio.write_image(fig_list[14], "Correlations/fig14.png")
pio.write_image(fig_list[15], "Correlations/fig15.png")
pio.write_image(fig_list[16], "Correlations/fig16.png")
pio.write_image(fig_list[17], "fig17.png")
pio.write_image(fig_list[18], "Correlations/fig18.png")
pio.write_image(fig_list[19], "fig19.png")
pio.write_image(fig_list[20], "fig20.png")
pio.write_image(fig_list[21], "Correlations/fig21.png")
pio.write_image(fig_list[22], "Correlations/fig22.png")
pio.write_image(fig_list[23], "fig23.png")
pio.write_image(fig_list[24], "Correlations/fig24.png")
pio.write_image(fig_list[25], "fig25.png")
pio.write_image(fig_list[26], "Correlations/fig26.png")
pio.write_image(fig_list[27], "Correlations/fig27.png")
pio.write_image(fig_list[28], "Correlations/fig28.png")
pio.write_image(fig_list[29], "Correlations/fig29.png")
pio.write_image(fig_list[30], "fig30.png")
pio.write_image(fig_list[31], "Correlations/fig31.png")
pio.write_image(fig_list[32], "fig32.png")
pio.write_image(fig_list[33], "Correlations/fig33.png")
pio.write_image(fig_list[34], "Correlations/fig34.png")
pio.write_image(fig_list[35], "Correlations/fig35.png")
pio.write_image(fig_list[36], "fig36.png")
pio.write_image(fig_list[37], "Correlations/fig37.png")
pio.write_image(fig_list[38], "fig38.png")
pio.write_image(fig_list[39], "Correlations/fig39.png")
pio.write_image(fig_list[40], "fig40.png")
pio.write_image(fig_list[41], "Correlations/fig41.png")
pio.write_image(fig_list[42], "fig42.png")
pio.write_image(fig_list[43], "fig43.png")
pio.write_image(fig_list[44], "fig44.png")
pio.write_image(fig_list[45], "Correlations/fig45.png")
pio.write_image(fig_list[46], "fig46.png")
pio.write_image(fig_list[47], "fig47.png")
pio.write_image(fig_list[48], "fig48.png")
pio.write_image(fig_list[49], "Correlations/fig49.png")
pio.write_image(fig_list[50], "fig50.png")
pio.write_image(fig_list[51], "Correlations/fig51.png")
pio.write_image(fig_list[52], "Correlations/fig52.png")
pio.write_image(fig_list[53], "fig53.png")
pio.write_image(fig_list[54], "Correlations/fig54.png")

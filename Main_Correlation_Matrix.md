During the feature selection part of the model, we figured out that very different sets of features were chosen 
in different runs. Hence, a decision was made to search for highly correlated features among all the features 
in the dataset.

The following code looks for highly correlated features in the model's input data.


```python
# %%
#%load_ext autoreload
#%autoreload 2

import matplotlib.pyplot as plt
import numpy as np
import random
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import os
from sklearn.feature_selection import RFECV
import pandas as pd
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
import matplotlib.pyplot as plt

from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import (
    recall_score,
    f1_score,
    precision_score,
    confusion_matrix,
    make_scorer,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    KFold,
)
from sklearn.feature_selection import SelectKBest, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import importlib
import os
from sklearn.feature_selection import (
    SelectKBest,
    RFE,
    mutual_info_regression,
    f_regression,
    mutual_info_classif,
)
from sklearn.preprocessing import RobustScaler
#import eli5
#from eli5.sklearn import PermutationImportance
from sklearn.inspection import permutation_importance
import xgboost as xgb
import random
import pickle
import openpyxl
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
import pickle
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import random
import importlib

def splitting_train_test(df):

    # To save the train and test sets
    df_train_list = []
    df_test_list = []

    # List of typhoons that are to be used as a test set 
 
    typhoons_with_impact_data=list(np.unique(df.typhoon))

    for typhoon in typhoons_with_impact_data:
        if len(df[df["typhoon"] == typhoon]) >1:
            df_train_list.append(df[df["typhoon"] != typhoon])
            df_test_list.append(df[df["typhoon"] == typhoon])

    return df_train_list, df_test_list


def unweighted_random(y_train, y_test):
    options = y_train.value_counts(normalize=True)
    y_pred = random.choices(population=list(options.index), k=len(y_test))
    return y_pred

def weighted_random(y_train, y_test):
    options = y_train.value_counts()
    y_pred = random.choices(
        population=list(options.index), weights=list(options.values), k=len(y_test)
    )
    return y_pred


wor_dir="/home/mforooshani/Typhoon-Impact-based-forecasting-model-training-5:7/IBF-Typhoon-model/"
os.chdir(wor_dir)
cdir = os.getcwd()

# Import functions
from models.regression.rf_regression_0 import (rf_regression_features,rf_regression_performance,)
#from models.regression.xgb_regression import (xgb_regression_features,xgb_regression_performance,)


combined_input_data=pd.read_csv("Training-data-new/data/model_input/combined_input_data.csv")

typhoons_with_impact_data=['bopha2012', 'conson2010', 'durian2006', 'fengshen2008',
       'fung-wong2014', 'goni2015', 'goni2020', 'hagupit2014',
       'haima2016', 'haiyan2013', 'jangmi2014', 'kalmaegi2014',
       'kammuri2019', 'ketsana2009', 'koppu2015', 'krosa2013',
       'linfa2015', 'lingling2014', 'mangkhut2018', 'mekkhala2015',
       'melor2015', 'meranti2016', 'molave2020', 'mujigae2015',
       'nakri2019', 'nari2013', 'nesat2011', 'nock-ten2016', 'noul2015',
       'phanfone2019', 'rammasun2014', 'sarika2016', 'saudel2020',
       'tokage2016', 'trami2013', 'usagi2013', 'utor2013', 'vamco2020',
       'vongfong2020', 'yutu2018']

len(np.unique(combined_input_data.typhoon))
combined_input_data=combined_input_data[combined_input_data.typhoon.isin(typhoons_with_impact_data)]


def set_zeros(x):
    x_max = 25
    y_max = 50
    
    v_max = x[0]
    rainfall_max = x[1]
    damage = x[2]
    if pd.notnull(damage):
        value = damage
    elif v_max > x_max or rainfall_max > y_max:
        value =damage
    elif (v_max < np.sqrt((1- (rainfall_max**2/y_max ** 2))*x_max ** 2)):
        value = 0
    #elif ((v_max < x_max)  and  (rainfall_max_6h < y_max) ):
    #elif (v_max < x_max ):
    #value = 0
    else:
        value = np.nan

    return value
combined_input_data["DAM_perc_dmg"] = combined_input_data[["HAZ_v_max", "HAZ_rainfall_Total", "DAM_perc_dmg"]].apply(set_zeros, axis="columns")


np.mean(combined_input_data["DAM_perc_dmg"])
combined_input_data = combined_input_data[combined_input_data['DAM_perc_dmg'].notnull()]
np.mean(combined_input_data["DAM_perc_dmg"])
np.unique(combined_input_data.typhoon)

def cubeic(x):
    #x=float(x)
    value=x*x*x
    return value

combined_input_data['HAZ_v_max_3']=combined_input_data['HAZ_v_max'].apply(lambda x: x*x*x) 


combined_input_data =combined_input_data.filter(['typhoon','HAZ_rainfall_Total', 
        'HAZ_rainfall_max_6h',
        'HAZ_rainfall_max_24h',
        'HAZ_v_max',
        'HAZ_v_max_3',
        'HAZ_dis_track_min',
        'GEN_landslide_per',
        'GEN_stormsurge_per',
        'GEN_Bu_p_inSSA', 
        'GEN_Bu_p_LS', 
        'GEN_Red_per_LSbldg',
        'GEN_Or_per_LSblg', 
        'GEN_Yel_per_LSSAb', 
        'GEN_RED_per_SSAbldg',
        'GEN_OR_per_SSAbldg',
        'GEN_Yellow_per_LSbl',
        'TOP_mean_slope',
        'TOP_mean_elevation_m', 
        'TOP_ruggedness_stdev', 
        'TOP_mean_ruggedness',
        'TOP_slope_stdev', 
        'VUL_poverty_perc',
        'GEN_with_coast',
        'GEN_coast_length', 
        'VUL_Housing_Units',
        'VUL_StrongRoof_StrongWall', 
        'VUL_StrongRoof_LightWall',
        'VUL_StrongRoof_SalvageWall', 
        'VUL_LightRoof_StrongWall',
        'VUL_LightRoof_LightWall', 
        'VUL_LightRoof_SalvageWall',
        'VUL_SalvagedRoof_StrongWall',
        'VUL_SalvagedRoof_LightWall',
        'VUL_SalvagedRoof_SalvageWall', 
        'VUL_vulnerable_groups',
        'VUL_pantawid_pamilya_beneficiary', 
        'DAM_perc_dmg'])


features_name = combined_input_data.columns
#display(features_name)

features =['HAZ_rainfall_Total', 
        'HAZ_rainfall_max_6h',
        'HAZ_rainfall_max_24h',
        'HAZ_v_max',
        'HAZ_v_max_3',
        'HAZ_dis_track_min',
        'GEN_landslide_per',
        'GEN_stormsurge_per',
        'GEN_Bu_p_inSSA', 
        'GEN_Bu_p_LS', 
        'GEN_Red_per_LSbldg',
        'GEN_Or_per_LSblg', 
        'GEN_Yel_per_LSSAb', 
        'GEN_RED_per_SSAbldg',
        'GEN_OR_per_SSAbldg',
        'GEN_Yellow_per_LSbl',
        'TOP_mean_slope',
        'TOP_mean_elevation_m', 
        'TOP_ruggedness_stdev', 
        'TOP_mean_ruggedness',
        'TOP_slope_stdev', 
        'VUL_poverty_perc',
        'GEN_with_coast',
        'GEN_coast_length', 
        'VUL_Housing_Units',
        'VUL_StrongRoof_StrongWall', 
        'VUL_StrongRoof_LightWall',
        'VUL_StrongRoof_SalvageWall', 
        'VUL_LightRoof_StrongWall',
        'VUL_LightRoof_LightWall', 
        'VUL_LightRoof_SalvageWall',
        'VUL_SalvagedRoof_StrongWall',
        'VUL_SalvagedRoof_LightWall',
        'VUL_SalvagedRoof_SalvageWall', 
        'VUL_vulnerable_groups',
        'VUL_pantawid_pamilya_beneficiary']


df=combined_input_data.dropna()
display(df)
```

```python
#The correlation Matrix is also done for the input data where the damage value is greater than 10.
#df = df[df['DAM_perc_dmg'] > 10]
#df
```

```python
#get the correlation matrix
import seaborn as sn
import matplotlib.pyplot as plt

file_name = "All-my-test/output/corr_matrix.png"
path = os.path.join(wor_dir, file_name)

fig, ax = plt.subplots()

corrMatrix = df.corr()
#print (corrMatrix)

plt.rcParams["figure.figsize"] = (48,48)

sn.set(font_scale=2.5)
heatmap = sn.heatmap(corrMatrix, annot=True, cbar_kws={"shrink": .5}, annot_kws={"size": 16})
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':42}, pad=18)

#fig.savefig(path, format="png")

plt.show()
```

```python
#get the correlation matrix (creating a square matrix with dimensions equal to the number of features)
#get the absolute value of correlation
import seaborn as sn
import matplotlib.pyplot as plt

file_name = "All-my-test/output/corr_matrix_abs.png"
path = os.path.join(wor_dir, file_name)

fig, ax = plt.subplots()

corrMatrix_abs = df.corr().abs()
#print (corrMatrix)

plt.rcParams["figure.figsize"] = (48,48)

sn.set(font_scale=2.5)
heatmap = sn.heatmap(corrMatrix_abs, annot=True, cbar_kws={"shrink": .5}, annot_kws={"size": 16})
heatmap.set_title('Correlation Heatmap (abs)', fontdict={'fontsize':42}, pad=18)

#fig.savefig(path, format="png")

plt.show()
```

```python
pair = corrMatrix_abs.where(np.triu(np.ones(corrMatrix_abs.shape),k=1).astype(np.bool)).stack().sort_values(ascending=True)
pairs= pair[pair.gt(0.8)]
print(pairs)
```

```python
#Correlation matrix will be mirror image(all the diagonal elements=1). 
#does not matter that we select the upper triangular or lower triangular part of the correlation matrix.

upper_tri = corrMatrix_abs.where(np.triu(np.ones(corrMatrix_abs.shape),k=1).astype(np.bool))
#print(upper_tri)
```

```python
to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.80)]
print(to_drop)
```

```python
df = df.drop(df[to_drop], axis=1)
#print(df.head())

# %%
df = df.drop('typhoon', axis=1)
df
```

```python
df.columns.tolist()
```

```python
fig, ax = plt.subplots()

file_name2 = "All-my-test/output/corr_matrix_drop.png"
path = os.path.join(wor_dir, file_name2)

corrMatrix = df.corr()
#print (corrMatrix)

plt.rcParams["figure.figsize"] = (48,48)
#plt.figure(figsize=(36,36))

sn.set(font_scale=2.5)
heatmap = sn.heatmap(corrMatrix, annot=True, cbar_kws={"shrink": .5}, annot_kws={"size": 16})
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':42}, pad=18)

#fig.savefig(path, format="png")

plt.show()
```

```python
names = df.columns.tolist()
names

display(df[names])
```

```python
"""
VIF is another method for finding highly correlated features if there is still in existence.
VIF method, picks each feature and regresses it against all of the other features so VIF value for a feature 
demonstrates the correlation of that feature in total with all the other ones, and not only with one specific feature.
Normally if the estimated VIF value for a feature is greater than 7 so it can be considered a highly correlated feature.
"""

#Implementing VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor

#The independent variables set
X = df[names]
#Creating VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

#Calculating VIF for each feature
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]


print(vif_data)
```

```python
vif_data_sort = vif_data
vif_data_sort = vif_data.sort_values('VIF')
vif_data_sort = vif_data_sort.reset_index(drop=True)
display(vif_data_sort)
```

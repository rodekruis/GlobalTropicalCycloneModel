RMSE Estimation for Random Forest & XGBoost (before and after Reduced Overfitting) while not using train_test_split function and test and train set are fixed based on list of Typhoons.

The goal is that to compare RMSE estimated by train_test_split function (80/20 while one typhoon is considered as test each time) with the one according to train_test_split by typhoons (considering 8/39 typhoons as the test set).

```python
%load_ext jupyter_black
```

```python
import random
import statistics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from numpy.lib.function_base import average

from utils import get_clean_dataset

```

```python
df = get_clean_dataset()
```

```python
#Old and New set of bins
#bins2= [0, 1, 60, 101]    #Old bins
bins2 = [0, 0.00009, 1, 10, 50, 101]    #New bins
samples_per_bin2, binsP2 = np.histogram(df['DAM_perc_dmg'], bins=bins2)
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:],samples_per_bin2)

print(samples_per_bin2)
print(binsP2)

```

###The data (except target) needs to be standardize

```python
#Separate target from all other data
#df_scaled = df
#df_scaled

#Separate typhoon from other features
dfs = np.split(df, [1], axis=1)
dfa = np.split(dfs[1], [36], axis=1)
#print(dfs[0], dfs[1], dfa[0], dfa[1])

#Standardaize data 
scaler = preprocessing.StandardScaler().fit(dfa[0])
X1 = scaler.transform(dfa[0])
Xnew = pd.DataFrame(X1)
display(Xnew)


```

```python
dfa[1] = dfa[1].astype(float)
```

```python
Xnew = pd.concat([Xnew.reset_index(drop=True), dfa[1].reset_index(drop=True)], axis=1)
Xnew
```

```python
i=0
for feature in features:
    Xnew = Xnew.rename(columns={i: feature})
    i+=1
    
Xnew = pd.concat([dfs[0].reset_index(drop=True),Xnew.reset_index(drop=True)], axis=1)
#Xnew
```

```python
typhoons_lst = Xnew.typhoon.unique()
typhoons_lst.tolist()
```

####Check the mean value of each grop of typhoon to figure out most severe typhoons.

```python
#df["typhoon"].unique().tolist()
df_mean_value=pd.DataFrame(columns = ['mean_value'])
df_mean_value['mean_value'] = df.groupby('typhoon')['DAM_perc_dmg'].mean()
df_mean_value
```

```python
#Choose a test set randomly among all the typhoons(no matter in terms of typhoon severity)
#lst = random.sample(typhoons_lst.tolist(),k=8)

"""Better to choose one of the lists below which are balanced in terms of typhoon severity
(half severe, half with low mean value)
"""

lst=['durian2006', 'goni2020', 'melor2015', 'rammasun2014', 'tokage2016', 'utor2013', 'vamco2020', 'haima2016']
#lst=['haiyan2013', 'yutu2018', 'meranti2016', 'kammuri2019', 'saudel2020', 'hagupit2014', 'fengshen2008', 'mekkhala2015']

lst
```

###The next two cells, split test and train dataframe

```python
df_test=pd.DataFrame(Xnew, columns = ['typhoon','HAZ_rainfall_Total',
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

df_test = Xnew[Xnew['typhoon'] == lst[0]]
df_test=df_test.append(Xnew[Xnew['typhoon'] == lst[1]])
df_test=df_test.append(Xnew[Xnew['typhoon'] == lst[2]])
df_test=df_test.append(Xnew[Xnew['typhoon'] == lst[3]])
df_test=df_test.append(Xnew[Xnew['typhoon'] == lst[4]])
df_test=df_test.append(Xnew[Xnew['typhoon'] == lst[5]])
df_test=df_test.append(Xnew[Xnew['typhoon'] == lst[6]])
df_test=df_test.append(Xnew[Xnew['typhoon'] == lst[7]])

df_test
```

```python
Xnew.drop(Xnew.index[Xnew['typhoon'] == lst[0]], inplace=True)
Xnew.drop(Xnew.index[Xnew['typhoon'] == lst[1]], inplace=True)
Xnew.drop(Xnew.index[Xnew['typhoon'] == lst[2]], inplace=True)
Xnew.drop(Xnew.index[Xnew['typhoon'] == lst[3]], inplace=True)
Xnew.drop(Xnew.index[Xnew['typhoon'] == lst[4]], inplace=True)
Xnew.drop(Xnew.index[Xnew['typhoon'] == lst[5]], inplace=True)
Xnew.drop(Xnew.index[Xnew['typhoon'] == lst[6]], inplace=True)
Xnew.drop(Xnew.index[Xnew['typhoon'] == lst[7]], inplace=True)

df_train=Xnew
display(df_train)
```

```python
# Split X and y from dataframe features
X_test = df_test[features]
X_train = df_train[features]

y_train = df_train["DAM_perc_dmg"]
y_test = df_test["DAM_perc_dmg"]

bin_index_test=np.digitize(y_test, bins=binsP2)
bin_index_train=np.digitize(y_train, bins=binsP2)
```

###In the next two cells it is possible to select XGBoost or RandomForest model, before or after reducedOverfitting.

```python
#XGBoost

#xgb = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, gamma=1, reg_lambda=0.1, colsample_bytree=0.8)
#xgb_model=xgb.fit(X_train, y_train)
    
xgb = XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
                   colsample_bynode=0.8, colsample_bytree=0.8, gamma=3, eta=0.01,
                   importance_type='gain', learning_rate=0.1, max_delta_step=0,
                   max_depth=4, min_child_weight=1, missing=1, n_estimators=100, early_stopping_rounds=10,
                   n_jobs=1, nthread=None, objective='reg:squarederror', random_state=0,
                   reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                   silent=None, subsample=0.8, verbosity=1, eval_metric=["rmse", "logloss"])

    
eval_set = [(X_test, y_test)]
xgb_model=xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
```

```python
#Random Forest
    
rf = RandomForestRegressor(max_depth=None, n_estimators=100, min_samples_split=8,min_samples_leaf=5)
#rf = RandomForestRegressor(max_depth=None, n_estimators=100, min_samples_split=8,min_samples_leaf=5, max_samples=0.7)

rf_model=rf.fit(X_train, y_train)   
```

```python
X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())
```

```python
#import statsmodels.api as sm
#X2 = sm.add_constant(X_train)
#X2 = np.array(X2).astype(float)
#y_train = np.array(y_train).astype(float)
#est = sm.OLS(y_train, X2)
#est2 = est.fit()
#est2.summary()

```

```python
#RMSE Estimation for each bins

#If you run random forest then put rf as the model's name 
#If you run xgboost then put xgb as the model's name

y_pred_train = xgb.predict(X_train)

mse_train_idx1 = mean_squared_error(y_train[bin_index_train==1], y_pred_train[bin_index_train==1])
rmse_train_1 = np.sqrt(mse_train_idx1)
mse_train_idx2 = mean_squared_error(y_train[bin_index_train==2], y_pred_train[bin_index_train==2])
rmse_train_2 = np.sqrt(mse_train_idx2)
mse_train_idx3 = mean_squared_error(y_train[bin_index_train==3], y_pred_train[bin_index_train==3])
rmse_train_3 = np.sqrt(mse_train_idx3)
mse_train_idx4 = mean_squared_error(y_train[bin_index_train==4], y_pred_train[bin_index_train==4])
rmse_train_4 = np.sqrt(mse_train_idx4)
mse_train_idx5 = mean_squared_error(y_train[bin_index_train==5], y_pred_train[bin_index_train==5])
rmse_train_5 = np.sqrt(mse_train_idx5)


print('----- Training_bins_RMSE  ------')
print(f'Root mean squared error of bins_1: {rmse_train_1:.2f}')
print(f'Root mean squared error of bins_2: {rmse_train_2:.2f}')
print(f'Root mean squared error of bins_3: {rmse_train_3:.2f}')
print(f'Root mean squared error of bins_4: {rmse_train_4:.2f}')
print(f'Root mean squared error of bins_5: {rmse_train_5:.2f}')


y_pred = xgb.predict(X_test)
    
mse_idx1 = mean_squared_error(y_test[bin_index_test==1], y_pred[bin_index_test==1])
rmse_1 = np.sqrt(mse_idx1)
mse_idx2 = mean_squared_error(y_test[bin_index_test==2], y_pred[bin_index_test==2])
rmse_2 = np.sqrt(mse_idx2)
mse_idx3 = mean_squared_error(y_test[bin_index_test==3], y_pred[bin_index_test==3])
rmse_3 = np.sqrt(mse_idx3)
mse_idx4 = mean_squared_error(y_test[bin_index_test==4], y_pred[bin_index_test==4])
rmse_4 = np.sqrt(mse_idx4)
mse_idx5 = mean_squared_error(y_test[bin_index_test==5], y_pred[bin_index_test==5])
rmse_5 = np.sqrt(mse_idx5)


print('----- Test_bins_RMSE  ------')
print(f'Root mean squared error of bins_1: {rmse_1:.2f}')
print(f'Root mean squared error of bins_2: {rmse_2:.2f}')
print(f'Root mean squared error of bins_3: {rmse_3:.2f}')
print(f'Root mean squared error of bins_4: {rmse_4:.2f}')
print(f'Root mean squared error of bins_5: {rmse_5:.2f}')


```

```python
#Different Error Estimation
    
y_pred_train = xgb.predict(X_train)
mae_train = mean_absolute_error(y_train, y_pred_train)
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
mx_train = max_error(y_train, y_pred_train)
me_train = (y_pred_train - y_train).sum()/len(y_train)

y_pred = xgb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mx = max_error(y_test, y_pred)
me = (y_pred - y_test).sum()/len(y_test)

print('----- Test  ------')
print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')
print(f'Max error: {mx:.2f}')
print(f"Average Error: {me:.2f}")

print('---- Training -----')
print(f'Mean absolute error: {mae_train:.2f}')
print(f'Mean squared error: {mse_train:.2f}')
print(f'Root mean squared error: {rmse_train:.2f}')
print(f'Max error: {mx_train:.2f}')
print(f"Average Error: {me_train:.2f}")
   
    
score = xgb.score(X_train, y_train)  
print("Training score coefficient of determination for the model R^2: %.3f " % (score))
```

```python

```

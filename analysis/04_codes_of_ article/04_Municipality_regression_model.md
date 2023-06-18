## Run XGBoost Regression Model for Municipality_based dataset


```python
%load_ext jupyter_black
```



<script type="application/javascript" id="jupyter_black">
(function() {
    if (window.IPython === undefined) {
        return
    }
    var msg = "WARNING: it looks like you might have loaded " +
        "jupyter_black in a non-lab notebook with " +
        "`is_lab=True`. Please double check, and if " +
        "loading with `%load_ext` please review the README!"
    console.log(msg)
    alert(msg)
})()
</script>




```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
import pandas as pd
import numpy as np
import statistics
import os

from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, max_error
from collections import defaultdict
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
```

    /Users/mersedehkooshki/opt/anaconda3/envs/global-storm/lib/python3.8/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      from pandas import MultiIndex, Int64Index



```python
# Import the CSV file to a dataframe
df = pd.read_csv("data/df_merged_2.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mun_Code</th>
      <th>typhoon</th>
      <th>HAZ_rainfall_Total</th>
      <th>HAZ_rainfall_max_6h</th>
      <th>HAZ_rainfall_max_24h</th>
      <th>HAZ_v_max</th>
      <th>HAZ_dis_track_min</th>
      <th>GEN_landslide_per</th>
      <th>GEN_stormsurge_per</th>
      <th>GEN_Bu_p_inSSA</th>
      <th>...</th>
      <th>VUL_LightRoof_LightWall</th>
      <th>VUL_LightRoof_SalvageWall</th>
      <th>VUL_SalvagedRoof_StrongWall</th>
      <th>VUL_SalvagedRoof_LightWall</th>
      <th>VUL_SalvagedRoof_SalvageWall</th>
      <th>VUL_vulnerable_groups</th>
      <th>VUL_pantawid_pamilya_beneficiary</th>
      <th>DAM_perc_dmg</th>
      <th>HAZ_v_max_3</th>
      <th>y_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH175101000</td>
      <td>DURIAN2006</td>
      <td>185.828571</td>
      <td>14.716071</td>
      <td>7.381696</td>
      <td>55.032241</td>
      <td>2.478142</td>
      <td>2.64</td>
      <td>6.18</td>
      <td>6.18</td>
      <td>...</td>
      <td>41.892832</td>
      <td>1.002088</td>
      <td>0.000000</td>
      <td>0.027836</td>
      <td>0.083507</td>
      <td>2.951511</td>
      <td>46.931106</td>
      <td>3.632568</td>
      <td>166667.757548</td>
      <td>3.34975</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH083701000</td>
      <td>DURIAN2006</td>
      <td>8.818750</td>
      <td>0.455208</td>
      <td>0.255319</td>
      <td>8.728380</td>
      <td>288.358553</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>13.645253</td>
      <td>0.549120</td>
      <td>0.030089</td>
      <td>0.090266</td>
      <td>0.112833</td>
      <td>3.338873</td>
      <td>25.989168</td>
      <td>0.000000</td>
      <td>664.968323</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH015501000</td>
      <td>DURIAN2006</td>
      <td>24.175000</td>
      <td>2.408333</td>
      <td>0.957639</td>
      <td>10.945624</td>
      <td>274.953818</td>
      <td>1.52</td>
      <td>1.28</td>
      <td>1.28</td>
      <td>...</td>
      <td>15.592295</td>
      <td>0.075838</td>
      <td>0.000000</td>
      <td>0.015168</td>
      <td>0.075838</td>
      <td>2.131755</td>
      <td>32.185651</td>
      <td>0.000000</td>
      <td>1311.358762</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH015502000</td>
      <td>DURIAN2006</td>
      <td>14.930000</td>
      <td>1.650000</td>
      <td>0.586250</td>
      <td>12.108701</td>
      <td>252.828578</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>7.100454</td>
      <td>0.023280</td>
      <td>0.011640</td>
      <td>0.000000</td>
      <td>0.128041</td>
      <td>1.589369</td>
      <td>29.612385</td>
      <td>0.000000</td>
      <td>1775.385328</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH175302000</td>
      <td>DURIAN2006</td>
      <td>13.550000</td>
      <td>1.054167</td>
      <td>0.528125</td>
      <td>10.660943</td>
      <td>258.194381</td>
      <td>5.52</td>
      <td>0.36</td>
      <td>0.36</td>
      <td>...</td>
      <td>30.354796</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032852</td>
      <td>0.000000</td>
      <td>1.387007</td>
      <td>35.052562</td>
      <td>0.000000</td>
      <td>1211.676901</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8068</th>
      <td>PH084823000</td>
      <td>NOUL2015</td>
      <td>9.700000</td>
      <td>0.408333</td>
      <td>0.216146</td>
      <td>8.136932</td>
      <td>277.107823</td>
      <td>1.80</td>
      <td>6.25</td>
      <td>6.25</td>
      <td>...</td>
      <td>32.492212</td>
      <td>0.311526</td>
      <td>0.031153</td>
      <td>0.155763</td>
      <td>0.031153</td>
      <td>2.827833</td>
      <td>31.308411</td>
      <td>0.000000</td>
      <td>538.743551</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8069</th>
      <td>PH015547000</td>
      <td>NOUL2015</td>
      <td>17.587500</td>
      <td>1.414583</td>
      <td>0.386458</td>
      <td>9.818999</td>
      <td>305.789817</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>4.703833</td>
      <td>0.027875</td>
      <td>0.000000</td>
      <td>0.034843</td>
      <td>0.097561</td>
      <td>1.073268</td>
      <td>12.766551</td>
      <td>0.000000</td>
      <td>946.676507</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8070</th>
      <td>PH025014000</td>
      <td>NOUL2015</td>
      <td>11.487500</td>
      <td>0.614583</td>
      <td>0.230319</td>
      <td>15.791907</td>
      <td>210.313249</td>
      <td>0.06</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>...</td>
      <td>3.063753</td>
      <td>0.022528</td>
      <td>0.000000</td>
      <td>0.067583</td>
      <td>0.022528</td>
      <td>1.140109</td>
      <td>9.348952</td>
      <td>0.000000</td>
      <td>3938.254316</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8071</th>
      <td>PH140127000</td>
      <td>NOUL2015</td>
      <td>11.600000</td>
      <td>1.400000</td>
      <td>0.412766</td>
      <td>13.867145</td>
      <td>218.189328</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>3.119093</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.837537</td>
      <td>21.928166</td>
      <td>0.000000</td>
      <td>2666.620370</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8072</th>
      <td>PH051612000</td>
      <td>NOUL2015</td>
      <td>32.305556</td>
      <td>1.744444</td>
      <td>1.210417</td>
      <td>15.647639</td>
      <td>219.542224</td>
      <td>4.15</td>
      <td>3.05</td>
      <td>3.05</td>
      <td>...</td>
      <td>36.191860</td>
      <td>0.280316</td>
      <td>0.010382</td>
      <td>0.031146</td>
      <td>0.103821</td>
      <td>2.518110</td>
      <td>31.634136</td>
      <td>0.000000</td>
      <td>3831.302757</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
<p>8073 rows × 40 columns</p>
</div>




```python
# display(combined_input_data)
df.hist(column="y_norm", figsize=(4, 3))
```




    array([[<AxesSubplot:title={'center':'y_norm'}>]], dtype=object)




    
![png](output_4_1.png)
    



```python
# The Old and New set of bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
plt.figure(figsize=(4, 3))
samples_per_bin2, binsP2 = np.histogram(df["y_norm"], bins=bins2)
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```




    [<matplotlib.lines.Line2D at 0x7fcd816c9940>]




    
![png](output_5_1.png)
    



```python
print(samples_per_bin2)
print(binsP2)
```

    [3606 2874 1141  388   64]
    [0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]



```python
df["y_norm"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    3606
    (9e-05, 1.0]       2874
    (1.0, 10.0]        1141
    (10.0, 50.0]        388
    (50.0, 101.0]        64
    Name: y_norm, dtype: int64




```python
bin_index2 = np.digitize(df["y_norm"], bins=binsP2)
```


```python
y_input_strat = bin_index2
```


```python
all_features = [
    "HAZ_rainfall_Total",
    "HAZ_rainfall_max_6h",
    "HAZ_rainfall_max_24h",
    "HAZ_v_max",
    "HAZ_v_max_3",
    "HAZ_dis_track_min",
    "GEN_landslide_per",
    "GEN_stormsurge_per",
    #'GEN_Bu_p_inSSA',
    #'GEN_Bu_p_LS',
    "GEN_Red_per_LSbldg",
    "GEN_Or_per_LSblg",
    "GEN_Yel_per_LSSAb",
    "GEN_RED_per_SSAbldg",
    #'GEN_OR_per_SSAbldg',
    "GEN_Yellow_per_LSbl",
    "TOP_mean_slope",
    "TOP_mean_elevation_m",
    "TOP_ruggedness_stdev",
    #'TOP_mean_ruggedness',
    #'TOP_slope_stdev',
    "VUL_poverty_perc",
    "GEN_with_coast",
    "GEN_coast_length",
    "VUL_Housing_Units",
    "VUL_StrongRoof_StrongWall",
    "VUL_StrongRoof_LightWall",
    "VUL_StrongRoof_SalvageWall",
    "VUL_LightRoof_StrongWall",
    "VUL_LightRoof_LightWall",
    "VUL_LightRoof_SalvageWall",
    "VUL_SalvagedRoof_StrongWall",
    "VUL_SalvagedRoof_LightWall",
    "VUL_SalvagedRoof_SalvageWall",
    "VUL_vulnerable_groups",
    "VUL_pantawid_pamilya_beneficiary",
]


glob_features = [  #'HAZ_rainfall_Total',
    "HAZ_rainfall_max_6h",
    "HAZ_rainfall_max_24h",
    "HAZ_v_max",
    #'HAZ_v_max_3',
    "HAZ_dis_track_min",
    #'GEN_landslide_per',
    #'GEN_stormsurge_per',
    #'GEN_Bu_p_inSSA',
    #'GEN_Bu_p_LS',
    #'GEN_Red_per_LSbldg',
    #'GEN_Or_per_LSblg',
    #'GEN_Yel_per_LSSAb',
    #'GEN_RED_per_SSAbldg',
    #'GEN_OR_per_SSAbldg',
    #'GEN_Yellow_per_LSbl',
    "TOP_mean_slope",
    "TOP_mean_elevation_m",
    "TOP_ruggedness_stdev",
    "TOP_mean_ruggedness",
    "TOP_slope_stdev",
    #'VUL_poverty_perc',
    "GEN_with_coast",
    "GEN_coast_length",
    "VUL_Housing_Units",
    #'VUL_StrongRoof_StrongWall',
    #'VUL_StrongRoof_LightWall',
    #'VUL_StrongRoof_SalvageWall',
    #'VUL_LightRoof_StrongWall',
    #'VUL_LightRoof_LightWall',
    #'VUL_LightRoof_SalvageWall',
    #'VUL_SalvagedRoof_StrongWall',
    #'VUL_SalvagedRoof_LightWall',
    #'VUL_SalvagedRoof_SalvageWall',
    #'VUL_vulnerable_groups',
    #'VUL_pantawid_pamilya_beneficiary',
]
```


```python
# Defin two lists to save total RMSE of test and train data

test_RMSE = defaultdict(list)
train_RMSE = defaultdict(list)

test_AVE = defaultdict(list)
```


```python
# Ask the user whether to use all features set or global features set
feature_set = int(input("Enter 1 for all features, 2 for global features: "))
```

    Enter 1 for all features, 2 for global features: 1



```python
if feature_set == 1:
    features = all_features
    print(len(features))

elif feature_set == 2:
    features = glob_features
    print(len(features))

else:
    print("Invalid input. Please enter 1 or 2")
```

    31



```python
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["y_norm"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```


    Index(['HAZ_rainfall_Total', 'HAZ_rainfall_max_6h', 'HAZ_rainfall_max_24h',
           'HAZ_v_max', 'HAZ_v_max_3', 'HAZ_dis_track_min', 'GEN_landslide_per',
           'GEN_stormsurge_per', 'GEN_Red_per_LSbldg', 'GEN_Or_per_LSblg',
           'GEN_Yel_per_LSSAb', 'GEN_RED_per_SSAbldg', 'GEN_Yellow_per_LSbl',
           'TOP_mean_slope', 'TOP_mean_elevation_m', 'TOP_ruggedness_stdev',
           'VUL_poverty_perc', 'GEN_with_coast', 'GEN_coast_length',
           'VUL_Housing_Units', 'VUL_StrongRoof_StrongWall',
           'VUL_StrongRoof_LightWall', 'VUL_StrongRoof_SalvageWall',
           'VUL_LightRoof_StrongWall', 'VUL_LightRoof_LightWall',
           'VUL_LightRoof_SalvageWall', 'VUL_SalvagedRoof_StrongWall',
           'VUL_SalvagedRoof_LightWall', 'VUL_SalvagedRoof_SalvageWall',
           'VUL_vulnerable_groups', 'VUL_pantawid_pamilya_beneficiary'],
          dtype='object')



```python
# Run XGBoost Reduced Overfitting in for loop to estimate RMSE per bins

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df["y_norm"], stratify=y_input_strat, test_size=0.2
    )

    # XGBoost Reduced Overfitting
    xgb = XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        colsample_bylevel=0.8,
        colsample_bynode=0.8,
        colsample_bytree=0.8,
        gamma=3,
        eta=0.01,
        importance_type="gain",
        learning_rate=0.1,
        max_delta_step=0,
        max_depth=4,
        min_child_weight=1,
        missing=1,
        n_estimators=100,
        early_stopping_rounds=10,
        n_jobs=1,
        nthread=None,
        objective="reg:squarederror",
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        seed=None,
        silent=None,
        subsample=0.8,
        verbosity=1,
        eval_metric=["rmse", "logloss"],
        random_state=0,
    )

    eval_set = [(X_train, y_train)]
    xgb_model = xgb.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
    )

    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    print(est2.summary())

    X2_test = sm.add_constant(X_test)

    y_pred_train_LREG = est2.predict(X2)
    mse_train_idx_LREG = mean_squared_error(y_train, y_pred_train_LREG)
    rmse_train_LREG = np.sqrt(mse_train_idx_LREG)

    ypred_LREG = est2.predict(X2_test)
    mse_idx_LREG = mean_squared_error(y_test, ypred_LREG)
    rmse_LREG = np.sqrt(mse_idx_LREG)

    print("----- Training ------")
    print(f"LREG Root mean squared error: {rmse_train_LREG:.2f}")
    print("----- Test ------")
    print(f"LREG Root mean squared error: {rmse_LREG:.2f}")

    # Calculate RMSE in total

    y_pred_train = xgb.predict(X_train)

    # Clip the predicted values to be within the range of zero to 100
    y_pred_train_clipped = y_pred_train.clip(0, 100)
    mse_train_idx = mean_squared_error(y_train, y_pred_train_clipped)
    rmse_train = np.sqrt(mse_train_idx)

    y_pred = xgb.predict(X_test)

    # Clip the predicted values to be within the range of zero to 100
    y_pred_clipped = y_pred.clip(0, 100)
    mse_idx = mean_squared_error(y_test, y_pred_clipped)
    rmse = np.sqrt(mse_idx)

    print("----- Training ------")
    print(f"Root mean squared error: {rmse_train:.2f}")

    print("----- Test ------")
    print(f"Root mean squared error: {rmse:.2f}")

    test_RMSE["all"].append(rmse)
    train_RMSE["all"].append(rmse_train)

    # Calculate Average Error in total

    ave = (y_pred_clipped - y_test).sum() / len(y_test)

    print("----- Test ------")
    print(f"Average Error: {ave:.2f}")

    test_AVE["all"].append(ave)

    # Calculate RMSE per bins

    bin_index_test = np.digitize(y_test, bins=binsP2)
    bin_index_train = np.digitize(y_train, bins=binsP2)

    for bin_num in range(1, 6):

        mse_train_idx = mean_squared_error(
            y_train[bin_index_train == bin_num],
            y_pred_train_clipped[bin_index_train == bin_num],
        )
        rmse_train = np.sqrt(mse_train_idx)

        mse_idx = mean_squared_error(
            y_test[bin_index_test == bin_num],
            y_pred_clipped[bin_index_test == bin_num],
        )
        rmse = np.sqrt(mse_idx)

        train_RMSE[bin_num].append(rmse_train)
        test_RMSE[bin_num].append(rmse)

        # Calculate Average Error per bins

        ave = (
            y_pred_clipped[bin_index_test == bin_num]
            - y_test[bin_index_test == bin_num]
        ).sum() / len(y_test[bin_index_test == bin_num])
        test_AVE[bin_num].append(ave)
```

    [15:22:58] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.556
    Model:                            OLS   Adj. R-squared:                  0.554
    Method:                 Least Squares   F-statistic:                     259.5
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:22:58   Log-Likelihood:                -19986.
    No. Observations:                6458   AIC:                         4.004e+04
    Df Residuals:                    6426   BIC:                         4.025e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.1938      0.067     32.902      0.000       2.063       2.325
    x1            -1.1759      0.301     -3.907      0.000      -1.766      -0.586
    x2             1.7087      0.223      7.672      0.000       1.272       2.145
    x3             0.3055      0.417      0.733      0.464      -0.511       1.123
    x4            -4.8173      0.254    -18.934      0.000      -5.316      -4.319
    x5             8.8821      0.173     51.430      0.000       8.544       9.221
    x6            -0.5346      0.155     -3.460      0.001      -0.837      -0.232
    x7            -4.1150     47.773     -0.086      0.931     -97.767      89.537
    x8            -0.0292      2.284     -0.013      0.990      -4.507       4.449
    x9             2.5484     28.323      0.090      0.928     -52.975      58.071
    x10           -0.0616      0.363     -0.169      0.865      -0.774       0.651
    x11            3.1836     38.600      0.082      0.934     -72.485      78.852
    x12            0.1204      2.284      0.053      0.958      -4.356       4.597
    x13           -0.0899      0.075     -1.193      0.233      -0.238       0.058
    x14           -0.0336      0.210     -0.160      0.873      -0.445       0.378
    x15           -0.2139      0.138     -1.554      0.120      -0.484       0.056
    x16            0.0153      0.145      0.106      0.916      -0.269       0.299
    x17            0.6523      0.206      3.159      0.002       0.248       1.057
    x18            0.0664      0.093      0.712      0.476      -0.116       0.249
    x19           -0.1654      0.084     -1.977      0.048      -0.329      -0.001
    x20           -0.1237      0.070     -1.757      0.079      -0.262       0.014
    x21           -1.4734      2.247     -0.656      0.512      -5.879       2.932
    x22           -1.0612      1.652     -0.642      0.521      -4.301       2.178
    x23            0.3843      0.111      3.457      0.001       0.166       0.602
    x24            0.0743      0.323      0.230      0.818      -0.559       0.707
    x25           -0.8925      1.514     -0.590      0.555      -3.860       2.075
    x26            0.0948      0.089      1.065      0.287      -0.080       0.269
    x27            0.0743      0.084      0.880      0.379      -0.091       0.240
    x28            0.2093      0.089      2.351      0.019       0.035       0.384
    x29            0.0651      0.093      0.704      0.482      -0.116       0.247
    x30            0.5945      0.111      5.369      0.000       0.377       0.812
    x31           -0.5627      0.192     -2.927      0.003      -0.940      -0.186
    ==============================================================================
    Omnibus:                     5594.203   Durbin-Watson:                   2.019
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           361140.208
    Skew:                           3.817   Prob(JB):                         0.00
    Kurtosis:                      38.830   Cond. No.                     2.29e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.29e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.34
    ----- Test ------
    LREG Root mean squared error: 5.52
    ----- Training ------
    Root mean squared error: 2.61
    ----- Test ------
    Root mean squared error: 4.44
    ----- Test ------
    Average Error: -0.12
    [15:22:58] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.540
    Model:                            OLS   Adj. R-squared:                  0.537
    Method:                 Least Squares   F-statistic:                     243.0
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:22:59   Log-Likelihood:                -20094.
    No. Observations:                6458   AIC:                         4.025e+04
    Df Residuals:                    6426   BIC:                         4.047e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.1952      0.068     32.368      0.000       2.062       2.328
    x1            -1.2423      0.310     -4.013      0.000      -1.849      -0.636
    x2             1.5729      0.227      6.923      0.000       1.127       2.018
    x3             0.4993      0.425      1.174      0.240      -0.334       1.333
    x4            -4.7440      0.261    -18.153      0.000      -5.256      -4.232
    x5             8.7222      0.177     49.405      0.000       8.376       9.068
    x6            -0.5604      0.158     -3.550      0.000      -0.870      -0.251
    x7           -43.5439     49.065     -0.887      0.375    -139.727      52.639
    x8             0.3618      2.371      0.153      0.879      -4.286       5.010
    x9            25.9680     29.089      0.893      0.372     -31.055      82.991
    x10            0.2337      0.373      0.626      0.531      -0.498       0.966
    x11           35.0356     39.643      0.884      0.377     -42.678     112.749
    x12           -0.2938      2.371     -0.124      0.901      -4.941       4.354
    x13           -0.1119      0.077     -1.458      0.145      -0.262       0.039
    x14           -0.0299      0.211     -0.141      0.888      -0.444       0.385
    x15           -0.2146      0.142     -1.512      0.131      -0.493       0.064
    x16           -0.0204      0.146     -0.140      0.889      -0.306       0.265
    x17            0.7760      0.217      3.576      0.000       0.351       1.201
    x18            0.2186      0.096      2.282      0.023       0.031       0.406
    x19           -0.1529      0.086     -1.783      0.075      -0.321       0.015
    x20           -0.1296      0.072     -1.809      0.070      -0.270       0.011
    x21           -0.8776      2.293     -0.383      0.702      -5.373       3.617
    x22           -0.6244      1.686     -0.370      0.711      -3.929       2.681
    x23            0.3797      0.111      3.414      0.001       0.162       0.598
    x24            0.1608      0.330      0.487      0.626      -0.486       0.807
    x25           -0.5316      1.545     -0.344      0.731      -3.560       2.497
    x26            0.1529      0.103      1.484      0.138      -0.049       0.355
    x27           -0.0768      0.087     -0.887      0.375      -0.247       0.093
    x28            0.2183      0.082      2.664      0.008       0.058       0.379
    x29            0.0435      0.094      0.461      0.645      -0.141       0.228
    x30            0.5212      0.113      4.607      0.000       0.299       0.743
    x31           -0.6599      0.200     -3.292      0.001      -1.053      -0.267
    ==============================================================================
    Omnibus:                     5928.111   Durbin-Watson:                   2.027
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           447808.367
    Skew:                           4.146   Prob(JB):                         0.00
    Kurtosis:                      42.943   Cond. No.                     2.30e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.3e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.43
    ----- Test ------
    LREG Root mean squared error: 5.16
    ----- Training ------
    Root mean squared error: 2.78
    ----- Test ------
    Root mean squared error: 3.97
    ----- Test ------
    Average Error: 0.03
    [15:22:59] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.562
    Model:                            OLS   Adj. R-squared:                  0.560
    Method:                 Least Squares   F-statistic:                     265.7
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:00   Log-Likelihood:                -19872.
    No. Observations:                6458   AIC:                         3.981e+04
    Df Residuals:                    6426   BIC:                         4.002e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.1768      0.066     33.230      0.000       2.048       2.305
    x1            -1.5293      0.296     -5.168      0.000      -2.109      -0.949
    x2             1.1427      0.224      5.106      0.000       0.704       1.581
    x3             1.1898      0.412      2.890      0.004       0.383       1.997
    x4            -4.9209      0.250    -19.689      0.000      -5.411      -4.431
    x5             8.9018      0.169     52.565      0.000       8.570       9.234
    x6            -0.6010      0.153     -3.930      0.000      -0.901      -0.301
    x7            -9.2681     46.482     -0.199      0.842    -100.388      81.852
    x8             0.2997      2.335      0.128      0.898      -4.278       4.877
    x9             5.6760     27.556      0.206      0.837     -48.344      59.696
    x10            0.0784      0.355      0.221      0.825      -0.617       0.773
    x11            7.3478     37.556      0.196      0.845     -66.275      80.971
    x12           -0.1873      2.334     -0.080      0.936      -4.763       4.388
    x13           -0.1325      0.076     -1.738      0.082      -0.282       0.017
    x14           -0.2960      0.207     -1.430      0.153      -0.702       0.110
    x15           -0.1536      0.138     -1.116      0.264      -0.423       0.116
    x16            0.1025      0.142      0.724      0.469      -0.175       0.380
    x17            0.6761      0.213      3.179      0.001       0.259       1.093
    x18            0.0640      0.092      0.697      0.486      -0.116       0.244
    x19           -0.1479      0.083     -1.787      0.074      -0.310       0.014
    x20           -0.1616      0.072     -2.237      0.025      -0.303      -0.020
    x21           -2.9799      2.224     -1.340      0.180      -7.340       1.381
    x22           -2.1237      1.636     -1.298      0.194      -5.330       1.083
    x23            0.3513      0.107      3.282      0.001       0.141       0.561
    x24           -0.0389      0.320     -0.122      0.903      -0.666       0.588
    x25           -1.9449      1.498     -1.298      0.194      -4.882       0.992
    x26            0.1159      0.088      1.314      0.189      -0.057       0.289
    x27           -0.0267      0.077     -0.348      0.728      -0.177       0.124
    x28            0.0734      0.078      0.937      0.349      -0.080       0.227
    x29            0.0247      0.090      0.274      0.784      -0.152       0.201
    x30            0.6705      0.108      6.203      0.000       0.459       0.882
    x31           -0.6528      0.201     -3.245      0.001      -1.047      -0.258
    ==============================================================================
    Omnibus:                     5407.403   Durbin-Watson:                   1.960
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           318383.562
    Skew:                           3.642   Prob(JB):                         0.00
    Kurtosis:                      36.618   Cond. No.                     2.26e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.26e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.25
    ----- Test ------
    LREG Root mean squared error: 5.87
    ----- Training ------
    Root mean squared error: 2.50
    ----- Test ------
    Root mean squared error: 5.14
    ----- Test ------
    Average Error: -0.12
    [15:23:00] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.550
    Model:                            OLS   Adj. R-squared:                  0.547
    Method:                 Least Squares   F-statistic:                     253.0
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:00   Log-Likelihood:                -20018.
    No. Observations:                6458   AIC:                         4.010e+04
    Df Residuals:                    6426   BIC:                         4.032e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2036      0.067     32.883      0.000       2.072       2.335
    x1            -1.0980      0.301     -3.648      0.000      -1.688      -0.508
    x2             1.7831      0.227      7.862      0.000       1.339       2.228
    x3             0.1651      0.419      0.394      0.694      -0.657       0.987
    x4            -4.8372      0.258    -18.782      0.000      -5.342      -4.332
    x5             8.8988      0.175     50.922      0.000       8.556       9.241
    x6            -0.5589      0.156     -3.594      0.000      -0.864      -0.254
    x7           -43.1767     47.569     -0.908      0.364    -136.427      50.073
    x8             0.2908      2.294      0.127      0.899      -4.207       4.788
    x9            25.6970     28.201      0.911      0.362     -29.586      80.980
    x10            0.2461      0.363      0.678      0.498      -0.466       0.958
    x11           34.7702     38.433      0.905      0.366     -40.572     110.112
    x12           -0.2146      2.296     -0.093      0.926      -4.715       4.286
    x13           -0.1277      0.077     -1.660      0.097      -0.278       0.023
    x14           -0.2807      0.210     -1.335      0.182      -0.693       0.132
    x15           -0.0563      0.140     -0.403      0.687      -0.330       0.218
    x16            0.1073      0.145      0.740      0.460      -0.177       0.392
    x17            0.7022      0.208      3.383      0.001       0.295       1.109
    x18            0.0877      0.095      0.925      0.355      -0.098       0.274
    x19           -0.1702      0.084     -2.016      0.044      -0.336      -0.005
    x20           -0.1295      0.070     -1.845      0.065      -0.267       0.008
    x21           -2.6691      2.303     -1.159      0.246      -7.183       1.845
    x22           -1.8647      1.693     -1.101      0.271      -5.184       1.455
    x23            0.3445      0.112      3.066      0.002       0.124       0.565
    x24           -0.1079      0.330     -0.327      0.744      -0.755       0.539
    x25           -1.7232      1.551     -1.111      0.267      -4.765       1.318
    x26            0.1368      0.101      1.351      0.177      -0.062       0.335
    x27            0.0646      0.083      0.774      0.439      -0.099       0.228
    x28            0.1905      0.089      2.140      0.032       0.016       0.365
    x29           -0.0636      0.100     -0.635      0.525      -0.260       0.133
    x30            0.5104      0.111      4.614      0.000       0.294       0.727
    x31           -0.5412      0.197     -2.746      0.006      -0.927      -0.155
    ==============================================================================
    Omnibus:                     6077.726   Durbin-Watson:                   1.955
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           503201.255
    Skew:                           4.291   Prob(JB):                         0.00
    Kurtosis:                      45.384   Cond. No.                     2.24e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.24e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.37
    ----- Test ------
    LREG Root mean squared error: 5.42
    ----- Training ------
    Root mean squared error: 2.64
    ----- Test ------
    Root mean squared error: 4.51
    ----- Test ------
    Average Error: -0.20
    [15:23:00] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.536
    Model:                            OLS   Adj. R-squared:                  0.534
    Method:                 Least Squares   F-statistic:                     239.9
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:01   Log-Likelihood:                -20155.
    No. Observations:                6458   AIC:                         4.037e+04
    Df Residuals:                    6426   BIC:                         4.059e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2153      0.068     32.366      0.000       2.081       2.349
    x1            -1.3576      0.313     -4.343      0.000      -1.970      -0.745
    x2             1.5181      0.230      6.593      0.000       1.067       1.970
    x3             0.6846      0.433      1.581      0.114      -0.164       1.533
    x4            -4.7895      0.265    -18.089      0.000      -5.309      -4.271
    x5             8.8292      0.179     49.291      0.000       8.478       9.180
    x6            -0.5706      0.161     -3.547      0.000      -0.886      -0.255
    x7           -42.2683     49.717     -0.850      0.395    -139.731      55.194
    x8             0.7521      2.515      0.299      0.765      -4.177       5.681
    x9            25.1961     29.475      0.855      0.393     -32.585      82.977
    x10            0.2828      0.379      0.746      0.455      -0.460       1.025
    x11           34.0183     40.169      0.847      0.397     -44.727     112.764
    x12           -0.6479      2.513     -0.258      0.797      -5.575       4.279
    x13           -0.1122      0.078     -1.443      0.149      -0.265       0.040
    x14           -0.2461      0.216     -1.138      0.255      -0.670       0.178
    x15           -0.1575      0.143     -1.104      0.270      -0.437       0.122
    x16            0.0915      0.148      0.619      0.536      -0.198       0.381
    x17            0.7986      0.221      3.620      0.000       0.366       1.231
    x18            0.1057      0.096      1.100      0.271      -0.083       0.294
    x19           -0.1129      0.084     -1.349      0.177      -0.277       0.051
    x20           -0.1131      0.074     -1.527      0.127      -0.258       0.032
    x21           -1.9809      2.325     -0.852      0.394      -6.539       2.577
    x22           -1.4226      1.710     -0.832      0.405      -4.774       1.929
    x23            0.3822      0.113      3.385      0.001       0.161       0.604
    x24           -0.0541      0.334     -0.162      0.871      -0.710       0.601
    x25           -1.2052      1.566     -0.770      0.441      -4.274       1.864
    x26            0.0643      0.092      0.699      0.485      -0.116       0.245
    x27           -0.0035      0.078     -0.044      0.965      -0.156       0.150
    x28            0.1623      0.090      1.812      0.070      -0.013       0.338
    x29            0.0051      0.095      0.054      0.957      -0.180       0.191
    x30            0.6405      0.114      5.611      0.000       0.417       0.864
    x31           -0.7144      0.210     -3.408      0.001      -1.125      -0.303
    ==============================================================================
    Omnibus:                     5955.352   Durbin-Watson:                   2.024
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           457391.643
    Skew:                           4.173   Prob(JB):                         0.00
    Kurtosis:                      43.375   Cond. No.                     2.32e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.32e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.48
    ----- Test ------
    LREG Root mean squared error: 4.92
    ----- Training ------
    Root mean squared error: 2.79
    ----- Test ------
    Root mean squared error: 4.01
    ----- Test ------
    Average Error: -0.06
    [15:23:01] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.542
    Model:                            OLS   Adj. R-squared:                  0.540
    Method:                 Least Squares   F-statistic:                     245.5
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:02   Log-Likelihood:                -20026.
    No. Observations:                6458   AIC:                         4.012e+04
    Df Residuals:                    6426   BIC:                         4.033e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2443      0.067     33.446      0.000       2.113       2.376
    x1            -1.4478      0.300     -4.833      0.000      -2.035      -0.861
    x2             1.6064      0.226      7.096      0.000       1.163       2.050
    x3             0.7084      0.418      1.694      0.090      -0.111       1.528
    x4            -4.6613      0.257    -18.119      0.000      -5.166      -4.157
    x5             8.7459      0.176     49.799      0.000       8.402       9.090
    x6            -0.5184      0.156     -3.328      0.001      -0.824      -0.213
    x7           -49.3045     47.826     -1.031      0.303    -143.060      44.451
    x8             0.3254      2.266      0.144      0.886      -4.117       4.768
    x9            29.3799     28.354      1.036      0.300     -26.203      84.963
    x10            0.3259      0.365      0.893      0.372      -0.389       1.041
    x11           39.6811     38.643      1.027      0.305     -36.071     115.433
    x12           -0.1986      2.266     -0.088      0.930      -4.642       4.244
    x13           -0.0964      0.076     -1.265      0.206      -0.246       0.053
    x14           -0.2725      0.210     -1.295      0.195      -0.685       0.140
    x15           -0.1345      0.138     -0.972      0.331      -0.406       0.137
    x16            0.1155      0.145      0.797      0.426      -0.169       0.400
    x17            0.5894      0.201      2.936      0.003       0.196       0.983
    x18            0.1254      0.094      1.329      0.184      -0.060       0.310
    x19           -0.1823      0.086     -2.132      0.033      -0.350      -0.015
    x20           -0.0899      0.072     -1.251      0.211      -0.231       0.051
    x21           -3.3676      2.269     -1.484      0.138      -7.815       1.080
    x22           -2.4222      1.668     -1.452      0.146      -5.692       0.847
    x23            0.2280      0.114      2.007      0.045       0.005       0.451
    x24           -0.1916      0.326     -0.588      0.557      -0.831       0.447
    x25           -2.1020      1.528     -1.376      0.169      -5.097       0.893
    x26            0.1150      0.099      1.158      0.247      -0.080       0.310
    x27            0.0865      0.084      1.031      0.303      -0.078       0.251
    x28            0.1178      0.080      1.471      0.141      -0.039       0.275
    x29            0.0843      0.095      0.891      0.373      -0.101       0.270
    x30            0.6632      0.109      6.058      0.000       0.449       0.878
    x31           -0.5467      0.186     -2.947      0.003      -0.910      -0.183
    ==============================================================================
    Omnibus:                     5842.473   Durbin-Watson:                   2.014
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           431188.475
    Skew:                           4.054   Prob(JB):                         0.00
    Kurtosis:                      42.201   Cond. No.                     2.27e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.27e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.38
    ----- Test ------
    LREG Root mean squared error: 5.39
    ----- Training ------
    Root mean squared error: 2.62
    ----- Test ------
    Root mean squared error: 4.53
    ----- Test ------
    Average Error: 0.18
    [15:23:02] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.534
    Model:                            OLS   Adj. R-squared:                  0.532
    Method:                 Least Squares   F-statistic:                     237.9
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:02   Log-Likelihood:                -20044.
    No. Observations:                6458   AIC:                         4.015e+04
    Df Residuals:                    6426   BIC:                         4.037e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2241      0.067     33.047      0.000       2.092       2.356
    x1            -1.5101      0.302     -5.002      0.000      -2.102      -0.918
    x2             1.3654      0.226      6.033      0.000       0.922       1.809
    x3             0.9851      0.419      2.350      0.019       0.163       1.807
    x4            -4.8891      0.256    -19.074      0.000      -5.392      -4.387
    x5             8.8568      0.177     50.096      0.000       8.510       9.203
    x6            -0.6180      0.155     -3.993      0.000      -0.921      -0.315
    x7             0.6433     47.494      0.014      0.989     -92.461      93.748
    x8             0.3578      2.452      0.146      0.884      -4.449       5.165
    x9            -0.2279     28.157     -0.008      0.994     -55.426      54.970
    x10           -0.0656      0.362     -0.181      0.856      -0.775       0.644
    x11           -0.7299     38.373     -0.019      0.985     -75.954      74.494
    x12           -0.2951      2.452     -0.120      0.904      -5.103       4.513
    x13           -0.1201      0.077     -1.559      0.119      -0.271       0.031
    x14            0.0766      0.214      0.358      0.720      -0.343       0.496
    x15           -0.2682      0.140     -1.912      0.056      -0.543       0.007
    x16           -0.0212      0.147     -0.144      0.885      -0.308       0.266
    x17            0.6627      0.223      2.977      0.003       0.226       1.099
    x18            0.1520      0.094      1.617      0.106      -0.032       0.336
    x19           -0.1570      0.083     -1.899      0.058      -0.319       0.005
    x20           -0.1504      0.073     -2.051      0.040      -0.294      -0.007
    x21           -3.2972      2.403     -1.372      0.170      -8.007       1.413
    x22           -2.4518      1.767     -1.388      0.165      -5.915       1.012
    x23            0.3985      0.112      3.557      0.000       0.179       0.618
    x24           -0.2680      0.345     -0.777      0.437      -0.944       0.408
    x25           -2.0581      1.617     -1.272      0.203      -5.229       1.113
    x26            0.1786      0.101      1.774      0.076      -0.019       0.376
    x27            0.0322      0.076      0.422      0.673      -0.117       0.182
    x28            0.0506      0.081      0.622      0.534      -0.109       0.210
    x29           -0.0546      0.096     -0.571      0.568      -0.242       0.133
    x30            0.5136      0.112      4.576      0.000       0.294       0.734
    x31           -0.5606      0.214     -2.620      0.009      -0.980      -0.141
    ==============================================================================
    Omnibus:                     5893.975   Durbin-Watson:                   2.015
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           435592.438
    Skew:                           4.114   Prob(JB):                         0.00
    Kurtosis:                      42.384   Cond. No.                     2.24e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.24e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.39
    ----- Test ------
    LREG Root mean squared error: 5.33
    ----- Training ------
    Root mean squared error: 2.62
    ----- Test ------
    Root mean squared error: 4.38
    ----- Test ------
    Average Error: 0.03
    [15:23:02] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.549
    Model:                            OLS   Adj. R-squared:                  0.547
    Method:                 Least Squares   F-statistic:                     252.5
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:03   Log-Likelihood:                -19919.
    No. Observations:                6458   AIC:                         3.990e+04
    Df Residuals:                    6426   BIC:                         4.012e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.1925      0.066     33.208      0.000       2.063       2.322
    x1            -1.3715      0.296     -4.639      0.000      -1.951      -0.792
    x2             1.4415      0.220      6.546      0.000       1.010       1.873
    x3             0.8097      0.409      1.980      0.048       0.008       1.611
    x4            -4.8877      0.253    -19.306      0.000      -5.384      -4.391
    x5             8.8785      0.174     51.051      0.000       8.538       9.219
    x6            -0.5439      0.153     -3.555      0.000      -0.844      -0.244
    x7            24.2238     47.884      0.506      0.613     -69.645     118.092
    x8            -0.2231      2.199     -0.101      0.919      -4.534       4.088
    x9           -14.2569     28.389     -0.502      0.616     -69.908      41.395
    x10           -0.2650      0.365     -0.725      0.468      -0.981       0.451
    x11          -19.6568     38.689     -0.508      0.611     -95.499      56.186
    x12            0.2844      2.201      0.129      0.897      -4.030       4.599
    x13           -0.0534      0.076     -0.702      0.482      -0.202       0.096
    x14           -0.3184      0.208     -1.533      0.125      -0.726       0.089
    x15           -0.0610      0.137     -0.445      0.656      -0.330       0.208
    x16            0.0613      0.142      0.431      0.667      -0.217       0.340
    x17            0.6558      0.245      2.674      0.008       0.175       1.136
    x18            0.1250      0.093      1.343      0.179      -0.058       0.308
    x19           -0.1416      0.082     -1.731      0.084      -0.302       0.019
    x20           -0.1973      0.083     -2.391      0.017      -0.359      -0.036
    x21           -2.9637      2.336     -1.269      0.205      -7.543       1.616
    x22           -2.2096      1.717     -1.287      0.198      -5.576       1.157
    x23            0.3853      0.111      3.462      0.001       0.167       0.603
    x24           -0.1342      0.332     -0.404      0.686      -0.785       0.517
    x25           -1.8617      1.573     -1.183      0.237      -4.946       1.223
    x26            0.0109      0.090      0.121      0.904      -0.165       0.187
    x27            0.0358      0.082      0.438      0.662      -0.125       0.196
    x28            0.1379      0.100      1.377      0.169      -0.058       0.334
    x29            0.0765      0.096      0.796      0.426      -0.112       0.265
    x30            0.6493      0.110      5.890      0.000       0.433       0.865
    x31           -0.5565      0.240     -2.317      0.021      -1.027      -0.086
    ==============================================================================
    Omnibus:                     5362.349   Durbin-Watson:                   1.956
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           306003.234
    Skew:                           3.604   Prob(JB):                         0.00
    Kurtosis:                      35.943   Cond. No.                     2.30e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.3e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.29
    ----- Test ------
    LREG Root mean squared error: 5.73
    ----- Training ------
    Root mean squared error: 2.64
    ----- Test ------
    Root mean squared error: 5.01
    ----- Test ------
    Average Error: -0.07
    [15:23:03] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.550
    Model:                            OLS   Adj. R-squared:                  0.548
    Method:                 Least Squares   F-statistic:                     253.3
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:04   Log-Likelihood:                -20054.
    No. Observations:                6458   AIC:                         4.017e+04
    Df Residuals:                    6426   BIC:                         4.039e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2598      0.067     33.533      0.000       2.128       2.392
    x1            -1.3268      0.306     -4.330      0.000      -1.928      -0.726
    x2             1.4967      0.227      6.592      0.000       1.052       1.942
    x3             0.7056      0.424      1.664      0.096      -0.125       1.537
    x4            -4.9498      0.260    -19.044      0.000      -5.459      -4.440
    x5             9.0900      0.178     51.105      0.000       8.741       9.439
    x6            -0.5568      0.157     -3.546      0.000      -0.865      -0.249
    x7            -1.0657     48.249     -0.022      0.982     -95.650      93.518
    x8             0.2350      2.488      0.094      0.925      -4.641       5.111
    x9             0.7203     28.605      0.025      0.980     -55.354      56.795
    x10           -0.0643      0.368     -0.175      0.861      -0.785       0.656
    x11            0.6820     38.984      0.017      0.986     -75.740      77.104
    x12           -0.1526      2.485     -0.061      0.951      -5.024       4.719
    x13           -0.0637      0.078     -0.816      0.414      -0.217       0.089
    x14            0.1688      0.213      0.792      0.429      -0.249       0.587
    x15           -0.2544      0.140     -1.814      0.070      -0.529       0.020
    x16           -0.2107      0.146     -1.442      0.149      -0.497       0.076
    x17            0.6949      0.222      3.135      0.002       0.260       1.129
    x18            0.1616      0.095      1.697      0.090      -0.025       0.348
    x19           -0.1951      0.086     -2.262      0.024      -0.364      -0.026
    x20           -0.1128      0.070     -1.611      0.107      -0.250       0.024
    x21           -3.4800      2.322     -1.498      0.134      -8.033       1.073
    x22           -2.4979      1.707     -1.463      0.143      -5.844       0.849
    x23            0.2870      0.112      2.565      0.010       0.068       0.506
    x24           -0.1294      0.333     -0.389      0.698      -0.782       0.523
    x25           -2.3485      1.564     -1.501      0.133      -5.415       0.718
    x26            0.1415      0.101      1.405      0.160      -0.056       0.339
    x27            0.0733      0.082      0.890      0.374      -0.088       0.235
    x28            0.0978      0.080      1.215      0.225      -0.060       0.256
    x29           -0.0260      0.094     -0.276      0.783      -0.211       0.159
    x30            0.6131      0.111      5.509      0.000       0.395       0.831
    x31           -0.5830      0.210     -2.780      0.005      -0.994      -0.172
    ==============================================================================
    Omnibus:                     5759.183   Durbin-Watson:                   2.031
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           425805.372
    Skew:                           3.955   Prob(JB):                         0.00
    Kurtosis:                      41.985   Cond. No.                     2.27e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.27e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.40
    ----- Test ------
    LREG Root mean squared error: 5.30
    ----- Training ------
    Root mean squared error: 2.63
    ----- Test ------
    Root mean squared error: 4.25
    ----- Test ------
    Average Error: 0.14
    [15:23:04] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.546
    Model:                            OLS   Adj. R-squared:                  0.544
    Method:                 Least Squares   F-statistic:                     249.0
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:04   Log-Likelihood:                -20131.
    No. Observations:                6458   AIC:                         4.033e+04
    Df Residuals:                    6426   BIC:                         4.054e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2285      0.068     32.680      0.000       2.095       2.362
    x1            -1.3695      0.312     -4.393      0.000      -1.981      -0.758
    x2             1.4319      0.232      6.174      0.000       0.977       1.887
    x3             0.7002      0.431      1.625      0.104      -0.145       1.545
    x4            -4.8243      0.261    -18.511      0.000      -5.335      -4.313
    x5             8.9520      0.178     50.202      0.000       8.602       9.302
    x6            -0.5946      0.158     -3.766      0.000      -0.904      -0.285
    x7           -13.4748     49.288     -0.273      0.785    -110.096      83.146
    x8             0.6154      2.432      0.253      0.800      -4.152       5.382
    x9             8.1194     29.221      0.278      0.781     -49.164      65.402
    x10            0.0200      0.375      0.053      0.957      -0.715       0.755
    x11           10.7571     39.823      0.270      0.787     -67.310      88.824
    x12           -0.5176      2.430     -0.213      0.831      -5.282       4.247
    x13           -0.0994      0.078     -1.274      0.203      -0.252       0.054
    x14           -0.1578      0.216     -0.730      0.466      -0.582       0.266
    x15           -0.1099      0.142     -0.773      0.440      -0.389       0.169
    x16            0.0429      0.148      0.290      0.772      -0.247       0.333
    x17            0.8287      0.221      3.749      0.000       0.395       1.262
    x18            0.0482      0.096      0.503      0.615      -0.140       0.236
    x19           -0.0796      0.086     -0.929      0.353      -0.248       0.088
    x20           -0.0921      0.081     -1.130      0.258      -0.252       0.068
    x21           -2.9188      2.384     -1.224      0.221      -7.592       1.754
    x22           -2.1474      1.753     -1.225      0.221      -5.584       1.289
    x23            0.4233      0.114      3.719      0.000       0.200       0.646
    x24           -0.1351      0.341     -0.396      0.692      -0.803       0.533
    x25           -1.8496      1.606     -1.152      0.249      -4.998       1.299
    x26            0.0266      0.091      0.294      0.769      -0.151       0.204
    x27            0.0524      0.085      0.613      0.540      -0.115       0.220
    x28            0.0309      0.082      0.378      0.705      -0.129       0.191
    x29            0.0676      0.099      0.681      0.496      -0.127       0.262
    x30            0.6247      0.113      5.534      0.000       0.403       0.846
    x31           -0.6652      0.212     -3.142      0.002      -1.080      -0.250
    ==============================================================================
    Omnibus:                     5642.995   Durbin-Watson:                   1.982
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           377919.644
    Skew:                           3.859   Prob(JB):                         0.00
    Kurtosis:                      39.673   Cond. No.                     2.30e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.3e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.46
    ----- Test ------
    LREG Root mean squared error: 5.02
    ----- Training ------
    Root mean squared error: 2.68
    ----- Test ------
    Root mean squared error: 4.19
    ----- Test ------
    Average Error: 0.07
    [15:23:04] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.549
    Model:                            OLS   Adj. R-squared:                  0.547
    Method:                 Least Squares   F-statistic:                     252.5
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:05   Log-Likelihood:                -20003.
    No. Observations:                6458   AIC:                         4.007e+04
    Df Residuals:                    6426   BIC:                         4.029e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2227      0.067     33.245      0.000       2.092       2.354
    x1            -1.1561      0.300     -3.853      0.000      -1.744      -0.568
    x2             1.7672      0.222      7.956      0.000       1.332       2.203
    x3             0.2355      0.415      0.568      0.570      -0.578       1.049
    x4            -4.8379      0.258    -18.777      0.000      -5.343      -4.333
    x5             8.9568      0.176     50.758      0.000       8.611       9.303
    x6            -0.5633      0.155     -3.634      0.000      -0.867      -0.259
    x7            27.9389     48.679      0.574      0.566     -67.489     123.367
    x8             0.7910      2.331      0.339      0.734      -3.778       5.360
    x9           -16.4089     28.859     -0.569      0.570     -72.982      40.164
    x10           -0.2796      0.370     -0.755      0.450      -1.005       0.446
    x11          -22.6775     39.331     -0.577      0.564     -99.780      54.425
    x12           -0.6880      2.330     -0.295      0.768      -5.257       3.881
    x13           -0.1461      0.076     -1.927      0.054      -0.295       0.003
    x14           -0.1847      0.211     -0.877      0.380      -0.597       0.228
    x15           -0.1516      0.138     -1.101      0.271      -0.421       0.118
    x16            0.0798      0.144      0.555      0.579      -0.202       0.362
    x17            0.8872      0.214      4.139      0.000       0.467       1.307
    x18            0.0749      0.094      0.794      0.427      -0.110       0.260
    x19           -0.0999      0.085     -1.172      0.241      -0.267       0.067
    x20           -0.1601      0.085     -1.874      0.061      -0.328       0.007
    x21           -1.8533      2.191     -0.846      0.398      -6.148       2.442
    x22           -1.2304      1.611     -0.764      0.445      -4.388       1.927
    x23            0.2862      0.109      2.633      0.008       0.073       0.499
    x24           -0.0769      0.316     -0.243      0.808      -0.696       0.542
    x25           -1.1723      1.475     -0.795      0.427      -4.064       1.719
    x26            0.1514      0.089      1.699      0.089      -0.023       0.326
    x27            0.0499      0.077      0.650      0.516      -0.101       0.200
    x28            0.1288      0.080      1.613      0.107      -0.028       0.285
    x29            0.1125      0.092      1.224      0.221      -0.068       0.293
    x30            0.5215      0.110      4.734      0.000       0.306       0.737
    x31           -0.7697      0.200     -3.846      0.000      -1.162      -0.377
    ==============================================================================
    Omnibus:                     5498.378   Durbin-Watson:                   1.982
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           341346.547
    Skew:                           3.724   Prob(JB):                         0.00
    Kurtosis:                      37.829   Cond. No.                     2.31e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.31e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.36
    ----- Test ------
    LREG Root mean squared error: 5.47
    ----- Training ------
    Root mean squared error: 2.61
    ----- Test ------
    Root mean squared error: 4.61
    ----- Test ------
    Average Error: 0.02
    [15:23:05] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.539
    Model:                            OLS   Adj. R-squared:                  0.537
    Method:                 Least Squares   F-statistic:                     242.7
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:06   Log-Likelihood:                -20070.
    No. Observations:                6458   AIC:                         4.020e+04
    Df Residuals:                    6426   BIC:                         4.042e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2237      0.068     32.913      0.000       2.091       2.356
    x1            -1.4431      0.305     -4.726      0.000      -2.042      -0.845
    x2             1.5611      0.229      6.819      0.000       1.112       2.010
    x3             0.7639      0.421      1.815      0.070      -0.061       1.589
    x4            -4.8190      0.258    -18.671      0.000      -5.325      -4.313
    x5             8.8357      0.177     50.056      0.000       8.490       9.182
    x6            -0.5723      0.157     -3.647      0.000      -0.880      -0.265
    x7           -25.9471     48.954     -0.530      0.596    -121.914      70.020
    x8             0.6043      2.456      0.246      0.806      -4.210       5.418
    x9            15.5173     29.023      0.535      0.593     -41.377      72.412
    x10            0.1238      0.373      0.332      0.740      -0.608       0.855
    x11           20.7700     39.554      0.525      0.600     -56.769      98.309
    x12           -0.5294      2.455     -0.216      0.829      -5.342       4.284
    x13           -0.1515      0.077     -1.980      0.048      -0.301      -0.001
    x14            0.0748      0.213      0.351      0.725      -0.343       0.492
    x15           -0.2670      0.141     -1.891      0.059      -0.544       0.010
    x16           -0.0402      0.145     -0.277      0.782      -0.325       0.245
    x17            0.6502      0.216      3.013      0.003       0.227       1.073
    x18            0.1270      0.095      1.336      0.182      -0.059       0.313
    x19           -0.0905      0.085     -1.069      0.285      -0.256       0.075
    x20           -0.1351      0.075     -1.799      0.072      -0.282       0.012
    x21           -3.1297      2.523     -1.241      0.215      -8.075       1.815
    x22           -2.2387      1.854     -1.208      0.227      -5.872       1.395
    x23            0.3644      0.117      3.122      0.002       0.136       0.593
    x24           -0.1091      0.358     -0.304      0.761      -0.812       0.594
    x25           -2.0562      1.699     -1.210      0.226      -5.387       1.275
    x26            0.0395      0.092      0.427      0.669      -0.142       0.221
    x27           -0.0448      0.084     -0.533      0.594      -0.210       0.120
    x28            0.1248      0.083      1.509      0.131      -0.037       0.287
    x29            0.0832      0.097      0.858      0.391      -0.107       0.273
    x30            0.5613      0.111      5.062      0.000       0.344       0.779
    x31           -0.5441      0.204     -2.672      0.008      -0.943      -0.145
    ==============================================================================
    Omnibus:                     6020.035   Durbin-Watson:                   1.970
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           485079.683
    Skew:                           4.231   Prob(JB):                         0.00
    Kurtosis:                      44.606   Cond. No.                     2.30e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.3e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.41
    ----- Test ------
    LREG Root mean squared error: 5.24
    ----- Training ------
    Root mean squared error: 2.67
    ----- Test ------
    Root mean squared error: 4.25
    ----- Test ------
    Average Error: 0.16
    [15:23:06] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.551
    Model:                            OLS   Adj. R-squared:                  0.549
    Method:                 Least Squares   F-statistic:                     254.5
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:06   Log-Likelihood:                -20009.
    No. Observations:                6458   AIC:                         4.008e+04
    Df Residuals:                    6426   BIC:                         4.030e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2238      0.067     33.231      0.000       2.093       2.355
    x1            -1.2223      0.299     -4.083      0.000      -1.809      -0.635
    x2             1.6308      0.226      7.215      0.000       1.188       2.074
    x3             0.4492      0.419      1.072      0.284      -0.372       1.270
    x4            -4.8872      0.256    -19.102      0.000      -5.389      -4.386
    x5             8.9072      0.174     51.084      0.000       8.565       9.249
    x6            -0.5869      0.155     -3.797      0.000      -0.890      -0.284
    x7           -15.0813     48.007     -0.314      0.753    -109.190      79.028
    x8             0.0570      2.380      0.024      0.981      -4.609       4.723
    x9             9.0458     28.461      0.318      0.751     -46.748      64.839
    x10            0.0397      0.366      0.109      0.913      -0.677       0.756
    x11           12.0730     38.788      0.311      0.756     -63.964      88.111
    x12            0.0208      2.379      0.009      0.993      -4.643       4.684
    x13           -0.0887      0.077     -1.154      0.249      -0.239       0.062
    x14           -0.2682      0.211     -1.274      0.203      -0.681       0.145
    x15           -0.0968      0.138     -0.704      0.482      -0.366       0.173
    x16            0.1045      0.145      0.720      0.471      -0.180       0.389
    x17            0.7993      0.218      3.669      0.000       0.372       1.226
    x18            0.1355      0.094      1.439      0.150      -0.049       0.320
    x19           -0.1208      0.082     -1.464      0.143      -0.282       0.041
    x20           -0.1192      0.072     -1.645      0.100      -0.261       0.023
    x21           -2.0306      2.261     -0.898      0.369      -6.463       2.401
    x22           -1.5038      1.662     -0.905      0.366      -4.762       1.755
    x23            0.3967      0.111      3.567      0.000       0.179       0.615
    x24           -0.0346      0.324     -0.107      0.915      -0.670       0.601
    x25           -1.2969      1.524     -0.851      0.395      -4.283       1.690
    x26            0.1355      0.090      1.511      0.131      -0.040       0.311
    x27           -0.0390      0.082     -0.476      0.634      -0.200       0.122
    x28            0.1452      0.080      1.823      0.068      -0.011       0.301
    x29            0.0715      0.098      0.731      0.465      -0.120       0.263
    x30            0.5736      0.109      5.264      0.000       0.360       0.787
    x31           -0.6327      0.208     -3.049      0.002      -1.039      -0.226
    ==============================================================================
    Omnibus:                     5526.658   Durbin-Watson:                   2.016
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           339625.597
    Skew:                           3.760   Prob(JB):                         0.00
    Kurtosis:                      37.722   Cond. No.                     2.30e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.3e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.36
    ----- Test ------
    LREG Root mean squared error: 5.44
    ----- Training ------
    Root mean squared error: 2.66
    ----- Test ------
    Root mean squared error: 4.40
    ----- Test ------
    Average Error: -0.00
    [15:23:06] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.554
    Model:                            OLS   Adj. R-squared:                  0.552
    Method:                 Least Squares   F-statistic:                     257.2
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:07   Log-Likelihood:                -19974.
    No. Observations:                6458   AIC:                         4.001e+04
    Df Residuals:                    6426   BIC:                         4.023e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2002      0.067     33.055      0.000       2.070       2.331
    x1            -1.3567      0.307     -4.415      0.000      -1.959      -0.754
    x2             1.5630      0.221      7.079      0.000       1.130       1.996
    x3             0.6147      0.421      1.461      0.144      -0.210       1.439
    x4            -4.8587      0.255    -19.087      0.000      -5.358      -4.360
    x5             8.9663      0.173     51.726      0.000       8.627       9.306
    x6            -0.5333      0.155     -3.445      0.001      -0.837      -0.230
    x7           -59.5924     46.898     -1.271      0.204    -151.528      32.343
    x8             0.3443      2.503      0.138      0.891      -4.563       5.252
    x9            35.4631     27.804      1.275      0.202     -19.043      89.969
    x10            0.3837      0.357      1.074      0.283      -0.317       1.084
    x11           48.0459     37.893      1.268      0.205     -26.236     122.328
    x12           -0.2244      2.500     -0.090      0.928      -5.125       4.677
    x13           -0.1155      0.077     -1.499      0.134      -0.267       0.036
    x14           -0.3641      0.209     -1.738      0.082      -0.775       0.047
    x15           -0.0820      0.138     -0.593      0.553      -0.353       0.189
    x16            0.1897      0.143      1.327      0.185      -0.091       0.470
    x17            0.7744      0.207      3.743      0.000       0.369       1.180
    x18            0.0968      0.094      1.028      0.304      -0.088       0.281
    x19           -0.1928      0.084     -2.296      0.022      -0.357      -0.028
    x20           -0.1286      0.081     -1.592      0.111      -0.287       0.030
    x21           -2.2791      2.288     -0.996      0.319      -6.764       2.206
    x22           -1.5881      1.681     -0.944      0.345      -4.884       1.708
    x23            0.2711      0.112      2.423      0.015       0.052       0.490
    x24           -0.0792      0.327     -0.242      0.809      -0.721       0.562
    x25           -1.3909      1.541     -0.902      0.367      -4.412       1.631
    x26            0.0800      0.091      0.877      0.380      -0.099       0.259
    x27           -0.0111      0.076     -0.146      0.884      -0.160       0.138
    x28            0.1196      0.080      1.499      0.134      -0.037       0.276
    x29            0.0886      0.094      0.945      0.344      -0.095       0.272
    x30            0.4787      0.110      4.349      0.000       0.263       0.694
    x31           -0.6005      0.192     -3.134      0.002      -0.976      -0.225
    ==============================================================================
    Omnibus:                     5677.783   Durbin-Watson:                   2.022
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           398472.069
    Skew:                           3.881   Prob(JB):                         0.00
    Kurtosis:                      40.691   Cond. No.                     2.24e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.24e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.33
    ----- Test ------
    LREG Root mean squared error: 5.56
    ----- Training ------
    Root mean squared error: 2.66
    ----- Test ------
    Root mean squared error: 4.31
    ----- Test ------
    Average Error: -0.13
    [15:23:07] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.544
    Model:                            OLS   Adj. R-squared:                  0.541
    Method:                 Least Squares   F-statistic:                     247.0
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:08   Log-Likelihood:                -20072.
    No. Observations:                6458   AIC:                         4.021e+04
    Df Residuals:                    6426   BIC:                         4.042e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2259      0.068     32.939      0.000       2.093       2.358
    x1            -1.3325      0.304     -4.380      0.000      -1.929      -0.736
    x2             1.4559      0.226      6.449      0.000       1.013       1.898
    x3             0.7561      0.424      1.784      0.074      -0.075       1.587
    x4            -4.7140      0.258    -18.237      0.000      -5.221      -4.207
    x5             8.7935      0.176     50.022      0.000       8.449       9.138
    x6            -0.5179      0.157     -3.293      0.001      -0.826      -0.210
    x7           -28.0631     48.123     -0.583      0.560    -122.399      66.273
    x8            -0.5152      2.364     -0.218      0.828      -5.150       4.120
    x9            16.7625     28.530      0.588      0.557     -39.165      72.690
    x10            0.1707      0.367      0.465      0.642      -0.550       0.891
    x11           22.5258     38.882      0.579      0.562     -53.695      98.747
    x12            0.5894      2.365      0.249      0.803      -4.047       5.226
    x13           -0.1567      0.078     -2.020      0.043      -0.309      -0.005
    x14           -0.1521      0.215     -0.709      0.479      -0.573       0.269
    x15           -0.1419      0.141     -1.009      0.313      -0.418       0.134
    x16           -0.0104      0.148     -0.070      0.944      -0.300       0.279
    x17            0.8066      0.220      3.665      0.000       0.375       1.238
    x18            0.1891      0.095      1.991      0.047       0.003       0.375
    x19           -0.1406      0.086     -1.627      0.104      -0.310       0.029
    x20           -0.1714      0.072     -2.392      0.017      -0.312      -0.031
    x21           -3.4557      2.299     -1.503      0.133      -7.962       1.050
    x22           -2.5263      1.689     -1.495      0.135      -5.838       0.785
    x23            0.3068      0.111      2.770      0.006       0.090       0.524
    x24           -0.2178      0.330     -0.661      0.509      -0.864       0.428
    x25           -2.2841      1.549     -1.475      0.140      -5.320       0.752
    x26            0.1347      0.091      1.477      0.140      -0.044       0.313
    x27           -0.0119      0.079     -0.152      0.879      -0.166       0.142
    x28            0.1369      0.090      1.524      0.128      -0.039       0.313
    x29            0.0043      0.092      0.047      0.963      -0.176       0.184
    x30            0.6526      0.112      5.830      0.000       0.433       0.872
    x31           -0.7157      0.207     -3.457      0.001      -1.122      -0.310
    ==============================================================================
    Omnibus:                     6007.360   Durbin-Watson:                   2.011
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           487251.408
    Skew:                           4.213   Prob(JB):                         0.00
    Kurtosis:                      44.711   Cond. No.                     2.28e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.28e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.41
    ----- Test ------
    LREG Root mean squared error: 5.23
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 4.38
    ----- Test ------
    Average Error: 0.06
    [15:23:08] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.555
    Model:                            OLS   Adj. R-squared:                  0.553
    Method:                 Least Squares   F-statistic:                     258.6
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:09   Log-Likelihood:                -19935.
    No. Observations:                6458   AIC:                         3.993e+04
    Df Residuals:                    6426   BIC:                         4.015e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.1997      0.066     33.248      0.000       2.070       2.329
    x1            -1.2120      0.299     -4.055      0.000      -1.798      -0.626
    x2             1.7593      0.223      7.898      0.000       1.323       2.196
    x3             0.3193      0.414      0.772      0.440      -0.491       1.130
    x4            -4.9629      0.255    -19.491      0.000      -5.462      -4.464
    x5             8.9480      0.172     52.013      0.000       8.611       9.285
    x6            -0.6092      0.155     -3.927      0.000      -0.913      -0.305
    x7            -3.3402     47.729     -0.070      0.944     -96.905      90.224
    x8             0.0573      2.349      0.024      0.981      -4.548       4.663
    x9             2.0870     28.297      0.074      0.941     -53.384      57.558
    x10           -0.0578      0.363     -0.159      0.873      -0.769       0.653
    x11            2.6187     38.564      0.068      0.946     -72.979      78.216
    x12            0.0740      2.348      0.032      0.975      -4.530       4.678
    x13           -0.1099      0.075     -1.461      0.144      -0.257       0.038
    x14           -0.0092      0.207     -0.045      0.964      -0.416       0.397
    x15           -0.2416      0.137     -1.763      0.078      -0.510       0.027
    x16            0.0030      0.144      0.021      0.984      -0.279       0.285
    x17            0.8615      0.212      4.073      0.000       0.447       1.276
    x18            0.0847      0.093      0.914      0.361      -0.097       0.267
    x19           -0.1830      0.082     -2.240      0.025      -0.343      -0.023
    x20           -0.1359      0.075     -1.824      0.068      -0.282       0.010
    x21           -1.6873      2.265     -0.745      0.456      -6.127       2.753
    x22           -1.0585      1.665     -0.636      0.525      -4.323       2.206
    x23            0.3071      0.109      2.822      0.005       0.094       0.520
    x24            0.0498      0.326      0.153      0.878      -0.589       0.688
    x25           -1.0892      1.525     -0.714      0.475      -4.079       1.901
    x26            0.1300      0.098      1.321      0.187      -0.063       0.323
    x27            0.0713      0.080      0.895      0.371      -0.085       0.228
    x28            0.1293      0.088      1.476      0.140      -0.042       0.301
    x29            0.0594      0.091      0.649      0.516      -0.120       0.239
    x30            0.3979      0.109      3.645      0.000       0.184       0.612
    x31           -0.6376      0.201     -3.169      0.002      -1.032      -0.243
    ==============================================================================
    Omnibus:                     5876.848   Durbin-Watson:                   2.012
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           463286.361
    Skew:                           4.068   Prob(JB):                         0.00
    Kurtosis:                      43.688   Cond. No.                     2.27e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.27e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.30
    ----- Test ------
    LREG Root mean squared error: 5.68
    ----- Training ------
    Root mean squared error: 2.59
    ----- Test ------
    Root mean squared error: 4.72
    ----- Test ------
    Average Error: -0.04
    [15:23:09] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.536
    Model:                            OLS   Adj. R-squared:                  0.533
    Method:                 Least Squares   F-statistic:                     239.1
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:09   Log-Likelihood:                -20041.
    No. Observations:                6458   AIC:                         4.015e+04
    Df Residuals:                    6426   BIC:                         4.036e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2265      0.067     33.108      0.000       2.095       2.358
    x1            -1.3527      0.300     -4.516      0.000      -1.940      -0.766
    x2             1.1616      0.226      5.142      0.000       0.719       1.604
    x3             0.9825      0.415      2.367      0.018       0.169       1.796
    x4            -4.6341      0.258    -17.953      0.000      -5.140      -4.128
    x5             8.6957      0.177     49.242      0.000       8.350       9.042
    x6            -0.5586      0.156     -3.588      0.000      -0.864      -0.253
    x7            10.8848     48.382      0.225      0.822     -83.960     105.730
    x8            -0.3332      2.287     -0.146      0.884      -4.817       4.150
    x9            -6.2750     28.683     -0.219      0.827     -62.504      49.954
    x10           -0.1566      0.368     -0.426      0.670      -0.877       0.564
    x11           -8.9451     39.091     -0.229      0.819     -85.576      67.686
    x12            0.4466      2.287      0.195      0.845      -4.037       4.930
    x13           -0.1159      0.077     -1.501      0.133      -0.267       0.035
    x14           -0.1387      0.213     -0.651      0.515      -0.556       0.279
    x15           -0.1722      0.139     -1.240      0.215      -0.445       0.100
    x16            0.0135      0.146      0.093      0.926      -0.272       0.299
    x17            0.7646      0.214      3.567      0.000       0.344       1.185
    x18            0.1177      0.094      1.248      0.212      -0.067       0.303
    x19           -0.0900      0.085     -1.055      0.292      -0.257       0.077
    x20           -0.1169      0.069     -1.683      0.093      -0.253       0.019
    x21           -3.4888      2.392     -1.458      0.145      -8.178       1.201
    x22           -2.4780      1.758     -1.409      0.159      -5.925       0.969
    x23            0.3425      0.111      3.073      0.002       0.124       0.561
    x24           -0.2894      0.342     -0.847      0.397      -0.959       0.380
    x25           -2.2092      1.611     -1.372      0.170      -5.367       0.948
    x26            0.0860      0.090      0.952      0.341      -0.091       0.263
    x27            0.0536      0.076      0.709      0.479      -0.095       0.202
    x28            0.1220      0.088      1.381      0.167      -0.051       0.295
    x29           -0.0556      0.096     -0.579      0.563      -0.244       0.133
    x30            0.4804      0.112      4.293      0.000       0.261       0.700
    x31           -0.6222      0.202     -3.078      0.002      -1.018      -0.226
    ==============================================================================
    Omnibus:                     6194.586   Durbin-Watson:                   1.966
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           546224.329
    Skew:                           4.409   Prob(JB):                         0.00
    Kurtosis:                      47.183   Cond. No.                     2.28e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.28e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.39
    ----- Test ------
    LREG Root mean squared error: 5.34
    ----- Training ------
    Root mean squared error: 2.64
    ----- Test ------
    Root mean squared error: 4.36
    ----- Test ------
    Average Error: -0.02
    [15:23:09] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.544
    Model:                            OLS   Adj. R-squared:                  0.542
    Method:                 Least Squares   F-statistic:                     247.8
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:10   Log-Likelihood:                -20029.
    No. Observations:                6458   AIC:                         4.012e+04
    Df Residuals:                    6426   BIC:                         4.034e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2269      0.067     33.172      0.000       2.095       2.359
    x1            -1.5810      0.307     -5.142      0.000      -2.184      -0.978
    x2             1.5382      0.226      6.796      0.000       1.095       1.982
    x3             0.8741      0.423      2.069      0.039       0.046       1.702
    x4            -4.9268      0.257    -19.138      0.000      -5.431      -4.422
    x5             8.9831      0.176     50.951      0.000       8.637       9.329
    x6            -0.5835      0.156     -3.744      0.000      -0.889      -0.278
    x7           -54.6254     47.960     -1.139      0.255    -148.644      39.393
    x8             0.5195      2.341      0.222      0.824      -4.069       5.108
    x9            32.4813     28.433      1.142      0.253     -23.257      88.220
    x10            0.3699      0.366      1.012      0.312      -0.347       1.087
    x11           43.9864     38.750      1.135      0.256     -31.976     119.949
    x12           -0.4185      2.341     -0.179      0.858      -5.007       4.170
    x13           -0.1054      0.077     -1.367      0.172      -0.257       0.046
    x14           -0.1754      0.211     -0.831      0.406      -0.589       0.238
    x15           -0.1221      0.139     -0.878      0.380      -0.395       0.150
    x16            0.0786      0.145      0.542      0.588      -0.206       0.363
    x17            0.8095      0.216      3.753      0.000       0.387       1.232
    x18            0.1021      0.095      1.078      0.281      -0.084       0.288
    x19           -0.1396      0.086     -1.630      0.103      -0.308       0.028
    x20           -0.1370      0.073     -1.870      0.061      -0.281       0.007
    x21           -1.6257      2.292     -0.709      0.478      -6.119       2.868
    x22           -1.1324      1.686     -0.672      0.502      -4.437       2.172
    x23            0.2541      0.112      2.264      0.024       0.034       0.474
    x24            0.0405      0.329      0.123      0.902      -0.605       0.686
    x25           -1.0051      1.544     -0.651      0.515      -4.032       2.021
    x26            0.2302      0.092      2.504      0.012       0.050       0.410
    x27           -0.0470      0.079     -0.593      0.553      -0.202       0.108
    x28            0.1775      0.088      2.018      0.044       0.005       0.350
    x29            0.0616      0.098      0.631      0.528      -0.130       0.253
    x30            0.5020      0.111      4.514      0.000       0.284       0.720
    x31           -0.6776      0.206     -3.283      0.001      -1.082      -0.273
    ==============================================================================
    Omnibus:                     5788.603   Durbin-Watson:                   1.979
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           425785.892
    Skew:                           3.991   Prob(JB):                         0.00
    Kurtosis:                      41.970   Cond. No.                     2.27e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.27e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.38
    ----- Test ------
    LREG Root mean squared error: 5.38
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 4.52
    ----- Test ------
    Average Error: 0.13
    [15:23:10] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.547
    Model:                            OLS   Adj. R-squared:                  0.544
    Method:                 Least Squares   F-statistic:                     249.8
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:11   Log-Likelihood:                -19910.
    No. Observations:                6458   AIC:                         3.988e+04
    Df Residuals:                    6426   BIC:                         4.010e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.1938      0.066     33.281      0.000       2.065       2.323
    x1            -1.1502      0.295     -3.904      0.000      -1.728      -0.573
    x2             1.6127      0.222      7.263      0.000       1.177       2.048
    x3             0.3810      0.413      0.922      0.356      -0.429       1.191
    x4            -4.7214      0.255    -18.522      0.000      -5.221      -4.222
    x5             8.7763      0.174     50.387      0.000       8.435       9.118
    x6            -0.5025      0.153     -3.278      0.001      -0.803      -0.202
    x7           -12.1515     46.983     -0.259      0.796    -104.253      79.950
    x8             0.5997      2.462      0.244      0.808      -4.227       5.427
    x9             7.2786     27.855      0.261      0.794     -47.326      61.883
    x10            0.0683      0.358      0.191      0.849      -0.634       0.771
    x11            9.6601     37.960      0.254      0.799     -64.754      84.075
    x12           -0.5263      2.459     -0.214      0.831      -5.348       4.295
    x13           -0.1022      0.076     -1.353      0.176      -0.250       0.046
    x14           -0.0668      0.209     -0.320      0.749      -0.477       0.343
    x15           -0.2309      0.137     -1.685      0.092      -0.500       0.038
    x16            0.0506      0.143      0.354      0.723      -0.230       0.331
    x17            0.6171      0.204      3.023      0.003       0.217       1.017
    x18            0.0170      0.092      0.185      0.854      -0.164       0.198
    x19           -0.0658      0.083     -0.791      0.429      -0.229       0.097
    x20           -0.1389      0.070     -1.995      0.046      -0.275      -0.002
    x21           -2.0825      2.345     -0.888      0.374      -6.679       2.514
    x22           -1.5201      1.725     -0.881      0.378      -4.901       1.861
    x23            0.2748      0.111      2.472      0.013       0.057       0.493
    x24           -0.0159      0.334     -0.048      0.962      -0.672       0.640
    x25           -1.3053      1.579     -0.827      0.408      -4.401       1.790
    x26            0.0597      0.089      0.672      0.501      -0.114       0.234
    x27           -0.0295      0.081     -0.364      0.716      -0.188       0.129
    x28            0.1236      0.079      1.555      0.120      -0.032       0.279
    x29            0.1625      0.095      1.716      0.086      -0.023       0.348
    x30            0.6349      0.110      5.750      0.000       0.418       0.851
    x31           -0.5623      0.193     -2.916      0.004      -0.940      -0.184
    ==============================================================================
    Omnibus:                     5827.292   Durbin-Watson:                   1.990
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           440935.056
    Skew:                           4.026   Prob(JB):                         0.00
    Kurtosis:                      42.672   Cond. No.                     2.27e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.27e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.28
    ----- Test ------
    LREG Root mean squared error: 5.76
    ----- Training ------
    Root mean squared error: 2.59
    ----- Test ------
    Root mean squared error: 4.46
    ----- Test ------
    Average Error: -0.06
    [15:23:11] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.532
    Model:                            OLS   Adj. R-squared:                  0.530
    Method:                 Least Squares   F-statistic:                     235.6
    Date:                Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                        15:23:11   Log-Likelihood:                -19900.
    No. Observations:                6458   AIC:                         3.986e+04
    Df Residuals:                    6426   BIC:                         4.008e+04
    Df Model:                          31                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2099      0.066     33.581      0.000       2.081       2.339
    x1            -1.2941      0.295     -4.380      0.000      -1.873      -0.715
    x2             1.4410      0.221      6.512      0.000       1.007       1.875
    x3             0.6052      0.410      1.477      0.140      -0.198       1.409
    x4            -4.4692      0.253    -17.669      0.000      -4.965      -3.973
    x5             8.5067      0.174     48.796      0.000       8.165       8.848
    x6            -0.5444      0.152     -3.570      0.000      -0.843      -0.245
    x7           -20.4182     46.414     -0.440      0.660    -111.405      70.568
    x8             1.0544      2.283      0.462      0.644      -3.420       5.529
    x9            12.2293     27.517      0.444      0.657     -41.714      66.172
    x10            0.1176      0.355      0.331      0.740      -0.578       0.814
    x11           16.3318     37.501      0.436      0.663     -57.183      89.846
    x12           -0.9633      2.282     -0.422      0.673      -5.437       3.510
    x13           -0.1275      0.075     -1.699      0.089      -0.275       0.020
    x14           -0.4221      0.208     -2.033      0.042      -0.829      -0.015
    x15           -0.0665      0.139     -0.477      0.633      -0.340       0.207
    x16            0.2215      0.142      1.559      0.119      -0.057       0.500
    x17            0.4971      0.217      2.292      0.022       0.072       0.922
    x18            0.0573      0.093      0.616      0.538      -0.125       0.240
    x19           -0.1257      0.084     -1.498      0.134      -0.290       0.039
    x20           -0.1062      0.071     -1.505      0.132      -0.244       0.032
    x21           -3.6805      2.250     -1.636      0.102      -8.092       0.731
    x22           -2.6957      1.654     -1.630      0.103      -5.937       0.546
    x23            0.2690      0.111      2.415      0.016       0.051       0.487
    x24           -0.3168      0.322     -0.983      0.326      -0.949       0.315
    x25           -2.3264      1.516     -1.535      0.125      -5.298       0.645
    x26            0.1998      0.088      2.272      0.023       0.027       0.372
    x27           -0.0175      0.076     -0.231      0.817      -0.166       0.131
    x28            0.1473      0.086      1.709      0.088      -0.022       0.316
    x29           -0.0571      0.091     -0.626      0.531      -0.236       0.122
    x30            0.5909      0.108      5.457      0.000       0.379       0.803
    x31           -0.3751      0.210     -1.783      0.075      -0.787       0.037
    ==============================================================================
    Omnibus:                     5951.104   Durbin-Watson:                   2.024
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           473666.589
    Skew:                           4.153   Prob(JB):                         0.00
    Kurtosis:                      44.125   Cond. No.                     2.25e+03
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 2.25e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.27
    ----- Test ------
    LREG Root mean squared error: 5.80
    ----- Training ------
    Root mean squared error: 2.61
    ----- Test ------
    Root mean squared error: 4.87
    ----- Test ------
    Average Error: -0.10



```python
# Define a function to plot RMSEs
def rmse_bin_plot(te_rmse, tr_rmse, te_ave, min_rg, max_rg, step):

    m_test_rmse = statistics.mean(te_rmse)
    plt.figure(figsize=(4, 3))
    plt.axvline(m_test_rmse, color="b", linestyle="dashed")
    plt.hist(
        te_rmse,
        bins=np.arange(min_rg, max_rg, step),
        edgecolor="k",
        histtype="bar",
        density=True,
    )
    sd_test_rmse = statistics.stdev(te_rmse)

    m_train_rmse = statistics.mean(tr_rmse)
    plt.axvline(m_train_rmse, color="red", linestyle="dashed")
    plt.hist(
        tr_rmse,
        bins=np.arange(min_rg, max_rg, step),
        color="orange",
        edgecolor="k",
        histtype="bar",
        density=True,
        alpha=0.7,
    )
    sd_train_rmse = statistics.stdev(tr_rmse)

    # Average Error
    m_test_ave = statistics.mean(te_ave)
    sd_test_ave = statistics.stdev(te_ave)

    print(f"mean_RMSE_test: {m_test_rmse:.2f}")
    # print(f"mean_RMSE_train: {m_train_rmse:.2f}")

    print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
    # print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

    print(f"mean_AVE_test: {m_test_ave:.2f}")

    print(f"stdev_AVE_test: {sd_test_ave:.2f}")

    # create legend
    labels = ["Mean_test", "Mean_train", "test", "train"]
    plt.legend(labels)

    plt.xlabel("The RMSE error")
    plt.ylabel("Frequency")
    plt.title("histogram of the RMSE distribution")
    plt.show()
```


```python
# RMSE in Total

print("RMSE in total", "\n")
rmse_bin_plot(test_RMSE["all"], train_RMSE["all"], test_AVE["all"], 0.5, 7.0, 0.45)
```

    RMSE in total 
    
    mean_RMSE_test: 4.47
    stdev_RMSE_test: 0.30
    mean_AVE_test: -0.01
    stdev_AVE_test: 0.11



    
![png](output_17_1.png)
    



```python
bin_params = {
    1: (0.05, 1.0, 0.07),
    2: (0.5, 2.5, 0.1),
    3: (0.5, 5, 0.35),
    4: (-6.0, 16.0, 1.5),
    5: (-21.0, 35.0, 4),
}


for bin_num in range(1, 6):

    print(f"RMSE per bin {bin_num}\n")
    rmse_bin_plot(
        test_RMSE[bin_num], train_RMSE[bin_num], test_AVE[bin_num], *bin_params[bin_num]
    )
```

    RMSE per bin 1
    
    mean_RMSE_test: 0.27
    stdev_RMSE_test: 0.08
    mean_AVE_test: 0.08
    stdev_AVE_test: 0.01



    
![png](output_18_1.png)
    


    RMSE per bin 2
    
    mean_RMSE_test: 2.10
    stdev_RMSE_test: 0.28
    mean_AVE_test: 0.88
    stdev_AVE_test: 0.07



    
![png](output_18_3.png)
    


    RMSE per bin 3
    
    mean_RMSE_test: 4.45
    stdev_RMSE_test: 0.51
    mean_AVE_test: 0.91
    stdev_AVE_test: 0.35



    
![png](output_18_5.png)
    


    RMSE per bin 4
    
    mean_RMSE_test: 13.54
    stdev_RMSE_test: 0.85
    mean_AVE_test: -6.42
    stdev_AVE_test: 1.37



    
![png](output_18_7.png)
    


    RMSE per bin 5
    
    mean_RMSE_test: 28.39
    stdev_RMSE_test: 5.05
    mean_AVE_test: -21.31
    stdev_AVE_test: 4.73



    
![png](output_18_9.png)
    



```python

```

# Converting from grid-based to municipality-based

The following steps were done to convert created grid_based dataset to municipality_based one:

import the weight file to a dataframe

assign the values to each grid 

multiply the damaged values with the weights

Aggregate the values by municipality and typhoon_name 


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
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils import get_training_dataset, weight_file
```

    /Users/mersedehkooshki/opt/anaconda3/envs/global-storm/lib/python3.8/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      from pandas import MultiIndex, Int64Index



```python
# Import the created dataset to a df
df = get_training_dataset()
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
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>percent_houses_damaged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>101</td>
      <td>0.0</td>
      <td>303.180555</td>
      <td>0.122917</td>
      <td>0.085417</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4475</td>
      <td>0.0</td>
      <td>638.027502</td>
      <td>0.091667</td>
      <td>0.027083</td>
      <td>3.301020</td>
      <td>-0.527000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4639</td>
      <td>0.0</td>
      <td>603.631997</td>
      <td>0.535417</td>
      <td>0.146354</td>
      <td>12.103741</td>
      <td>-0.283000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4640</td>
      <td>0.0</td>
      <td>614.675270</td>
      <td>0.356250</td>
      <td>0.101562</td>
      <td>645.899660</td>
      <td>-0.358889</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4641</td>
      <td>0.0</td>
      <td>625.720905</td>
      <td>0.202083</td>
      <td>0.057812</td>
      <td>1071.731293</td>
      <td>-0.462800</td>
      <td>0.0</td>
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
    </tr>
    <tr>
      <th>141253</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20677</td>
      <td>0.0</td>
      <td>644.575831</td>
      <td>2.543750</td>
      <td>0.778646</td>
      <td>4449.357133</td>
      <td>0.508167</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>141254</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20678</td>
      <td>0.0</td>
      <td>655.685233</td>
      <td>2.558333</td>
      <td>0.861458</td>
      <td>1521.435795</td>
      <td>-0.174100</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>141255</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20679</td>
      <td>0.0</td>
      <td>666.794635</td>
      <td>2.975000</td>
      <td>0.949479</td>
      <td>930.647069</td>
      <td>-0.244286</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>141256</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20680</td>
      <td>0.0</td>
      <td>677.904037</td>
      <td>2.889583</td>
      <td>1.083333</td>
      <td>1800.666044</td>
      <td>0.038000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>141257</th>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20681</td>
      <td>0.0</td>
      <td>689.013439</td>
      <td>2.985417</td>
      <td>2.218056</td>
      <td>373.146778</td>
      <td>-0.175000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>141258 rows × 10 columns</p>
</div>




```python
# Check values greater than 100
(df["percent_houses_damaged"] > 100).sum()

for i in range(len(df)):
    a = df.loc[i, "percent_houses_damaged"]
    if a > 100:
        print(a)
```

    251.474732044657
    100.25359097066338
    100.36104807097172



```python
# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```


```python
# Read the new weight CSV file and import to a df
df_weight = weight_file("/ggl_grid_to_mun_weights.csv")
df_weight.head()
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
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>id</th>
      <th>numbuildings</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012801000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>1052</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.586399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012810000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>0</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012815000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>742</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.413601</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012801000</td>
      <td>11050.0</td>
      <td>120.9E_18.4N</td>
      <td>193</td>
      <td>11050</td>
      <td>196</td>
      <td>0.984694</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012810000</td>
      <td>11050.0</td>
      <td>120.9E_18.4N</td>
      <td>0</td>
      <td>11050</td>
      <td>196</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Check if two grid_id columns are equal first need to convert 'id_x' columns from float to int

# df_weight["id_x"] = df_weight["id_x"].astype(np.int64)
# df_weight["id_x"].equals(df_weight["id"])
```


```python
# Change name of column ['id'] to ['grid_point_id'] the same name as in input df
df_weight.rename(columns={"id": "grid_point_id"}, inplace=True)
df_weight.head()
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
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>grid_point_id</th>
      <th>numbuildings</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012801000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>1052</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.586399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012810000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>0</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012815000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>742</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.413601</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012801000</td>
      <td>11050.0</td>
      <td>120.9E_18.4N</td>
      <td>193</td>
      <td>11050</td>
      <td>196</td>
      <td>0.984694</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012810000</td>
      <td>11050.0</td>
      <td>120.9E_18.4N</td>
      <td>0</td>
      <td>11050</td>
      <td>196</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Following Steps are to convert grid_based model into Municipality based one


```python
# Hist plot after data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```




    [<matplotlib.lines.Line2D at 0x7fe9b1a344f0>]




    
![png](output_10_1.png)
    



```python
# Check the bins' intervalls
df["percent_houses_damaged"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    129600
    (9e-05, 1.0]         7938
    (1.0, 10.0]          2634
    (10.0, 50.0]          939
    (50.0, 101.0]         147
    Name: percent_houses_damaged, dtype: int64




```python
# Remove zeros from wind_speed
# df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df_data = df.drop(columns=["grid_point_id", "typhoon_year"])
```


```python
display(df.head())
display(df_data.head())
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
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>percent_houses_damaged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>101</td>
      <td>0.0</td>
      <td>303.180555</td>
      <td>0.122917</td>
      <td>0.085417</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4475</td>
      <td>0.0</td>
      <td>638.027502</td>
      <td>0.091667</td>
      <td>0.027083</td>
      <td>3.301020</td>
      <td>-0.527000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4639</td>
      <td>0.0</td>
      <td>603.631997</td>
      <td>0.535417</td>
      <td>0.146354</td>
      <td>12.103741</td>
      <td>-0.283000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4640</td>
      <td>0.0</td>
      <td>614.675270</td>
      <td>0.356250</td>
      <td>0.101562</td>
      <td>645.899660</td>
      <td>-0.358889</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4641</td>
      <td>0.0</td>
      <td>625.720905</td>
      <td>0.202083</td>
      <td>0.057812</td>
      <td>1071.731293</td>
      <td>-0.462800</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



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
      <th>typhoon_name</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>percent_houses_damaged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>303.180555</td>
      <td>0.122917</td>
      <td>0.085417</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>638.027502</td>
      <td>0.091667</td>
      <td>0.027083</td>
      <td>3.301020</td>
      <td>-0.527000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>603.631997</td>
      <td>0.535417</td>
      <td>0.146354</td>
      <td>12.103741</td>
      <td>-0.283000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>614.675270</td>
      <td>0.356250</td>
      <td>0.101562</td>
      <td>645.899660</td>
      <td>-0.358889</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>625.720905</td>
      <td>0.202083</td>
      <td>0.057812</td>
      <td>1071.731293</td>
      <td>-0.462800</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
# Hist plot after removing rows where windspeed is 0
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df_data["percent_houses_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```




    [<matplotlib.lines.Line2D at 0x7fe9a07a89d0>]




    
![png](output_14_1.png)
    



```python
print(samples_per_bin2)
print(binsP2)
```

    [129600   7938   2634    939    147]
    [0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]



```python
# Check the bins' intervalls
df_data["percent_houses_damaged"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    129600
    (9e-05, 1.0]         7938
    (1.0, 10.0]          2634
    (10.0, 50.0]          939
    (50.0, 101.0]         147
    Name: percent_houses_damaged, dtype: int64




```python
bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
```


```python
y_input_strat = bin_index2
```


```python
features = [
    "wind_speed",
    # "track_distance",
    # "total_houses",
    # "rainfall_max_6h",
    # "rainfall_max_24h"
    # "rwi"
]

# Split X and y from dataframe features
X = df_data[features]
display(X.columns)
y = df_data["percent_houses_damaged"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```


    Index(['wind_speed'], dtype='object')



```python
# Define train-test-split function

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df_data["percent_houses_damaged"],
    stratify=y_input_strat,
    test_size=0.2,
)

# Define XGBoost Reduced Overfitting model
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

eval_set = [(X_test, y_test)]
xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())
```

    [14:49:14] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.123
    Model:                                OLS   Adj. R-squared:                  0.123
    Method:                     Least Squares   F-statistic:                 1.579e+04
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            14:49:15   Log-Likelihood:            -2.7894e+05
    No. Observations:                  113006   AIC:                         5.579e+05
    Df Residuals:                      113004   BIC:                         5.579e+05
    Df Model:                               1                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.2994      0.008     35.236      0.000       0.283       0.316
    x1             1.0681      0.008    125.671      0.000       1.051       1.085
    ==============================================================================
    Omnibus:                   220403.974   Durbin-Watson:                   2.002
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):        524940600.485
    Skew:                          15.648   Prob(JB):                         0.00
    Kurtosis:                     335.425   Cond. No.                         1.00
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



```python
# Combine test and training set to have predictiob for all
y_pred_all = xgb.predict(X_scaled)
pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])
pred_df["y_all"] = df_data["percent_houses_damaged"]
pred_df["y_pred_all"] = y_pred_all

pred_df
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
      <th>y_all</th>
      <th>y_pred_all</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>141253</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>141254</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>141255</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>141256</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
    <tr>
      <th>141257</th>
      <td>0.0</td>
      <td>0.020261</td>
    </tr>
  </tbody>
</table>
<p>141258 rows × 2 columns</p>
</div>




```python
# Join data with y_all and y_all_pred
df_data_w_pred = pd.merge(pred_df, df_data, left_index=True, right_index=True)
# Join data with grid_point_id typhoon_year
df_data_w_pred_grid = pd.merge(
    df[["grid_point_id", "typhoon_year"]],
    df_data_w_pred,
    left_index=True,
    right_index=True,
)
```


```python
df_data_w_pred_grid.sort_values("y_pred_all", ascending=False)
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
      <th>grid_point_id</th>
      <th>typhoon_year</th>
      <th>y_all</th>
      <th>y_pred_all</th>
      <th>typhoon_name</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>percent_houses_damaged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>32064</th>
      <td>18303</td>
      <td>2013</td>
      <td>57.235598</td>
      <td>79.558975</td>
      <td>HAIYAN</td>
      <td>72.251930</td>
      <td>31.753148</td>
      <td>9.260417</td>
      <td>5.607813</td>
      <td>2383.683635</td>
      <td>0.308800</td>
      <td>57.235598</td>
    </tr>
    <tr>
      <th>32121</th>
      <td>18470</td>
      <td>2013</td>
      <td>94.285260</td>
      <td>79.558975</td>
      <td>HAIYAN</td>
      <td>72.231420</td>
      <td>33.730735</td>
      <td>8.783333</td>
      <td>5.559896</td>
      <td>251.827606</td>
      <td>-0.666000</td>
      <td>94.285260</td>
    </tr>
    <tr>
      <th>32379</th>
      <td>19307</td>
      <td>2013</td>
      <td>80.748749</td>
      <td>78.423950</td>
      <td>HAIYAN</td>
      <td>73.230522</td>
      <td>21.753512</td>
      <td>9.266667</td>
      <td>5.344792</td>
      <td>1625.734579</td>
      <td>-0.096571</td>
      <td>80.748749</td>
    </tr>
    <tr>
      <th>32122</th>
      <td>18471</td>
      <td>2013</td>
      <td>95.240679</td>
      <td>78.423950</td>
      <td>HAIYAN</td>
      <td>73.246757</td>
      <td>22.798157</td>
      <td>8.610417</td>
      <td>6.385937</td>
      <td>2786.766802</td>
      <td>0.045714</td>
      <td>95.240679</td>
    </tr>
    <tr>
      <th>31892</th>
      <td>17803</td>
      <td>2013</td>
      <td>78.666474</td>
      <td>70.301445</td>
      <td>HAIYAN</td>
      <td>72.451338</td>
      <td>14.887811</td>
      <td>8.708333</td>
      <td>4.740104</td>
      <td>11310.366765</td>
      <td>-0.212280</td>
      <td>78.666474</td>
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
    </tr>
    <tr>
      <th>58571</th>
      <td>10889</td>
      <td>2014</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>LINGLING</td>
      <td>0.000000</td>
      <td>1149.656762</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>290.917874</td>
      <td>-0.411714</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>58570</th>
      <td>10888</td>
      <td>2014</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>LINGLING</td>
      <td>0.000000</td>
      <td>1158.054191</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>9.442258</td>
      <td>NaN</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>58569</th>
      <td>10887</td>
      <td>2014</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>LINGLING</td>
      <td>0.000000</td>
      <td>1166.496984</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1197.958689</td>
      <td>-0.354000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>58568</th>
      <td>10886</td>
      <td>2014</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>LINGLING</td>
      <td>0.000000</td>
      <td>1174.984163</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5664.484019</td>
      <td>-0.089571</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>141257</th>
      <td>20681</td>
      <td>2020</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.000000</td>
      <td>689.013439</td>
      <td>2.985417</td>
      <td>2.218056</td>
      <td>373.146778</td>
      <td>-0.175000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>141258 rows × 12 columns</p>
</div>




```python
# join with weights df
join_df = df_data_w_pred_grid.merge(df_weight, on="grid_point_id", how="left")
join_df
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
      <th>grid_point_id</th>
      <th>typhoon_year</th>
      <th>y_all</th>
      <th>y_pred_all</th>
      <th>typhoon_name</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>percent_houses_damaged</th>
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>101</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>303.180555</td>
      <td>0.122917</td>
      <td>0.085417</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>PH175321000</td>
      <td>101.0</td>
      <td>114.3E_11.1N</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4475</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>638.027502</td>
      <td>0.091667</td>
      <td>0.027083</td>
      <td>3.301020</td>
      <td>-0.527000</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4475.0</td>
      <td>116.9E_7.9N</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4639</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>603.631997</td>
      <td>0.535417</td>
      <td>0.146354</td>
      <td>12.103741</td>
      <td>-0.283000</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4639.0</td>
      <td>117.0E_8.2N</td>
      <td>11</td>
      <td>11</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4640</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>614.675270</td>
      <td>0.356250</td>
      <td>0.101562</td>
      <td>645.899660</td>
      <td>-0.358889</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4640.0</td>
      <td>117.0E_8.1N</td>
      <td>587</td>
      <td>587</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4641</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>625.720905</td>
      <td>0.202083</td>
      <td>0.057812</td>
      <td>1071.731293</td>
      <td>-0.462800</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4641.0</td>
      <td>117.0E_8.0N</td>
      <td>974</td>
      <td>974</td>
      <td>1.000000</td>
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
    </tr>
    <tr>
      <th>381415</th>
      <td>20679</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>666.794635</td>
      <td>2.975000</td>
      <td>0.949479</td>
      <td>930.647069</td>
      <td>-0.244286</td>
      <td>0.0</td>
      <td>PH112501000</td>
      <td>20679.0</td>
      <td>126.6E_7.4N</td>
      <td>459</td>
      <td>1484</td>
      <td>0.309299</td>
    </tr>
    <tr>
      <th>381416</th>
      <td>20679</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>666.794635</td>
      <td>2.975000</td>
      <td>0.949479</td>
      <td>930.647069</td>
      <td>-0.244286</td>
      <td>0.0</td>
      <td>PH112504000</td>
      <td>20679.0</td>
      <td>126.6E_7.4N</td>
      <td>1025</td>
      <td>1484</td>
      <td>0.690701</td>
    </tr>
    <tr>
      <th>381417</th>
      <td>20680</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>677.904037</td>
      <td>2.889583</td>
      <td>1.083333</td>
      <td>1800.666044</td>
      <td>0.038000</td>
      <td>0.0</td>
      <td>PH112504000</td>
      <td>20680.0</td>
      <td>126.6E_7.3N</td>
      <td>2708</td>
      <td>2798</td>
      <td>0.967834</td>
    </tr>
    <tr>
      <th>381418</th>
      <td>20680</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>677.904037</td>
      <td>2.889583</td>
      <td>1.083333</td>
      <td>1800.666044</td>
      <td>0.038000</td>
      <td>0.0</td>
      <td>PH112508000</td>
      <td>20680.0</td>
      <td>126.6E_7.3N</td>
      <td>90</td>
      <td>2798</td>
      <td>0.032166</td>
    </tr>
    <tr>
      <th>381419</th>
      <td>20681</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>689.013439</td>
      <td>2.985417</td>
      <td>2.218056</td>
      <td>373.146778</td>
      <td>-0.175000</td>
      <td>0.0</td>
      <td>PH112508000</td>
      <td>20681.0</td>
      <td>126.6E_7.2N</td>
      <td>468</td>
      <td>468</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>381420 rows × 18 columns</p>
</div>




```python
# Indicate where values are valid and not missing
join_df = join_df.loc[join_df["weight"].notna()]
join_df
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
      <th>grid_point_id</th>
      <th>typhoon_year</th>
      <th>y_all</th>
      <th>y_pred_all</th>
      <th>typhoon_name</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>percent_houses_damaged</th>
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>101</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>303.180555</td>
      <td>0.122917</td>
      <td>0.085417</td>
      <td>31.000000</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>PH175321000</td>
      <td>101.0</td>
      <td>114.3E_11.1N</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4475</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>638.027502</td>
      <td>0.091667</td>
      <td>0.027083</td>
      <td>3.301020</td>
      <td>-0.527000</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4475.0</td>
      <td>116.9E_7.9N</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4639</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>603.631997</td>
      <td>0.535417</td>
      <td>0.146354</td>
      <td>12.103741</td>
      <td>-0.283000</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4639.0</td>
      <td>117.0E_8.2N</td>
      <td>11</td>
      <td>11</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4640</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>614.675270</td>
      <td>0.356250</td>
      <td>0.101562</td>
      <td>645.899660</td>
      <td>-0.358889</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4640.0</td>
      <td>117.0E_8.1N</td>
      <td>587</td>
      <td>587</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4641</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>625.720905</td>
      <td>0.202083</td>
      <td>0.057812</td>
      <td>1071.731293</td>
      <td>-0.462800</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4641.0</td>
      <td>117.0E_8.0N</td>
      <td>974</td>
      <td>974</td>
      <td>1.000000</td>
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
    </tr>
    <tr>
      <th>381415</th>
      <td>20679</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>666.794635</td>
      <td>2.975000</td>
      <td>0.949479</td>
      <td>930.647069</td>
      <td>-0.244286</td>
      <td>0.0</td>
      <td>PH112501000</td>
      <td>20679.0</td>
      <td>126.6E_7.4N</td>
      <td>459</td>
      <td>1484</td>
      <td>0.309299</td>
    </tr>
    <tr>
      <th>381416</th>
      <td>20679</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>666.794635</td>
      <td>2.975000</td>
      <td>0.949479</td>
      <td>930.647069</td>
      <td>-0.244286</td>
      <td>0.0</td>
      <td>PH112504000</td>
      <td>20679.0</td>
      <td>126.6E_7.4N</td>
      <td>1025</td>
      <td>1484</td>
      <td>0.690701</td>
    </tr>
    <tr>
      <th>381417</th>
      <td>20680</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>677.904037</td>
      <td>2.889583</td>
      <td>1.083333</td>
      <td>1800.666044</td>
      <td>0.038000</td>
      <td>0.0</td>
      <td>PH112504000</td>
      <td>20680.0</td>
      <td>126.6E_7.3N</td>
      <td>2708</td>
      <td>2798</td>
      <td>0.967834</td>
    </tr>
    <tr>
      <th>381418</th>
      <td>20680</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>677.904037</td>
      <td>2.889583</td>
      <td>1.083333</td>
      <td>1800.666044</td>
      <td>0.038000</td>
      <td>0.0</td>
      <td>PH112508000</td>
      <td>20680.0</td>
      <td>126.6E_7.3N</td>
      <td>90</td>
      <td>2798</td>
      <td>0.032166</td>
    </tr>
    <tr>
      <th>381419</th>
      <td>20681</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>689.013439</td>
      <td>2.985417</td>
      <td>2.218056</td>
      <td>373.146778</td>
      <td>-0.175000</td>
      <td>0.0</td>
      <td>PH112508000</td>
      <td>20681.0</td>
      <td>126.6E_7.2N</td>
      <td>468</td>
      <td>468</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>381420 rows × 18 columns</p>
</div>




```python
# Multiply weight by y_all and y_pred_all
join_df["weight*y_pred*houses"] = (
    join_df["y_pred_all"] * join_df["weight"] * join_df["total_houses"] / 100
)
join_df["weight*y*houses"] = (
    join_df["y_all"] * join_df["weight"] * join_df["total_houses"] / 100
)
join_df["weight*houses"] = join_df["weight"] * join_df["total_houses"]
join_df
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
      <th>grid_point_id</th>
      <th>typhoon_year</th>
      <th>y_all</th>
      <th>y_pred_all</th>
      <th>typhoon_name</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>...</th>
      <th>percent_houses_damaged</th>
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
      <th>weight</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>weight*houses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>101</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>303.180555</td>
      <td>0.122917</td>
      <td>0.085417</td>
      <td>31.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH175321000</td>
      <td>101.0</td>
      <td>114.3E_11.1N</td>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>0.006281</td>
      <td>0.0</td>
      <td>31.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4475</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>638.027502</td>
      <td>0.091667</td>
      <td>0.027083</td>
      <td>3.301020</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4475.0</td>
      <td>116.9E_7.9N</td>
      <td>3</td>
      <td>3</td>
      <td>1.000000</td>
      <td>0.000669</td>
      <td>0.0</td>
      <td>3.301020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4639</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>603.631997</td>
      <td>0.535417</td>
      <td>0.146354</td>
      <td>12.103741</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4639.0</td>
      <td>117.0E_8.2N</td>
      <td>11</td>
      <td>11</td>
      <td>1.000000</td>
      <td>0.002452</td>
      <td>0.0</td>
      <td>12.103741</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4640</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>614.675270</td>
      <td>0.356250</td>
      <td>0.101562</td>
      <td>645.899660</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4640.0</td>
      <td>117.0E_8.1N</td>
      <td>587</td>
      <td>587</td>
      <td>1.000000</td>
      <td>0.130868</td>
      <td>0.0</td>
      <td>645.899660</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4641</td>
      <td>2006</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>DURIAN</td>
      <td>0.0</td>
      <td>625.720905</td>
      <td>0.202083</td>
      <td>0.057812</td>
      <td>1071.731293</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH175304000</td>
      <td>4641.0</td>
      <td>117.0E_8.0N</td>
      <td>974</td>
      <td>974</td>
      <td>1.000000</td>
      <td>0.217148</td>
      <td>0.0</td>
      <td>1071.731293</td>
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
      <th>381415</th>
      <td>20679</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>666.794635</td>
      <td>2.975000</td>
      <td>0.949479</td>
      <td>930.647069</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH112501000</td>
      <td>20679.0</td>
      <td>126.6E_7.4N</td>
      <td>459</td>
      <td>1484</td>
      <td>0.309299</td>
      <td>0.058322</td>
      <td>0.0</td>
      <td>287.848386</td>
    </tr>
    <tr>
      <th>381416</th>
      <td>20679</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>666.794635</td>
      <td>2.975000</td>
      <td>0.949479</td>
      <td>930.647069</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH112504000</td>
      <td>20679.0</td>
      <td>126.6E_7.4N</td>
      <td>1025</td>
      <td>1484</td>
      <td>0.690701</td>
      <td>0.130240</td>
      <td>0.0</td>
      <td>642.798683</td>
    </tr>
    <tr>
      <th>381417</th>
      <td>20680</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>677.904037</td>
      <td>2.889583</td>
      <td>1.083333</td>
      <td>1800.666044</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH112504000</td>
      <td>20680.0</td>
      <td>126.6E_7.3N</td>
      <td>2708</td>
      <td>2798</td>
      <td>0.967834</td>
      <td>0.353105</td>
      <td>0.0</td>
      <td>1742.746122</td>
    </tr>
    <tr>
      <th>381418</th>
      <td>20680</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>677.904037</td>
      <td>2.889583</td>
      <td>1.083333</td>
      <td>1800.666044</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH112508000</td>
      <td>20680.0</td>
      <td>126.6E_7.3N</td>
      <td>90</td>
      <td>2798</td>
      <td>0.032166</td>
      <td>0.011735</td>
      <td>0.0</td>
      <td>57.919923</td>
    </tr>
    <tr>
      <th>381419</th>
      <td>20681</td>
      <td>2020</td>
      <td>0.0</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.0</td>
      <td>689.013439</td>
      <td>2.985417</td>
      <td>2.218056</td>
      <td>373.146778</td>
      <td>...</td>
      <td>0.0</td>
      <td>PH112508000</td>
      <td>20681.0</td>
      <td>126.6E_7.2N</td>
      <td>468</td>
      <td>468</td>
      <td>1.000000</td>
      <td>0.075605</td>
      <td>0.0</td>
      <td>373.146778</td>
    </tr>
  </tbody>
</table>
<p>381420 rows × 21 columns</p>
</div>




```python
join_df.sort_values("y_pred_all", ascending=False)
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
      <th>grid_point_id</th>
      <th>typhoon_year</th>
      <th>y_all</th>
      <th>y_pred_all</th>
      <th>typhoon_name</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>...</th>
      <th>percent_houses_damaged</th>
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
      <th>weight</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>weight*houses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>86903</th>
      <td>18470</td>
      <td>2013</td>
      <td>94.285260</td>
      <td>79.558975</td>
      <td>HAIYAN</td>
      <td>72.231420</td>
      <td>33.730735</td>
      <td>8.783333</td>
      <td>5.559896</td>
      <td>251.827606</td>
      <td>...</td>
      <td>94.285260</td>
      <td>PH082612000</td>
      <td>18470.0</td>
      <td>125.3E_11.2N</td>
      <td>339</td>
      <td>401</td>
      <td>0.845387</td>
      <td>169.374428</td>
      <td>200.725460</td>
      <td>212.891667</td>
    </tr>
    <tr>
      <th>86902</th>
      <td>18470</td>
      <td>2013</td>
      <td>94.285260</td>
      <td>79.558975</td>
      <td>HAIYAN</td>
      <td>72.231420</td>
      <td>33.730735</td>
      <td>8.783333</td>
      <td>5.559896</td>
      <td>251.827606</td>
      <td>...</td>
      <td>94.285260</td>
      <td>PH082602000</td>
      <td>18470.0</td>
      <td>125.3E_11.2N</td>
      <td>62</td>
      <td>401</td>
      <td>0.154613</td>
      <td>30.977034</td>
      <td>36.710851</td>
      <td>38.935939</td>
    </tr>
    <tr>
      <th>86752</th>
      <td>18303</td>
      <td>2013</td>
      <td>57.235598</td>
      <td>79.558975</td>
      <td>HAIYAN</td>
      <td>72.251930</td>
      <td>31.753148</td>
      <td>9.260417</td>
      <td>5.607813</td>
      <td>2383.683635</td>
      <td>...</td>
      <td>57.235598</td>
      <td>PH086010000</td>
      <td>18303.0</td>
      <td>125.2E_11.2N</td>
      <td>3848</td>
      <td>3853</td>
      <td>0.998702</td>
      <td>1893.973289</td>
      <td>1362.545114</td>
      <td>2380.590353</td>
    </tr>
    <tr>
      <th>86751</th>
      <td>18303</td>
      <td>2013</td>
      <td>57.235598</td>
      <td>79.558975</td>
      <td>HAIYAN</td>
      <td>72.251930</td>
      <td>31.753148</td>
      <td>9.260417</td>
      <td>5.607813</td>
      <td>2383.683635</td>
      <td>...</td>
      <td>57.235598</td>
      <td>PH082612000</td>
      <td>18303.0</td>
      <td>125.2E_11.2N</td>
      <td>5</td>
      <td>3853</td>
      <td>0.001298</td>
      <td>2.460984</td>
      <td>1.770459</td>
      <td>3.093283</td>
    </tr>
    <tr>
      <th>87506</th>
      <td>19307</td>
      <td>2013</td>
      <td>80.748749</td>
      <td>78.423950</td>
      <td>HAIYAN</td>
      <td>73.230522</td>
      <td>21.753512</td>
      <td>9.266667</td>
      <td>5.344792</td>
      <td>1625.734579</td>
      <td>...</td>
      <td>80.748749</td>
      <td>PH082609000</td>
      <td>19307.0</td>
      <td>125.8E_11.0N</td>
      <td>2556</td>
      <td>2556</td>
      <td>1.000000</td>
      <td>1274.965276</td>
      <td>1312.760341</td>
      <td>1625.734579</td>
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
      <th>159774</th>
      <td>12228</td>
      <td>2014</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>LINGLING</td>
      <td>0.000000</td>
      <td>1067.080548</td>
      <td>0.045833</td>
      <td>0.011458</td>
      <td>3519.226784</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH021512000</td>
      <td>12228.0</td>
      <td>121.6E_17.5N</td>
      <td>1427</td>
      <td>8263</td>
      <td>0.172698</td>
      <td>0.123141</td>
      <td>0.000000</td>
      <td>607.761905</td>
    </tr>
    <tr>
      <th>159773</th>
      <td>12227</td>
      <td>2014</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>LINGLING</td>
      <td>0.000000</td>
      <td>1075.778390</td>
      <td>0.043750</td>
      <td>0.010937</td>
      <td>4562.888107</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH143211000</td>
      <td>12227.0</td>
      <td>121.6E_17.6N</td>
      <td>505</td>
      <td>10925</td>
      <td>0.046224</td>
      <td>0.042735</td>
      <td>0.000000</td>
      <td>210.916109</td>
    </tr>
    <tr>
      <th>159772</th>
      <td>12227</td>
      <td>2014</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>LINGLING</td>
      <td>0.000000</td>
      <td>1075.778390</td>
      <td>0.043750</td>
      <td>0.010937</td>
      <td>4562.888107</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH021528000</td>
      <td>12227.0</td>
      <td>121.6E_17.6N</td>
      <td>21</td>
      <td>10925</td>
      <td>0.001922</td>
      <td>0.001777</td>
      <td>0.000000</td>
      <td>8.770769</td>
    </tr>
    <tr>
      <th>159771</th>
      <td>12227</td>
      <td>2014</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>LINGLING</td>
      <td>0.000000</td>
      <td>1075.778390</td>
      <td>0.043750</td>
      <td>0.010937</td>
      <td>4562.888107</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH021527000</td>
      <td>12227.0</td>
      <td>121.6E_17.6N</td>
      <td>9532</td>
      <td>10925</td>
      <td>0.872494</td>
      <td>0.806625</td>
      <td>0.000000</td>
      <td>3981.093770</td>
    </tr>
    <tr>
      <th>381419</th>
      <td>20681</td>
      <td>2020</td>
      <td>0.000000</td>
      <td>0.020261</td>
      <td>MOLAVE</td>
      <td>0.000000</td>
      <td>689.013439</td>
      <td>2.985417</td>
      <td>2.218056</td>
      <td>373.146778</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH112508000</td>
      <td>20681.0</td>
      <td>126.6E_7.2N</td>
      <td>468</td>
      <td>468</td>
      <td>1.000000</td>
      <td>0.075605</td>
      <td>0.000000</td>
      <td>373.146778</td>
    </tr>
  </tbody>
</table>
<p>381420 rows × 21 columns</p>
</div>




```python
# Groupby by municipality and typhoon_name with sum as the aggregation function
agg_df = join_df.groupby(["ADM3_PCODE", "typhoon_name", "typhoon_year"]).agg("sum")

# Normalize by the sum of the weights
agg_df["y_pred_norm"] = agg_df["weight*y_pred*houses"] / agg_df["weight*houses"] * 100
agg_df["y_norm"] = agg_df["weight*y*houses"] / agg_df["weight*houses"] * 100

# Drop not required column y and y_pred before multiplying by weight
agg_df.drop("y_all", axis=1, inplace=True)
agg_df.drop("y_pred_all", axis=1, inplace=True)

agg_df
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
      <th></th>
      <th></th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>percent_houses_damaged</th>
      <th>id_x</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
      <th>weight</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>weight*houses</th>
      <th>y_pred_norm</th>
      <th>y_norm</th>
    </tr>
    <tr>
      <th>ADM3_PCODE</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">PH012801000</th>
      <th>BOPHA</th>
      <th>2012</th>
      <td>44532</td>
      <td>74.049638</td>
      <td>654.265189</td>
      <td>4.362500</td>
      <td>2.143452</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.000000</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.122758</td>
      <td>0.000000</td>
      <td>477.856074</td>
      <td>0.025689</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CONSON</th>
      <th>2010</th>
      <td>44532</td>
      <td>0.000000</td>
      <td>1702.037073</td>
      <td>4.295833</td>
      <td>1.855729</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.000000</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.096820</td>
      <td>0.000000</td>
      <td>477.856074</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>DURIAN</th>
      <th>2006</th>
      <td>44532</td>
      <td>0.000000</td>
      <td>2200.940398</td>
      <td>51.497917</td>
      <td>18.063194</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.000000</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.096820</td>
      <td>0.000000</td>
      <td>477.856074</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>FENGSHEN</th>
      <th>2008</th>
      <td>44532</td>
      <td>44.394129</td>
      <td>1158.700144</td>
      <td>0.837500</td>
      <td>0.430208</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.000000</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.122758</td>
      <td>0.000000</td>
      <td>477.856074</td>
      <td>0.025689</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>FUNG-WONG</th>
      <th>2014</th>
      <td>44532</td>
      <td>61.869817</td>
      <td>64.984344</td>
      <td>27.481250</td>
      <td>21.412500</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.075522</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.122758</td>
      <td>0.309934</td>
      <td>477.856074</td>
      <td>0.025689</td>
      <td>0.064859</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
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
    </tr>
    <tr>
      <th rowspan="5" valign="top">PH175917000</th>
      <th>USAGI</th>
      <th>2013</th>
      <td>39174</td>
      <td>0.000000</td>
      <td>2529.061534</td>
      <td>6.262500</td>
      <td>3.472396</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.383107</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>UTOR</th>
      <th>2013</th>
      <td>39174</td>
      <td>0.000000</td>
      <td>1128.281754</td>
      <td>31.064583</td>
      <td>9.348958</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.383107</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VAMCO</th>
      <th>2020</th>
      <td>39174</td>
      <td>39.244147</td>
      <td>771.230647</td>
      <td>23.208333</td>
      <td>10.819271</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.485740</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.025689</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VONGFONG</th>
      <th>2020</th>
      <td>39174</td>
      <td>26.429974</td>
      <td>352.868347</td>
      <td>8.916667</td>
      <td>4.213021</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.383107</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YUTU</th>
      <th>2018</th>
      <td>39174</td>
      <td>0.000000</td>
      <td>1440.651706</td>
      <td>0.333333</td>
      <td>0.183854</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.383107</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>64233 rows × 17 columns</p>
</div>




```python
# agg_df.isnull().values.any()

# Remove rows with NaN after normalization
final_df = agg_df.dropna()
final_df
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
      <th></th>
      <th></th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>percent_houses_damaged</th>
      <th>id_x</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
      <th>weight</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>weight*houses</th>
      <th>y_pred_norm</th>
      <th>y_norm</th>
    </tr>
    <tr>
      <th>ADM3_PCODE</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">PH012801000</th>
      <th>BOPHA</th>
      <th>2012</th>
      <td>44532</td>
      <td>74.049638</td>
      <td>654.265189</td>
      <td>4.362500</td>
      <td>2.143452</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.000000</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.122758</td>
      <td>0.000000</td>
      <td>477.856074</td>
      <td>0.025689</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>CONSON</th>
      <th>2010</th>
      <td>44532</td>
      <td>0.000000</td>
      <td>1702.037073</td>
      <td>4.295833</td>
      <td>1.855729</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.000000</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.096820</td>
      <td>0.000000</td>
      <td>477.856074</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>DURIAN</th>
      <th>2006</th>
      <td>44532</td>
      <td>0.000000</td>
      <td>2200.940398</td>
      <td>51.497917</td>
      <td>18.063194</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.000000</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.096820</td>
      <td>0.000000</td>
      <td>477.856074</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>FENGSHEN</th>
      <th>2008</th>
      <td>44532</td>
      <td>44.394129</td>
      <td>1158.700144</td>
      <td>0.837500</td>
      <td>0.430208</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.000000</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.122758</td>
      <td>0.000000</td>
      <td>477.856074</td>
      <td>0.025689</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>FUNG-WONG</th>
      <th>2014</th>
      <td>44532</td>
      <td>61.869817</td>
      <td>64.984344</td>
      <td>27.481250</td>
      <td>21.412500</td>
      <td>1955.653450</td>
      <td>-1.810171</td>
      <td>0.075522</td>
      <td>44532.0</td>
      <td>1245</td>
      <td>4360</td>
      <td>1.571093</td>
      <td>0.122758</td>
      <td>0.309934</td>
      <td>477.856074</td>
      <td>0.025689</td>
      <td>0.064859</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
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
    </tr>
    <tr>
      <th rowspan="5" valign="top">PH175917000</th>
      <th>USAGI</th>
      <th>2013</th>
      <td>39174</td>
      <td>0.000000</td>
      <td>2529.061534</td>
      <td>6.262500</td>
      <td>3.472396</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.383107</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>UTOR</th>
      <th>2013</th>
      <td>39174</td>
      <td>0.000000</td>
      <td>1128.281754</td>
      <td>31.064583</td>
      <td>9.348958</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.383107</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VAMCO</th>
      <th>2020</th>
      <td>39174</td>
      <td>39.244147</td>
      <td>771.230647</td>
      <td>23.208333</td>
      <td>10.819271</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.485740</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.025689</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>VONGFONG</th>
      <th>2020</th>
      <td>39174</td>
      <td>26.429974</td>
      <td>352.868347</td>
      <td>8.916667</td>
      <td>4.213021</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.383107</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>YUTU</th>
      <th>2018</th>
      <td>39174</td>
      <td>0.000000</td>
      <td>1440.651706</td>
      <td>0.333333</td>
      <td>0.183854</td>
      <td>13442.349141</td>
      <td>-0.795551</td>
      <td>0.000000</td>
      <td>39174.0</td>
      <td>1892</td>
      <td>14630</td>
      <td>0.986935</td>
      <td>0.383107</td>
      <td>0.000000</td>
      <td>1890.820029</td>
      <td>0.020261</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>64233 rows × 17 columns</p>
</div>




```python
# Calculate RMSE in total for converted grid_based model to Mun_based

rmse = sqrt(mean_squared_error(final_df["y_norm"], final_df["y_pred_norm"]))
print(f"RMSE for grid_based model: {rmse:.2f}")
```

    RMSE for grid_based model: 1.64


### Calculate RMSE per bin for converted grid_based model to Mun_based


```python
bin_index = np.digitize(final_df["y_norm"], bins=binsP2)
# Define a function to estimate RMSE per bin
def rmse_bin(n):
    mse = mean_squared_error(
        final_df["y_norm"][bin_index == n],
        final_df["y_pred_norm"][bin_index == n],
    )
    rmse = np.sqrt(mse)
    print(f"RMSE per bin_{n}: {rmse:.2f}")


for bin_num in range(1, 6):
    rmse_bin(bin_num)
```

    RMSE per bin_1: 0.25
    RMSE per bin_2: 1.43
    RMSE per bin_3: 4.22
    RMSE per bin_4: 14.26
    RMSE per bin_5: 28.77


### Check if y_norm is the the same as the damage ground truth in the original model


```python
# Read the weight CSV file and import to df
df_old_data = pd.read_csv("data/old_data.csv")
df_old_data.drop("Unnamed: 0", axis=1, inplace=True)
df_old_data.columns = df_old_data.columns.str.replace("Mun_Code", "ADM3_PCODE")
df_old_data.head()
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
      <th>ADM3_PCODE</th>
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
      <th>VUL_LightRoof_StrongWall</th>
      <th>VUL_LightRoof_LightWall</th>
      <th>VUL_LightRoof_SalvageWall</th>
      <th>VUL_SalvagedRoof_StrongWall</th>
      <th>VUL_SalvagedRoof_LightWall</th>
      <th>VUL_SalvagedRoof_SalvageWall</th>
      <th>VUL_vulnerable_groups</th>
      <th>VUL_pantawid_pamilya_beneficiary</th>
      <th>DAM_perc_dmg</th>
      <th>HAZ_v_max_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH175101000</td>
      <td>durian2006</td>
      <td>185.828571</td>
      <td>14.716071</td>
      <td>7.381696</td>
      <td>55.032241</td>
      <td>2.478142</td>
      <td>2.64</td>
      <td>6.18</td>
      <td>6.18</td>
      <td>...</td>
      <td>2.533055</td>
      <td>41.892832</td>
      <td>1.002088</td>
      <td>0.000000</td>
      <td>0.027836</td>
      <td>0.083507</td>
      <td>2.951511</td>
      <td>46.931106</td>
      <td>3.632568</td>
      <td>166667.757548</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH083701000</td>
      <td>durian2006</td>
      <td>8.818750</td>
      <td>0.455208</td>
      <td>0.255319</td>
      <td>8.728380</td>
      <td>288.358553</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>1.218595</td>
      <td>13.645253</td>
      <td>0.549120</td>
      <td>0.030089</td>
      <td>0.090266</td>
      <td>0.112833</td>
      <td>3.338873</td>
      <td>25.989168</td>
      <td>0.000000</td>
      <td>664.968323</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH015501000</td>
      <td>durian2006</td>
      <td>24.175000</td>
      <td>2.408333</td>
      <td>0.957639</td>
      <td>10.945624</td>
      <td>274.953818</td>
      <td>1.52</td>
      <td>1.28</td>
      <td>1.28</td>
      <td>...</td>
      <td>0.667374</td>
      <td>15.592295</td>
      <td>0.075838</td>
      <td>0.000000</td>
      <td>0.015168</td>
      <td>0.075838</td>
      <td>2.131755</td>
      <td>32.185651</td>
      <td>0.000000</td>
      <td>1311.358762</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH015502000</td>
      <td>durian2006</td>
      <td>14.930000</td>
      <td>1.650000</td>
      <td>0.586250</td>
      <td>12.108701</td>
      <td>252.828578</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.675125</td>
      <td>7.100454</td>
      <td>0.023280</td>
      <td>0.011640</td>
      <td>0.000000</td>
      <td>0.128041</td>
      <td>1.589369</td>
      <td>29.612385</td>
      <td>0.000000</td>
      <td>1775.385328</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH175302000</td>
      <td>durian2006</td>
      <td>13.550000</td>
      <td>1.054167</td>
      <td>0.528125</td>
      <td>10.660943</td>
      <td>258.194381</td>
      <td>5.52</td>
      <td>0.36</td>
      <td>0.36</td>
      <td>...</td>
      <td>0.821288</td>
      <td>30.354796</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032852</td>
      <td>0.000000</td>
      <td>1.387007</td>
      <td>35.052562</td>
      <td>0.000000</td>
      <td>1211.676901</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 39 columns</p>
</div>




```python
# Capitalize strings typhoon column and change the typhoon column's name
for i in range(len(df_old_data)):
    df_old_data.at[i, "typhoon_name"] = df_old_data.loc[i, "typhoon"].upper()

del df_old_data["typhoon"]
df_old_data
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
      <th>ADM3_PCODE</th>
      <th>HAZ_rainfall_Total</th>
      <th>HAZ_rainfall_max_6h</th>
      <th>HAZ_rainfall_max_24h</th>
      <th>HAZ_v_max</th>
      <th>HAZ_dis_track_min</th>
      <th>GEN_landslide_per</th>
      <th>GEN_stormsurge_per</th>
      <th>GEN_Bu_p_inSSA</th>
      <th>GEN_Bu_p_LS</th>
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
      <th>typhoon_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH175101000</td>
      <td>185.828571</td>
      <td>14.716071</td>
      <td>7.381696</td>
      <td>55.032241</td>
      <td>2.478142</td>
      <td>2.64</td>
      <td>6.18</td>
      <td>6.18</td>
      <td>2.64</td>
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
      <td>DURIAN2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH083701000</td>
      <td>8.818750</td>
      <td>0.455208</td>
      <td>0.255319</td>
      <td>8.728380</td>
      <td>288.358553</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.06</td>
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
      <td>DURIAN2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH015501000</td>
      <td>24.175000</td>
      <td>2.408333</td>
      <td>0.957639</td>
      <td>10.945624</td>
      <td>274.953818</td>
      <td>1.52</td>
      <td>1.28</td>
      <td>1.28</td>
      <td>1.52</td>
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
      <td>DURIAN2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH015502000</td>
      <td>14.930000</td>
      <td>1.650000</td>
      <td>0.586250</td>
      <td>12.108701</td>
      <td>252.828578</td>
      <td>0.00</td>
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
      <td>DURIAN2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH175302000</td>
      <td>13.550000</td>
      <td>1.054167</td>
      <td>0.528125</td>
      <td>10.660943</td>
      <td>258.194381</td>
      <td>5.52</td>
      <td>0.36</td>
      <td>0.36</td>
      <td>5.52</td>
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
      <td>DURIAN2006</td>
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
      <td>9.700000</td>
      <td>0.408333</td>
      <td>0.216146</td>
      <td>8.136932</td>
      <td>277.107823</td>
      <td>1.80</td>
      <td>6.25</td>
      <td>6.25</td>
      <td>1.80</td>
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
      <td>NOUL2015</td>
    </tr>
    <tr>
      <th>8069</th>
      <td>PH015547000</td>
      <td>17.587500</td>
      <td>1.414583</td>
      <td>0.386458</td>
      <td>9.818999</td>
      <td>305.789817</td>
      <td>0.00</td>
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
      <td>NOUL2015</td>
    </tr>
    <tr>
      <th>8070</th>
      <td>PH025014000</td>
      <td>11.487500</td>
      <td>0.614583</td>
      <td>0.230319</td>
      <td>15.791907</td>
      <td>210.313249</td>
      <td>0.06</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.06</td>
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
      <td>NOUL2015</td>
    </tr>
    <tr>
      <th>8071</th>
      <td>PH140127000</td>
      <td>11.600000</td>
      <td>1.400000</td>
      <td>0.412766</td>
      <td>13.867145</td>
      <td>218.189328</td>
      <td>0.00</td>
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
      <td>NOUL2015</td>
    </tr>
    <tr>
      <th>8072</th>
      <td>PH051612000</td>
      <td>32.305556</td>
      <td>1.744444</td>
      <td>1.210417</td>
      <td>15.647639</td>
      <td>219.542224</td>
      <td>4.15</td>
      <td>3.05</td>
      <td>3.05</td>
      <td>4.15</td>
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
      <td>NOUL2015</td>
    </tr>
  </tbody>
</table>
<p>8073 rows × 39 columns</p>
</div>




```python
df_old_data["typhoon_year"] = df_old_data["typhoon_name"].str[-4:]
df_old_data["typhoon_name"] = df_old_data["typhoon_name"].str[:-4]
df_old_data["typhoon_year"] = df_old_data["typhoon_year"].astype("int64")
df_old_data
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
      <th>ADM3_PCODE</th>
      <th>HAZ_rainfall_Total</th>
      <th>HAZ_rainfall_max_6h</th>
      <th>HAZ_rainfall_max_24h</th>
      <th>HAZ_v_max</th>
      <th>HAZ_dis_track_min</th>
      <th>GEN_landslide_per</th>
      <th>GEN_stormsurge_per</th>
      <th>GEN_Bu_p_inSSA</th>
      <th>GEN_Bu_p_LS</th>
      <th>...</th>
      <th>VUL_LightRoof_SalvageWall</th>
      <th>VUL_SalvagedRoof_StrongWall</th>
      <th>VUL_SalvagedRoof_LightWall</th>
      <th>VUL_SalvagedRoof_SalvageWall</th>
      <th>VUL_vulnerable_groups</th>
      <th>VUL_pantawid_pamilya_beneficiary</th>
      <th>DAM_perc_dmg</th>
      <th>HAZ_v_max_3</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH175101000</td>
      <td>185.828571</td>
      <td>14.716071</td>
      <td>7.381696</td>
      <td>55.032241</td>
      <td>2.478142</td>
      <td>2.64</td>
      <td>6.18</td>
      <td>6.18</td>
      <td>2.64</td>
      <td>...</td>
      <td>1.002088</td>
      <td>0.000000</td>
      <td>0.027836</td>
      <td>0.083507</td>
      <td>2.951511</td>
      <td>46.931106</td>
      <td>3.632568</td>
      <td>166667.757548</td>
      <td>DURIAN</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH083701000</td>
      <td>8.818750</td>
      <td>0.455208</td>
      <td>0.255319</td>
      <td>8.728380</td>
      <td>288.358553</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>...</td>
      <td>0.549120</td>
      <td>0.030089</td>
      <td>0.090266</td>
      <td>0.112833</td>
      <td>3.338873</td>
      <td>25.989168</td>
      <td>0.000000</td>
      <td>664.968323</td>
      <td>DURIAN</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH015501000</td>
      <td>24.175000</td>
      <td>2.408333</td>
      <td>0.957639</td>
      <td>10.945624</td>
      <td>274.953818</td>
      <td>1.52</td>
      <td>1.28</td>
      <td>1.28</td>
      <td>1.52</td>
      <td>...</td>
      <td>0.075838</td>
      <td>0.000000</td>
      <td>0.015168</td>
      <td>0.075838</td>
      <td>2.131755</td>
      <td>32.185651</td>
      <td>0.000000</td>
      <td>1311.358762</td>
      <td>DURIAN</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH015502000</td>
      <td>14.930000</td>
      <td>1.650000</td>
      <td>0.586250</td>
      <td>12.108701</td>
      <td>252.828578</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.023280</td>
      <td>0.011640</td>
      <td>0.000000</td>
      <td>0.128041</td>
      <td>1.589369</td>
      <td>29.612385</td>
      <td>0.000000</td>
      <td>1775.385328</td>
      <td>DURIAN</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH175302000</td>
      <td>13.550000</td>
      <td>1.054167</td>
      <td>0.528125</td>
      <td>10.660943</td>
      <td>258.194381</td>
      <td>5.52</td>
      <td>0.36</td>
      <td>0.36</td>
      <td>5.52</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032852</td>
      <td>0.000000</td>
      <td>1.387007</td>
      <td>35.052562</td>
      <td>0.000000</td>
      <td>1211.676901</td>
      <td>DURIAN</td>
      <td>2006</td>
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
      <td>9.700000</td>
      <td>0.408333</td>
      <td>0.216146</td>
      <td>8.136932</td>
      <td>277.107823</td>
      <td>1.80</td>
      <td>6.25</td>
      <td>6.25</td>
      <td>1.80</td>
      <td>...</td>
      <td>0.311526</td>
      <td>0.031153</td>
      <td>0.155763</td>
      <td>0.031153</td>
      <td>2.827833</td>
      <td>31.308411</td>
      <td>0.000000</td>
      <td>538.743551</td>
      <td>NOUL</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>8069</th>
      <td>PH015547000</td>
      <td>17.587500</td>
      <td>1.414583</td>
      <td>0.386458</td>
      <td>9.818999</td>
      <td>305.789817</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.027875</td>
      <td>0.000000</td>
      <td>0.034843</td>
      <td>0.097561</td>
      <td>1.073268</td>
      <td>12.766551</td>
      <td>0.000000</td>
      <td>946.676507</td>
      <td>NOUL</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>8070</th>
      <td>PH025014000</td>
      <td>11.487500</td>
      <td>0.614583</td>
      <td>0.230319</td>
      <td>15.791907</td>
      <td>210.313249</td>
      <td>0.06</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.06</td>
      <td>...</td>
      <td>0.022528</td>
      <td>0.000000</td>
      <td>0.067583</td>
      <td>0.022528</td>
      <td>1.140109</td>
      <td>9.348952</td>
      <td>0.000000</td>
      <td>3938.254316</td>
      <td>NOUL</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>8071</th>
      <td>PH140127000</td>
      <td>11.600000</td>
      <td>1.400000</td>
      <td>0.412766</td>
      <td>13.867145</td>
      <td>218.189328</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.837537</td>
      <td>21.928166</td>
      <td>0.000000</td>
      <td>2666.620370</td>
      <td>NOUL</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>8072</th>
      <td>PH051612000</td>
      <td>32.305556</td>
      <td>1.744444</td>
      <td>1.210417</td>
      <td>15.647639</td>
      <td>219.542224</td>
      <td>4.15</td>
      <td>3.05</td>
      <td>3.05</td>
      <td>4.15</td>
      <td>...</td>
      <td>0.280316</td>
      <td>0.010382</td>
      <td>0.031146</td>
      <td>0.103821</td>
      <td>2.518110</td>
      <td>31.634136</td>
      <td>0.000000</td>
      <td>3831.302757</td>
      <td>NOUL</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
<p>8073 rows × 40 columns</p>
</div>




```python
agg_df_old_data = df_old_data.groupby(["ADM3_PCODE"]).agg("sum")
agg_df_old_data
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
      <th>HAZ_rainfall_Total</th>
      <th>HAZ_rainfall_max_6h</th>
      <th>HAZ_rainfall_max_24h</th>
      <th>HAZ_v_max</th>
      <th>HAZ_dis_track_min</th>
      <th>GEN_landslide_per</th>
      <th>GEN_stormsurge_per</th>
      <th>GEN_Bu_p_inSSA</th>
      <th>GEN_Bu_p_LS</th>
      <th>GEN_Red_per_LSbldg</th>
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
      <th>typhoon_year</th>
    </tr>
    <tr>
      <th>ADM3_PCODE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PH012801000</th>
      <td>1540.191667</td>
      <td>69.743056</td>
      <td>40.703450</td>
      <td>267.204204</td>
      <td>1137.195510</td>
      <td>36.72</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>36.72</td>
      <td>36.72</td>
      <td>...</td>
      <td>21.897810</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>22.098214</td>
      <td>370.072993</td>
      <td>0.973236</td>
      <td>713418.086535</td>
      <td>18135</td>
    </tr>
    <tr>
      <th>PH012802000</th>
      <td>1567.800000</td>
      <td>66.133333</td>
      <td>41.231272</td>
      <td>289.548362</td>
      <td>1116.545916</td>
      <td>14.10</td>
      <td>15.40</td>
      <td>15.40</td>
      <td>14.10</td>
      <td>0.00</td>
      <td>...</td>
      <td>4.535441</td>
      <td>0.000000</td>
      <td>0.129584</td>
      <td>0.000000</td>
      <td>0.259168</td>
      <td>5.370169</td>
      <td>83.322535</td>
      <td>1.399508</td>
      <td>600411.585560</td>
      <td>20145</td>
    </tr>
    <tr>
      <th>PH012803000</th>
      <td>1482.887500</td>
      <td>84.754167</td>
      <td>44.353513</td>
      <td>260.810168</td>
      <td>1188.415514</td>
      <td>24.50</td>
      <td>16.80</td>
      <td>16.80</td>
      <td>24.50</td>
      <td>0.40</td>
      <td>...</td>
      <td>65.408805</td>
      <td>0.279525</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.419287</td>
      <td>14.581225</td>
      <td>208.106219</td>
      <td>7.491265</td>
      <td>366336.115659</td>
      <td>20147</td>
    </tr>
    <tr>
      <th>PH012804000</th>
      <td>1714.850000</td>
      <td>92.071667</td>
      <td>46.335691</td>
      <td>341.286734</td>
      <td>998.382173</td>
      <td>28.80</td>
      <td>15.10</td>
      <td>15.10</td>
      <td>28.80</td>
      <td>4.10</td>
      <td>...</td>
      <td>26.032824</td>
      <td>1.697793</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>11.177754</td>
      <td>149.405772</td>
      <td>1.245048</td>
      <td>878159.265616</td>
      <td>20144</td>
    </tr>
    <tr>
      <th>PH012805000</th>
      <td>2049.008333</td>
      <td>94.663258</td>
      <td>56.398160</td>
      <td>287.371061</td>
      <td>1908.790607</td>
      <td>0.39</td>
      <td>0.39</td>
      <td>0.39</td>
      <td>0.39</td>
      <td>0.39</td>
      <td>...</td>
      <td>24.425008</td>
      <td>0.210560</td>
      <td>0.105280</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.865800</td>
      <td>178.660512</td>
      <td>0.688371</td>
      <td>419859.619693</td>
      <td>26190</td>
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
      <th>PH175913000</th>
      <td>369.121429</td>
      <td>32.132143</td>
      <td>13.755053</td>
      <td>61.168774</td>
      <td>155.821370</td>
      <td>7.08</td>
      <td>4.22</td>
      <td>4.22</td>
      <td>7.08</td>
      <td>3.16</td>
      <td>...</td>
      <td>67.657694</td>
      <td>0.281237</td>
      <td>0.361591</td>
      <td>1.084773</td>
      <td>0.160707</td>
      <td>6.016072</td>
      <td>62.394536</td>
      <td>1.004419</td>
      <td>61326.291421</td>
      <td>4028</td>
    </tr>
    <tr>
      <th>PH175914000</th>
      <td>97.650000</td>
      <td>5.802273</td>
      <td>3.454654</td>
      <td>82.724509</td>
      <td>705.502314</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>156.085320</td>
      <td>0.501882</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.167294</td>
      <td>16.101461</td>
      <td>137.348390</td>
      <td>3.429527</td>
      <td>87982.096558</td>
      <td>8056</td>
    </tr>
    <tr>
      <th>PH175915000</th>
      <td>265.100000</td>
      <td>14.297424</td>
      <td>8.030009</td>
      <td>94.082701</td>
      <td>879.185712</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>240.314278</td>
      <td>0.135465</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>21.990309</td>
      <td>173.530209</td>
      <td>2.898943</td>
      <td>64533.566739</td>
      <td>10070</td>
    </tr>
    <tr>
      <th>PH175916000</th>
      <td>1276.875000</td>
      <td>74.493561</td>
      <td>40.133178</td>
      <td>237.894624</td>
      <td>940.086245</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>289.110708</td>
      <td>1.633394</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>37.478461</td>
      <td>270.598911</td>
      <td>4.355717</td>
      <td>308969.515725</td>
      <td>18138</td>
    </tr>
    <tr>
      <th>PH175917000</th>
      <td>1194.012500</td>
      <td>72.880492</td>
      <td>40.962489</td>
      <td>239.851930</td>
      <td>728.274274</td>
      <td>38.48</td>
      <td>4.16</td>
      <td>4.16</td>
      <td>38.48</td>
      <td>2.72</td>
      <td>...</td>
      <td>279.574468</td>
      <td>8.085106</td>
      <td>3.829787</td>
      <td>2.978723</td>
      <td>8.085106</td>
      <td>25.669958</td>
      <td>218.297872</td>
      <td>8.031915</td>
      <td>338853.287529</td>
      <td>16119</td>
    </tr>
  </tbody>
</table>
<p>1452 rows × 38 columns</p>
</div>




```python
df_merged = df_old_data.merge(
    final_df,
    how="left",
    on=["ADM3_PCODE", "typhoon_name", "typhoon_year"],
)

df_merged
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
      <th>ADM3_PCODE</th>
      <th>HAZ_rainfall_Total</th>
      <th>HAZ_rainfall_max_6h</th>
      <th>HAZ_rainfall_max_24h</th>
      <th>HAZ_v_max</th>
      <th>HAZ_dis_track_min</th>
      <th>GEN_landslide_per</th>
      <th>GEN_stormsurge_per</th>
      <th>GEN_Bu_p_inSSA</th>
      <th>GEN_Bu_p_LS</th>
      <th>...</th>
      <th>percent_houses_damaged</th>
      <th>id_x</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
      <th>weight</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>weight*houses</th>
      <th>y_pred_norm</th>
      <th>y_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH175101000</td>
      <td>185.828571</td>
      <td>14.716071</td>
      <td>7.381696</td>
      <td>55.032241</td>
      <td>2.478142</td>
      <td>2.64</td>
      <td>6.18</td>
      <td>6.18</td>
      <td>2.64</td>
      <td>...</td>
      <td>49.307819</td>
      <td>119093.0</td>
      <td>12525</td>
      <td>20510</td>
      <td>7.055036</td>
      <td>358.137887</td>
      <td>241.028059</td>
      <td>7195.404161</td>
      <td>4.977314</td>
      <td>3.34975</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH083701000</td>
      <td>8.818750</td>
      <td>0.455208</td>
      <td>0.255319</td>
      <td>8.728380</td>
      <td>288.358553</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.06</td>
      <td>...</td>
      <td>0.000000</td>
      <td>162440.0</td>
      <td>18403</td>
      <td>38825</td>
      <td>3.749379</td>
      <td>3.029766</td>
      <td>0.000000</td>
      <td>13093.458072</td>
      <td>0.023140</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH015501000</td>
      <td>24.175000</td>
      <td>2.408333</td>
      <td>0.957639</td>
      <td>10.945624</td>
      <td>274.953818</td>
      <td>1.52</td>
      <td>1.28</td>
      <td>1.28</td>
      <td>1.52</td>
      <td>...</td>
      <td>0.000000</td>
      <td>37111.0</td>
      <td>11529</td>
      <td>34559</td>
      <td>1.448341</td>
      <td>1.657080</td>
      <td>0.000000</td>
      <td>6450.447834</td>
      <td>0.025689</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH015502000</td>
      <td>14.930000</td>
      <td>1.650000</td>
      <td>0.586250</td>
      <td>12.108701</td>
      <td>252.828578</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000000</td>
      <td>59608.0</td>
      <td>18551</td>
      <td>119562</td>
      <td>2.160373</td>
      <td>2.188238</td>
      <td>0.000000</td>
      <td>8518.063512</td>
      <td>0.025689</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH175302000</td>
      <td>13.550000</td>
      <td>1.054167</td>
      <td>0.528125</td>
      <td>10.660943</td>
      <td>258.194381</td>
      <td>5.52</td>
      <td>0.36</td>
      <td>0.36</td>
      <td>5.52</td>
      <td>...</td>
      <td>0.000000</td>
      <td>89141.0</td>
      <td>3052</td>
      <td>3054</td>
      <td>7.000000</td>
      <td>0.781985</td>
      <td>0.000000</td>
      <td>3044.000000</td>
      <td>0.025689</td>
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
      <td>9.700000</td>
      <td>0.408333</td>
      <td>0.216146</td>
      <td>8.136932</td>
      <td>277.107823</td>
      <td>1.80</td>
      <td>6.25</td>
      <td>6.25</td>
      <td>1.80</td>
      <td>...</td>
      <td>0.000000</td>
      <td>84606.0</td>
      <td>4074</td>
      <td>25273</td>
      <td>1.109882</td>
      <td>0.790047</td>
      <td>0.000000</td>
      <td>3352.502863</td>
      <td>0.023566</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8069</th>
      <td>PH015547000</td>
      <td>17.587500</td>
      <td>1.414583</td>
      <td>0.386458</td>
      <td>9.818999</td>
      <td>305.789817</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000000</td>
      <td>20981.0</td>
      <td>34899</td>
      <td>107472</td>
      <td>0.619506</td>
      <td>3.039305</td>
      <td>0.000000</td>
      <td>15000.469589</td>
      <td>0.020261</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8070</th>
      <td>PH025014000</td>
      <td>11.487500</td>
      <td>0.614583</td>
      <td>0.230319</td>
      <td>15.791907</td>
      <td>210.313249</td>
      <td>0.06</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.06</td>
      <td>...</td>
      <td>0.000000</td>
      <td>22971.0</td>
      <td>10141</td>
      <td>29254</td>
      <td>0.646572</td>
      <td>1.248710</td>
      <td>0.000000</td>
      <td>4860.803528</td>
      <td>0.025689</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8071</th>
      <td>PH140127000</td>
      <td>11.600000</td>
      <td>1.400000</td>
      <td>0.412766</td>
      <td>13.867145</td>
      <td>218.189328</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.000000</td>
      <td>53128.0</td>
      <td>2761</td>
      <td>28504</td>
      <td>0.499031</td>
      <td>0.308938</td>
      <td>0.000000</td>
      <td>1202.589219</td>
      <td>0.025689</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8072</th>
      <td>PH051612000</td>
      <td>32.305556</td>
      <td>1.744444</td>
      <td>1.210417</td>
      <td>15.647639</td>
      <td>219.542224</td>
      <td>4.15</td>
      <td>3.05</td>
      <td>3.05</td>
      <td>4.15</td>
      <td>...</td>
      <td>0.000000</td>
      <td>130211.0</td>
      <td>13433</td>
      <td>52782</td>
      <td>5.272019</td>
      <td>2.541117</td>
      <td>0.000000</td>
      <td>9789.272845</td>
      <td>0.025958</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
<p>8073 rows × 57 columns</p>
</div>




```python
coef = df_merged["DAM_perc_dmg"].corr(df_merged["y_norm"])
print(f"Correlation Coefficient is {coef:.3f}")
```

    Correlation Coefficient is 0.920



```python
x = df_merged["DAM_perc_dmg"]
y = df_merged["y_norm"]
plt.rcParams.update({"figure.figsize": (6, 4), "figure.dpi": 100})
plt.scatter(x, y, c=y, cmap="Spectral")
plt.colorbar()
plt.title("Scatter plot of damaged in original model and damaged in grid_to_mun")
plt.xlabel("damaged_data")
plt.ylabel("y_norm")
plt.show()
```


    
![png](output_40_0.png)
    



```python
diff = df_merged["y_norm"] - df_merged["DAM_perc_dmg"]
diff.hist(bins=40, figsize=(4, 3))
plt.title("Histogram of diff")
plt.xlabel("diff")
plt.ylabel("Frequency")
```




    Text(0, 0.5, 'Frequency')




    
![png](output_41_1.png)
    



```python

```

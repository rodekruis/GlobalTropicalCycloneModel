# Combined  Model (XGBoost Undersampling + XGBoost Regression)

We developed a hybrid model using both xgboost regression and xgboost classification(while undersampling technique was implemented to enhance its performance). Subsequently, we evaluated the performance of this combined model on the test dataset and compared it with the result of the simple xgboost regression model.



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
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import imblearn
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBClassifier
from sty import fg, rs

from sklearn.metrics import confusion_matrix
from matplotlib import cm
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from utils import get_training_dataset
```

    pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.



```python
# Read csv file and import to df
df = get_training_dataset()
df.head()
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
      <th>mean_slope</th>
      <th>...</th>
      <th>std_tri</th>
      <th>mean_elev</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged</th>
      <th>percent_houses_damaged_5years</th>
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
      <td>1.018526</td>
      <td>...</td>
      <td>2.699781</td>
      <td>5.762712</td>
      <td>3445.709753</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>0.0</td>
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
      <td>1.579400</td>
      <td>...</td>
      <td>4.585088</td>
      <td>12.799127</td>
      <td>8602.645832</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>0.0</td>
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
      <td>0.551764</td>
      <td>...</td>
      <td>1.527495</td>
      <td>8.833333</td>
      <td>5084.012925</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.01</td>
      <td>0.99</td>
      <td>197.339034</td>
      <td>0.0</td>
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
      <td>2.107949</td>
      <td>...</td>
      <td>11.677657</td>
      <td>17.530431</td>
      <td>55607.865950</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.31</td>
      <td>0.69</td>
      <td>4970.477311</td>
      <td>0.0</td>
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
      <td>3.538881</td>
      <td>...</td>
      <td>17.074011</td>
      <td>31.931338</td>
      <td>35529.342507</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.77</td>
      <td>0.23</td>
      <td>12408.594656</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)

# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```


```python
# Remove zeros from wind_speed
df = (df[(df[["wind_speed"]] != 0).any(axis=1)]).reset_index(drop=True)
df = df.drop(columns=["grid_point_id", "typhoon_year"])
df.head()
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>mean_tri</th>
      <th>std_tri</th>
      <th>mean_elev</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged</th>
      <th>percent_houses_damaged_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DURIAN</td>
      <td>12.460039</td>
      <td>275.018491</td>
      <td>0.670833</td>
      <td>0.313021</td>
      <td>0.479848</td>
      <td>-0.213039</td>
      <td>12.896581</td>
      <td>7.450346</td>
      <td>74.625539</td>
      <td>34.629550</td>
      <td>42.218750</td>
      <td>5303.659490</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN</td>
      <td>11.428974</td>
      <td>297.027578</td>
      <td>0.929167</td>
      <td>0.343229</td>
      <td>55.649739</td>
      <td>0.206000</td>
      <td>14.070741</td>
      <td>6.514647</td>
      <td>68.681417</td>
      <td>25.475388</td>
      <td>72.283154</td>
      <td>61015.543599</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.14</td>
      <td>0.86</td>
      <td>276.871504</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>0.716667</td>
      <td>0.424479</td>
      <td>8.157414</td>
      <td>-0.636000</td>
      <td>19.758682</td>
      <td>10.940700</td>
      <td>104.453163</td>
      <td>54.353996</td>
      <td>102.215198</td>
      <td>66707.438070</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>0.89</td>
      <td>448.539453</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN</td>
      <td>12.511864</td>
      <td>273.639330</td>
      <td>0.568750</td>
      <td>0.336979</td>
      <td>88.292015</td>
      <td>-0.227500</td>
      <td>11.499097</td>
      <td>6.901584</td>
      <td>59.798108</td>
      <td>31.814048</td>
      <td>58.988877</td>
      <td>53841.050168</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>2101.708435</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN</td>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>0.589583</td>
      <td>0.290625</td>
      <td>962.766739</td>
      <td>-0.299667</td>
      <td>13.866633</td>
      <td>6.528689</td>
      <td>65.655280</td>
      <td>25.976413</td>
      <td>111.386527</td>
      <td>87378.257957</td>
      <td>1</td>
      <td>0.07</td>
      <td>0.46</td>
      <td>0.47</td>
      <td>11632.726327</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define bins for data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
bins_eval = [0, 1, 10, 20, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
```


```python
# Check the bins' intervalls (first bin means all zeros, second bin means 0 < values <= 1)
df["percent_houses_damaged"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    38901
    (9e-05, 1.0]        7232
    (1.0, 10.0]         2552
    (10.0, 50.0]         925
    (50.0, 101.0]        144
    Name: percent_houses_damaged, dtype: int64




```python
print(samples_per_bin2)
print(binsP2)
```

    [38901  7232  2552   925   144]
    [0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]



```python
bin_index2 = np.digitize(df["percent_houses_damaged"], bins=binsP2)
```


```python
y_input_strat = bin_index2
```


```python
features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "rwi",
    "mean_slope",
    "std_slope",
    "mean_tri",
    "std_tri",
    "mean_elev",
    "coast_length",
    "with_coast",
    "urban",
    "rural",
    "water",
    "total_pop",
    "percent_houses_damaged_5years",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["percent_houses_damaged"]
```


    Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
           'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
           'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
           'water', 'total_pop', 'percent_houses_damaged_5years'],
          dtype='object')



```python
# Define train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["percent_houses_damaged"],
    test_size=0.2,
    stratify=y_input_strat,
)
```

## First step is to train XGBoost Regression model for train data


```python
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

eval_set = [(X_test, y_test)]
xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
```

    pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.


    [02:13:33] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.

      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.





```python
# Make prediction on train and test data
y_pred_train = xgb.predict(X_train)
y_pred = xgb.predict(X_test)
```


```python
# Calculate RMSE in total

mse_train_idx = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train_idx)

mse_idx = mean_squared_error(y_test, y_pred)
rmseM1 = np.sqrt(mse_idx)

print(f"RMSE_test_in_total: {rmseM1:.2f}")
print(f"RMSE_train_in_total: {rmse_train:.2f}")
```

    RMSE_test_in_total: 3.08
    RMSE_train_in_total: 2.65



```python
# Calculate RMSE per bins

bin_index_test = np.digitize(y_test, bins=bins_eval)
bin_index_train = np.digitize(y_train, bins=bins_eval)

RSME_test_model1 = np.zeros(len(bins_eval) - 1)

for bin_num in range(1, len(bins_eval)):

    # Estimation of RMSE for train data
    mse_train_idx = mean_squared_error(
        y_train[bin_index_train == bin_num], y_pred_train[bin_index_train == bin_num]
    )
    rmse_train = np.sqrt(mse_train_idx)

    # Estimation of RMSE for test data
    mse_idx = mean_squared_error(
        y_test[bin_index_test == bin_num], y_pred[bin_index_test == bin_num]
    )
    RSME_test_model1[bin_num - 1] = np.sqrt(mse_idx)

    print(
        f"RMSE_test  [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {RSME_test_model1[bin_num-1]:.2f}"
    )
    print(
        f"RMSE_train [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {rmse_train:.2f}"
    )
```

    RMSE_test  [0,1): 1.17
    RMSE_train [0,1): 0.94
    RMSE_test  [1,10): 4.54
    RMSE_train [1,10): 3.93
    RMSE_test  [10,20): 9.31
    RMSE_train [10,20): 9.03
    RMSE_test  [20,50): 19.75
    RMSE_train [20,50): 15.83
    RMSE_test  [50,101): 33.02
    RMSE_train [50,101): 28.50


## Second step is to train XGBoost Binary model for same train data


```python
# Define a threshold to separate target into damaged and not_damaged
thres = 10.0
y_test_bool = y_test >= thres
y_train_bool = y_train >= thres
y_test_bin = (y_test_bool) * 1
y_train_bin = (y_train_bool) * 1
```


```python
sum(y_train_bin)
```




    855




```python
print(Counter(y_train_bin))
```

    Counter({0: 38948, 1: 855})



```python
# Undersampling

# Define undersampling strategy
under = RandomUnderSampler(sampling_strategy=0.1)
# Fit and apply the transform
X_train_us, y_train_us = under.fit_resample(X_train, y_train_bin)

print(Counter(y_train_us))
```

    Counter({0: 8550, 1: 855})



```python
# Use XGBClassifier as a Machine Learning model to fit the data
xgb_model = XGBClassifier(eval_metric=["error", "logloss"])

# eval_set = [(X_train, y_train), (X_train, y_train)]
eval_set = [(X_test, y_test_bin)]
xgb_model.fit(
    X_train_us,
    y_train_us,
    eval_set=eval_set,
    verbose=False,
)
```

    The use of label encoder in XGBClassifier is deprecated and will be removed in a future release. To remove this warning, do the following: 1) Pass option use_label_encoder=False when constructing XGBClassifier object; and 2) Encode your labels (y) as integers starting with 0, i.e. 0, 1, 2, ..., [num_class - 1].
    pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.





<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>XGBClassifier(base_score=0.5, booster=&#x27;gbtree&#x27;, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              eval_metric=[&#x27;error&#x27;, &#x27;logloss&#x27;], gamma=0, gpu_id=-1,
              importance_type=None, interaction_constraints=&#x27;&#x27;,
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints=&#x27;()&#x27;,
              n_estimators=100, n_jobs=8, num_parallel_tree=1, predictor=&#x27;auto&#x27;,
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=1, tree_method=&#x27;exact&#x27;, validate_parameters=1,
              verbosity=None)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=0.5, booster=&#x27;gbtree&#x27;, colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, enable_categorical=False,
              eval_metric=[&#x27;error&#x27;, &#x27;logloss&#x27;], gamma=0, gpu_id=-1,
              importance_type=None, interaction_constraints=&#x27;&#x27;,
              learning_rate=0.300000012, max_delta_step=0, max_depth=6,
              min_child_weight=1, missing=nan, monotone_constraints=&#x27;()&#x27;,
              n_estimators=100, n_jobs=8, num_parallel_tree=1, predictor=&#x27;auto&#x27;,
              random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
              subsample=1, tree_method=&#x27;exact&#x27;, validate_parameters=1,
              verbosity=None)</pre></div></div></div></div></div>




```python
# Make prediction on test data
y_pred_test = xgb_model.predict(X_test)
```


```python
# Print Confusion Matrix
cm = confusion_matrix(y_test_bin, y_pred_test)
cm
```




    array([[9601,  136],
           [  61,  153]])




```python
# Classification Report
print(metrics.classification_report(y_test_bin, y_pred_test))
print(metrics.confusion_matrix(y_test_bin, y_pred_test))
```

                  precision    recall  f1-score   support

               0       0.99      0.99      0.99      9737
               1       0.53      0.71      0.61       214

        accuracy                           0.98      9951
       macro avg       0.76      0.85      0.80      9951
    weighted avg       0.98      0.98      0.98      9951

    [[9601  136]
     [  61  153]]



```python
# Make prediction on train data
y_pred_train = xgb_model.predict(X_train)
```


```python
# Print Confusion Matrix
cm = confusion_matrix(y_train_bin, y_pred_train)
cm
```




    array([[38510,   438],
           [    0,   855]])




```python
# Classification Report
print(metrics.classification_report(y_train_bin, y_pred_train))
print(metrics.confusion_matrix(y_train_bin, y_pred_train))
```

                  precision    recall  f1-score   support

               0       1.00      0.99      0.99     38948
               1       0.66      1.00      0.80       855

        accuracy                           0.99     39803
       macro avg       0.83      0.99      0.90     39803
    weighted avg       0.99      0.99      0.99     39803

    [[38510   438]
     [    0   855]]



```python
reduced_df = X_train.copy()
```


```python
reduced_df["percent_houses_damaged"] = y_train.values
reduced_df["predicted_value"] = y_pred_train
```


```python
fliterd_df = reduced_df[reduced_df.predicted_value == 1]
```


```python
fliterd_df
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>mean_tri</th>
      <th>std_tri</th>
      <th>mean_elev</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>predicted_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13145</th>
      <td>72.251930</td>
      <td>31.753148</td>
      <td>2383.683635</td>
      <td>9.260417</td>
      <td>5.607813</td>
      <td>0.308800</td>
      <td>8.698314</td>
      <td>6.900810</td>
      <td>42.025400</td>
      <td>29.846938</td>
      <td>196.562202</td>
      <td>20400.179150</td>
      <td>1</td>
      <td>0.14</td>
      <td>0.57</td>
      <td>0.29</td>
      <td>13763.519461</td>
      <td>1.508420</td>
      <td>57.235598</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40740</th>
      <td>45.274487</td>
      <td>13.634354</td>
      <td>930.303668</td>
      <td>16.427083</td>
      <td>7.480208</td>
      <td>-0.274000</td>
      <td>3.423450</td>
      <td>2.192708</td>
      <td>16.258848</td>
      <td>9.069180</td>
      <td>54.330210</td>
      <td>13168.851697</td>
      <td>1</td>
      <td>0.12</td>
      <td>0.05</td>
      <td>0.83</td>
      <td>12719.089291</td>
      <td>0.368513</td>
      <td>19.528704</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11787</th>
      <td>58.300001</td>
      <td>13.592497</td>
      <td>962.193200</td>
      <td>12.337500</td>
      <td>4.972396</td>
      <td>-0.368250</td>
      <td>11.041480</td>
      <td>7.713488</td>
      <td>55.112382</td>
      <td>33.720203</td>
      <td>75.541652</td>
      <td>89295.865888</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.61</td>
      <td>0.39</td>
      <td>2064.177740</td>
      <td>0.000000</td>
      <td>43.801797</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13078</th>
      <td>73.259273</td>
      <td>27.797975</td>
      <td>45673.266226</td>
      <td>7.850000</td>
      <td>4.606771</td>
      <td>0.575067</td>
      <td>5.679837</td>
      <td>7.339446</td>
      <td>29.279100</td>
      <td>33.668380</td>
      <td>37.761544</td>
      <td>36068.688998</td>
      <td>1</td>
      <td>0.58</td>
      <td>0.20</td>
      <td>0.22</td>
      <td>270572.544877</td>
      <td>0.066457</td>
      <td>40.204053</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40912</th>
      <td>45.000398</td>
      <td>22.089574</td>
      <td>12261.516957</td>
      <td>13.060417</td>
      <td>7.175000</td>
      <td>0.054200</td>
      <td>3.019431</td>
      <td>3.551144</td>
      <td>16.342650</td>
      <td>15.550940</td>
      <td>22.023757</td>
      <td>94201.780356</td>
      <td>1</td>
      <td>0.65</td>
      <td>0.26</td>
      <td>0.09</td>
      <td>65401.137000</td>
      <td>2.043719</td>
      <td>5.417064</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>12863</th>
      <td>59.637022</td>
      <td>46.784325</td>
      <td>3408.319457</td>
      <td>8.012500</td>
      <td>4.380208</td>
      <td>-0.303818</td>
      <td>5.977114</td>
      <td>3.937252</td>
      <td>29.124319</td>
      <td>16.237819</td>
      <td>69.981072</td>
      <td>25551.064830</td>
      <td>1</td>
      <td>0.04</td>
      <td>0.55</td>
      <td>0.41</td>
      <td>10457.396786</td>
      <td>0.020811</td>
      <td>33.660164</td>
      <td>1</td>
    </tr>
    <tr>
      <th>40886</th>
      <td>45.636009</td>
      <td>2.664347</td>
      <td>2576.956387</td>
      <td>10.502083</td>
      <td>6.399479</td>
      <td>0.090667</td>
      <td>1.593387</td>
      <td>1.504150</td>
      <td>9.580483</td>
      <td>7.237524</td>
      <td>9.295925</td>
      <td>8426.045863</td>
      <td>1</td>
      <td>0.15</td>
      <td>0.01</td>
      <td>0.84</td>
      <td>14043.023422</td>
      <td>0.489396</td>
      <td>6.892182</td>
      <td>1</td>
    </tr>
    <tr>
      <th>707</th>
      <td>58.454575</td>
      <td>22.214850</td>
      <td>2531.702542</td>
      <td>15.068750</td>
      <td>7.422396</td>
      <td>0.032500</td>
      <td>8.040434</td>
      <td>5.532480</td>
      <td>39.034626</td>
      <td>23.157728</td>
      <td>94.637186</td>
      <td>16279.288402</td>
      <td>1</td>
      <td>0.14</td>
      <td>0.22</td>
      <td>0.64</td>
      <td>11078.798147</td>
      <td>0.000000</td>
      <td>3.223924</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11773</th>
      <td>54.906015</td>
      <td>43.292512</td>
      <td>183.611479</td>
      <td>8.879167</td>
      <td>3.808333</td>
      <td>0.055000</td>
      <td>6.953282</td>
      <td>6.640077</td>
      <td>35.870349</td>
      <td>30.475411</td>
      <td>53.573066</td>
      <td>13977.710894</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.15</td>
      <td>0.85</td>
      <td>1077.304325</td>
      <td>0.000000</td>
      <td>31.399046</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7595</th>
      <td>44.337041</td>
      <td>20.560753</td>
      <td>764.805319</td>
      <td>12.816667</td>
      <td>5.881771</td>
      <td>-0.500909</td>
      <td>8.577312</td>
      <td>4.698056</td>
      <td>45.340866</td>
      <td>18.116541</td>
      <td>96.728775</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>3757.401456</td>
      <td>0.000000</td>
      <td>0.007280</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>1293 rows × 20 columns</p>
</div>



### Third step is to train XGBoost regression model for this reduced train data (including damg>10.0%)


```python
# Define bins for data stratification in regression model
bins2 = [0, 1, 10, 20, 50, 101]
samples_per_bin2, binsP2 = np.histogram(
    fliterd_df["percent_houses_damaged"], bins=bins2
)

print(samples_per_bin2)
print(binsP2)
```

    [168 270 373 367 115]
    [  0   1  10  20  50 101]



```python
bin_index2 = np.digitize(fliterd_df["percent_houses_damaged"], bins=binsP2)
```


```python
y_input_strat = bin_index2
```


```python
# Split X and y from dataframe features
X_r = fliterd_df[features]
display(X.columns)
y_r = fliterd_df["percent_houses_damaged"]
```


    Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
           'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
           'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
           'water', 'total_pop', 'percent_houses_damaged_5years'],
          dtype='object')



```python
# XGBoost Reduced Overfitting
xgbR = XGBRegressor(
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

eval_set = [(X_r, y_r)]
xgbR_model = xgbR.fit(X_r, y_r, eval_set=eval_set, verbose=False)
```

    [02:13:50] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.

      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.




    pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.



```python
# Make prediction on train and global test data
y_pred_r = xgbR.predict(X_r)
y_pred_test_total = xgbR.predict(X_test)
```


```python
# Calculate RMSE in total

mse_train_idxR = mean_squared_error(y_r, y_pred_r)
rmse_trainR = np.sqrt(mse_train_idxR)


mse_idxR = mean_squared_error(y_test, y_pred_test_total)
rmseR = np.sqrt(mse_idxR)

print(f"RMSE_test_in_total MR: {rmseR:.2f}")
print(f"RMSE_test_in_total M1: {rmseM1:.2f}")
print(f"RMSE_train_in_reduced: {rmse_trainR:.2f}")
```

    RMSE_test_in_total MR: 15.44
    RMSE_test_in_total M1: 3.08
    RMSE_train_in_reduced: 10.01



```python
# Calculate RMSE per bins
bin_index_r = np.digitize(y_r, bins=bins_eval)

RSME_test_model1R = np.zeros(len(bins_eval) - 1)
for bin_num in range(1, len(bins_eval)):

    # Estimation of RMSE for train data
    mse_train_idxR = mean_squared_error(
        y_r[bin_index_r == bin_num], y_pred_r[bin_index_r == bin_num]
    )
    rmse_trainR = np.sqrt(mse_train_idxR)

    # Estimation of RMSE for test data
    mse_idxR = mean_squared_error(
        y_test[bin_index_test == bin_num], y_pred_test_total[bin_index_test == bin_num]
    )
    RSME_test_model1R[bin_num - 1] = np.sqrt(mse_idxR)

    # print(f"RMSE_test: {rmse:.2f}")
    print(
        f"RMSE_train_reduced [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {rmse_trainR:.2f}"
    )
    print(
        f"RMSE_test_total_MR [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {RSME_test_model1R[bin_num-1]:.2f}"
    )
    print(
        f"RMSE_test_total_M1 [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {RSME_test_model1[bin_num-1]:.2f}"
    )
    RSME_test_model1
    # print(f"RMSE_train: {rmse_train:.2f}")
```

    RMSE_train_reduced [0,1): 11.16
    RMSE_test_total_MR [0,1): 15.60
    RMSE_test_total_M1 [0,1): 1.17
    RMSE_train_reduced [1,10): 8.32
    RMSE_test_total_MR [1,10): 12.26
    RMSE_test_total_M1 [1,10): 4.54
    RMSE_train_reduced [10,20): 4.63
    RMSE_test_total_MR [10,20): 6.80
    RMSE_test_total_M1 [10,20): 9.31
    RMSE_train_reduced [20,50): 9.69
    RMSE_test_total_MR [20,50): 15.19
    RMSE_test_total_M1 [20,50): 19.75
    RMSE_train_reduced [50,101): 20.31
    RMSE_test_total_MR [50,101): 30.13
    RMSE_test_total_M1 [50,101): 33.02


## Last step is to add model combination (model M1 with model MR)


```python
# Check the result of classifier for test set
reduced_test_df = X_test.copy()
```


```python
# joined X_test with countinous target and binary predicted values
reduced_test_df["percent_houses_damaged"] = y_test.values
reduced_test_df["predicted_value"] = y_pred_test

reduced_test_df
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>mean_tri</th>
      <th>std_tri</th>
      <th>mean_elev</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>predicted_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49601</th>
      <td>10.396554</td>
      <td>199.979958</td>
      <td>96.158352</td>
      <td>6.631250</td>
      <td>4.363542</td>
      <td>-0.380000</td>
      <td>9.022485</td>
      <td>3.787582</td>
      <td>50.970692</td>
      <td>13.488631</td>
      <td>34.924242</td>
      <td>4280.863151</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1284.250123</td>
      <td>1.667437</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>33863</th>
      <td>12.159087</td>
      <td>146.270346</td>
      <td>152.360892</td>
      <td>7.714583</td>
      <td>5.008854</td>
      <td>-0.637500</td>
      <td>11.081625</td>
      <td>6.365845</td>
      <td>54.018911</td>
      <td>26.881018</td>
      <td>568.237415</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>126.263593</td>
      <td>0.007442</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23361</th>
      <td>14.495280</td>
      <td>152.151368</td>
      <td>3001.479050</td>
      <td>5.568750</td>
      <td>3.439062</td>
      <td>-0.490250</td>
      <td>11.923229</td>
      <td>11.107064</td>
      <td>57.110345</td>
      <td>55.768360</td>
      <td>568.612936</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.08</td>
      <td>0.92</td>
      <td>0.00</td>
      <td>15023.231141</td>
      <td>0.086431</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>868</th>
      <td>46.227039</td>
      <td>45.016424</td>
      <td>8.198027</td>
      <td>22.633333</td>
      <td>9.579688</td>
      <td>-0.213039</td>
      <td>0.848832</td>
      <td>0.441849</td>
      <td>5.880668</td>
      <td>2.697199</td>
      <td>5.285714</td>
      <td>2068.237335</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.98</td>
      <td>109.011122</td>
      <td>0.000000</td>
      <td>28.078216</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13086</th>
      <td>54.615895</td>
      <td>59.662645</td>
      <td>11131.247660</td>
      <td>11.627083</td>
      <td>8.856771</td>
      <td>-0.046429</td>
      <td>12.245994</td>
      <td>8.618815</td>
      <td>60.532437</td>
      <td>39.613202</td>
      <td>215.824860</td>
      <td>18705.584002</td>
      <td>1</td>
      <td>0.37</td>
      <td>0.55</td>
      <td>0.08</td>
      <td>52800.638758</td>
      <td>0.003004</td>
      <td>1.054973</td>
      <td>1</td>
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
    </tr>
    <tr>
      <th>37057</th>
      <td>14.582867</td>
      <td>272.365088</td>
      <td>7.528400</td>
      <td>2.612500</td>
      <td>1.492708</td>
      <td>-0.268000</td>
      <td>8.548323</td>
      <td>6.229575</td>
      <td>41.852405</td>
      <td>26.515616</td>
      <td>293.939758</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>2.361292</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>34995</th>
      <td>28.151356</td>
      <td>98.926726</td>
      <td>3887.811319</td>
      <td>7.581250</td>
      <td>3.271875</td>
      <td>-0.488308</td>
      <td>6.347557</td>
      <td>4.674013</td>
      <td>31.508331</td>
      <td>18.730070</td>
      <td>80.469435</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.06</td>
      <td>0.94</td>
      <td>0.00</td>
      <td>26563.024972</td>
      <td>0.006633</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14737</th>
      <td>21.640331</td>
      <td>139.918881</td>
      <td>1736.128056</td>
      <td>9.329167</td>
      <td>4.877083</td>
      <td>0.244600</td>
      <td>6.914429</td>
      <td>4.618878</td>
      <td>35.696393</td>
      <td>19.693002</td>
      <td>44.132684</td>
      <td>25684.186927</td>
      <td>1</td>
      <td>0.13</td>
      <td>0.11</td>
      <td>0.76</td>
      <td>14250.107287</td>
      <td>0.048113</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11627</th>
      <td>18.899248</td>
      <td>169.396725</td>
      <td>809.229447</td>
      <td>4.327083</td>
      <td>2.413021</td>
      <td>-0.108000</td>
      <td>9.301184</td>
      <td>9.389671</td>
      <td>45.058957</td>
      <td>42.537441</td>
      <td>68.646985</td>
      <td>7531.806226</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>3481.665415</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26463</th>
      <td>14.825179</td>
      <td>211.182884</td>
      <td>655.678011</td>
      <td>11.368750</td>
      <td>10.119271</td>
      <td>-0.626857</td>
      <td>8.904219</td>
      <td>7.294501</td>
      <td>43.494130</td>
      <td>32.695058</td>
      <td>315.664094</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1502.700327</td>
      <td>0.000355</td>
      <td>0.014988</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>9951 rows × 20 columns</p>
</div>




```python
# damaged prediction
fliterd_test_df1 = reduced_test_df[reduced_test_df.predicted_value == 1]

# not damaged prediction
fliterd_test_df0 = reduced_test_df[reduced_test_df.predicted_value == 0]
```


```python
# Use X0 and X1 for the M1 and MR models' predictions
X1 = fliterd_test_df1[features]
X0 = fliterd_test_df0[features]
```


```python
# For the output equal to 1 apply MR to evaluate the performance
y1_pred = xgbR.predict(X1)
y1 = fliterd_test_df1["percent_houses_damaged"]
```


```python
# For the output equal to 0 apply M1 to evaluate the performance
y0_pred = xgb.predict(X0)
y0 = fliterd_test_df0["percent_houses_damaged"]
```


```python
## Combined the two outputs
```


```python
fliterd_test_df0["predicted_percent_damage"] = y0_pred
fliterd_test_df0
```


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy





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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>mean_tri</th>
      <th>std_tri</th>
      <th>...</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>predicted_value</th>
      <th>predicted_percent_damage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49601</th>
      <td>10.396554</td>
      <td>199.979958</td>
      <td>96.158352</td>
      <td>6.631250</td>
      <td>4.363542</td>
      <td>-0.380000</td>
      <td>9.022485</td>
      <td>3.787582</td>
      <td>50.970692</td>
      <td>13.488631</td>
      <td>...</td>
      <td>4280.863151</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1284.250123</td>
      <td>1.667437</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.077989</td>
    </tr>
    <tr>
      <th>33863</th>
      <td>12.159087</td>
      <td>146.270346</td>
      <td>152.360892</td>
      <td>7.714583</td>
      <td>5.008854</td>
      <td>-0.637500</td>
      <td>11.081625</td>
      <td>6.365845</td>
      <td>54.018911</td>
      <td>26.881018</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>126.263593</td>
      <td>0.007442</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.067659</td>
    </tr>
    <tr>
      <th>23361</th>
      <td>14.495280</td>
      <td>152.151368</td>
      <td>3001.479050</td>
      <td>5.568750</td>
      <td>3.439062</td>
      <td>-0.490250</td>
      <td>11.923229</td>
      <td>11.107064</td>
      <td>57.110345</td>
      <td>55.768360</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.08</td>
      <td>0.92</td>
      <td>0.00</td>
      <td>15023.231141</td>
      <td>0.086431</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.031464</td>
    </tr>
    <tr>
      <th>868</th>
      <td>46.227039</td>
      <td>45.016424</td>
      <td>8.198027</td>
      <td>22.633333</td>
      <td>9.579688</td>
      <td>-0.213039</td>
      <td>0.848832</td>
      <td>0.441849</td>
      <td>5.880668</td>
      <td>2.697199</td>
      <td>...</td>
      <td>2068.237335</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.02</td>
      <td>0.98</td>
      <td>109.011122</td>
      <td>0.000000</td>
      <td>28.078216</td>
      <td>0</td>
      <td>4.383256</td>
    </tr>
    <tr>
      <th>21558</th>
      <td>10.797003</td>
      <td>194.379571</td>
      <td>212.663162</td>
      <td>3.683333</td>
      <td>1.591667</td>
      <td>-0.695750</td>
      <td>21.473339</td>
      <td>9.728256</td>
      <td>103.875000</td>
      <td>45.545172</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1950.745470</td>
      <td>0.532006</td>
      <td>0.000000</td>
      <td>0</td>
      <td>-0.021949</td>
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
      <th>37057</th>
      <td>14.582867</td>
      <td>272.365088</td>
      <td>7.528400</td>
      <td>2.612500</td>
      <td>1.492708</td>
      <td>-0.268000</td>
      <td>8.548323</td>
      <td>6.229575</td>
      <td>41.852405</td>
      <td>26.515616</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>2.361292</td>
      <td>0.000000</td>
      <td>0</td>
      <td>-0.010648</td>
    </tr>
    <tr>
      <th>34995</th>
      <td>28.151356</td>
      <td>98.926726</td>
      <td>3887.811319</td>
      <td>7.581250</td>
      <td>3.271875</td>
      <td>-0.488308</td>
      <td>6.347557</td>
      <td>4.674013</td>
      <td>31.508331</td>
      <td>18.730070</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.06</td>
      <td>0.94</td>
      <td>0.00</td>
      <td>26563.024972</td>
      <td>0.006633</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.118362</td>
    </tr>
    <tr>
      <th>14737</th>
      <td>21.640331</td>
      <td>139.918881</td>
      <td>1736.128056</td>
      <td>9.329167</td>
      <td>4.877083</td>
      <td>0.244600</td>
      <td>6.914429</td>
      <td>4.618878</td>
      <td>35.696393</td>
      <td>19.693002</td>
      <td>...</td>
      <td>25684.186927</td>
      <td>1</td>
      <td>0.13</td>
      <td>0.11</td>
      <td>0.76</td>
      <td>14250.107287</td>
      <td>0.048113</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.022514</td>
    </tr>
    <tr>
      <th>11627</th>
      <td>18.899248</td>
      <td>169.396725</td>
      <td>809.229447</td>
      <td>4.327083</td>
      <td>2.413021</td>
      <td>-0.108000</td>
      <td>9.301184</td>
      <td>9.389671</td>
      <td>45.058957</td>
      <td>42.537441</td>
      <td>...</td>
      <td>7531.806226</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>3481.665415</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.077200</td>
    </tr>
    <tr>
      <th>26463</th>
      <td>14.825179</td>
      <td>211.182884</td>
      <td>655.678011</td>
      <td>11.368750</td>
      <td>10.119271</td>
      <td>-0.626857</td>
      <td>8.904219</td>
      <td>7.294501</td>
      <td>43.494130</td>
      <td>32.695058</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>1502.700327</td>
      <td>0.000355</td>
      <td>0.014988</td>
      <td>0</td>
      <td>0.074963</td>
    </tr>
  </tbody>
</table>
<p>9662 rows × 21 columns</p>
</div>




```python
fliterd_test_df1["predicted_percent_damage"] = y1_pred
fliterd_test_df1
```


    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead

    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy





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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>mean_tri</th>
      <th>std_tri</th>
      <th>...</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>predicted_value</th>
      <th>predicted_percent_damage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>13086</th>
      <td>54.615895</td>
      <td>59.662645</td>
      <td>11131.247660</td>
      <td>11.627083</td>
      <td>8.856771</td>
      <td>-0.046429</td>
      <td>12.245994</td>
      <td>8.618815</td>
      <td>60.532437</td>
      <td>39.613202</td>
      <td>...</td>
      <td>18705.584002</td>
      <td>1</td>
      <td>0.370000</td>
      <td>0.550000</td>
      <td>0.080000</td>
      <td>52800.638758</td>
      <td>0.003004</td>
      <td>1.054973</td>
      <td>1</td>
      <td>12.963193</td>
    </tr>
    <tr>
      <th>25327</th>
      <td>57.914400</td>
      <td>23.967321</td>
      <td>3777.509865</td>
      <td>23.145833</td>
      <td>7.442708</td>
      <td>-0.255571</td>
      <td>6.624408</td>
      <td>5.388203</td>
      <td>33.055565</td>
      <td>23.298431</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.200000</td>
      <td>0.800000</td>
      <td>0.000000</td>
      <td>34375.595370</td>
      <td>0.220207</td>
      <td>8.235989</td>
      <td>1</td>
      <td>24.898096</td>
    </tr>
    <tr>
      <th>24441</th>
      <td>57.065985</td>
      <td>23.656993</td>
      <td>6.359082</td>
      <td>29.652083</td>
      <td>11.751563</td>
      <td>-0.291000</td>
      <td>7.761711</td>
      <td>4.006668</td>
      <td>39.335665</td>
      <td>20.571214</td>
      <td>...</td>
      <td>7614.883914</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.990000</td>
      <td>26.183816</td>
      <td>0.005252</td>
      <td>0.312617</td>
      <td>1</td>
      <td>6.329030</td>
    </tr>
    <tr>
      <th>18752</th>
      <td>56.759480</td>
      <td>14.205862</td>
      <td>2866.363044</td>
      <td>6.739583</td>
      <td>5.038542</td>
      <td>-0.069727</td>
      <td>14.809156</td>
      <td>7.506341</td>
      <td>71.822364</td>
      <td>31.962377</td>
      <td>...</td>
      <td>13153.653463</td>
      <td>1</td>
      <td>0.180000</td>
      <td>0.560000</td>
      <td>0.260000</td>
      <td>18630.720582</td>
      <td>0.001808</td>
      <td>12.142118</td>
      <td>1</td>
      <td>8.230263</td>
    </tr>
    <tr>
      <th>35334</th>
      <td>59.161392</td>
      <td>11.245539</td>
      <td>5498.293421</td>
      <td>19.520833</td>
      <td>7.477604</td>
      <td>-0.221333</td>
      <td>11.949656</td>
      <td>8.944561</td>
      <td>58.190334</td>
      <td>39.587512</td>
      <td>...</td>
      <td>20812.246604</td>
      <td>1</td>
      <td>0.260000</td>
      <td>0.430000</td>
      <td>0.310000</td>
      <td>30647.367865</td>
      <td>0.001631</td>
      <td>27.527417</td>
      <td>1</td>
      <td>22.746412</td>
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
      <th>18519</th>
      <td>50.307346</td>
      <td>36.569580</td>
      <td>2835.828430</td>
      <td>11.170833</td>
      <td>4.689063</td>
      <td>-0.568571</td>
      <td>5.250933</td>
      <td>4.907627</td>
      <td>26.505635</td>
      <td>21.275108</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.010000</td>
      <td>0.990000</td>
      <td>0.000000</td>
      <td>15827.071106</td>
      <td>0.005740</td>
      <td>6.484813</td>
      <td>1</td>
      <td>10.533318</td>
    </tr>
    <tr>
      <th>12748</th>
      <td>67.845083</td>
      <td>26.702615</td>
      <td>6098.570466</td>
      <td>16.964583</td>
      <td>6.516667</td>
      <td>-0.270462</td>
      <td>5.565469</td>
      <td>3.809935</td>
      <td>33.778871</td>
      <td>19.031124</td>
      <td>...</td>
      <td>23962.089446</td>
      <td>1</td>
      <td>0.250000</td>
      <td>0.400000</td>
      <td>0.350000</td>
      <td>22459.643218</td>
      <td>0.012958</td>
      <td>34.585345</td>
      <td>1</td>
      <td>31.269945</td>
    </tr>
    <tr>
      <th>44339</th>
      <td>52.180348</td>
      <td>11.334142</td>
      <td>14719.075948</td>
      <td>19.141667</td>
      <td>7.269792</td>
      <td>-0.060727</td>
      <td>2.374585</td>
      <td>3.056920</td>
      <td>12.969910</td>
      <td>13.675511</td>
      <td>...</td>
      <td>2056.872703</td>
      <td>1</td>
      <td>0.572727</td>
      <td>0.418182</td>
      <td>0.009091</td>
      <td>90515.935382</td>
      <td>0.709506</td>
      <td>12.554129</td>
      <td>1</td>
      <td>11.210395</td>
    </tr>
    <tr>
      <th>12273</th>
      <td>64.733917</td>
      <td>23.859533</td>
      <td>680.145259</td>
      <td>8.412500</td>
      <td>5.266667</td>
      <td>-0.100167</td>
      <td>4.016969</td>
      <td>4.098531</td>
      <td>21.618080</td>
      <td>18.588007</td>
      <td>...</td>
      <td>14323.633429</td>
      <td>1</td>
      <td>0.060000</td>
      <td>0.040000</td>
      <td>0.900000</td>
      <td>4659.437934</td>
      <td>0.986445</td>
      <td>55.374130</td>
      <td>1</td>
      <td>32.997231</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>61.431050</td>
      <td>28.594912</td>
      <td>1239.800099</td>
      <td>11.885417</td>
      <td>6.933854</td>
      <td>-0.423500</td>
      <td>8.682745</td>
      <td>4.543843</td>
      <td>39.348646</td>
      <td>18.118345</td>
      <td>...</td>
      <td>18235.404637</td>
      <td>1</td>
      <td>0.110000</td>
      <td>0.180000</td>
      <td>0.710000</td>
      <td>9714.089711</td>
      <td>0.000000</td>
      <td>35.862069</td>
      <td>1</td>
      <td>17.184288</td>
    </tr>
  </tbody>
</table>
<p>289 rows × 21 columns</p>
</div>




```python
# Join two dataframes together

join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])
join_test_dfs
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
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>mean_tri</th>
      <th>std_tri</th>
      <th>...</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>predicted_value</th>
      <th>predicted_percent_damage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>49601</th>
      <td>10.396554</td>
      <td>199.979958</td>
      <td>96.158352</td>
      <td>6.631250</td>
      <td>4.363542</td>
      <td>-0.380000</td>
      <td>9.022485</td>
      <td>3.787582</td>
      <td>50.970692</td>
      <td>13.488631</td>
      <td>...</td>
      <td>4280.863151</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1284.250123</td>
      <td>1.667437</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.077989</td>
    </tr>
    <tr>
      <th>33863</th>
      <td>12.159087</td>
      <td>146.270346</td>
      <td>152.360892</td>
      <td>7.714583</td>
      <td>5.008854</td>
      <td>-0.637500</td>
      <td>11.081625</td>
      <td>6.365845</td>
      <td>54.018911</td>
      <td>26.881018</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>126.263593</td>
      <td>0.007442</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.067659</td>
    </tr>
    <tr>
      <th>23361</th>
      <td>14.495280</td>
      <td>152.151368</td>
      <td>3001.479050</td>
      <td>5.568750</td>
      <td>3.439062</td>
      <td>-0.490250</td>
      <td>11.923229</td>
      <td>11.107064</td>
      <td>57.110345</td>
      <td>55.768360</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.080000</td>
      <td>0.920000</td>
      <td>0.000000</td>
      <td>15023.231141</td>
      <td>0.086431</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.031464</td>
    </tr>
    <tr>
      <th>868</th>
      <td>46.227039</td>
      <td>45.016424</td>
      <td>8.198027</td>
      <td>22.633333</td>
      <td>9.579688</td>
      <td>-0.213039</td>
      <td>0.848832</td>
      <td>0.441849</td>
      <td>5.880668</td>
      <td>2.697199</td>
      <td>...</td>
      <td>2068.237335</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.020000</td>
      <td>0.980000</td>
      <td>109.011122</td>
      <td>0.000000</td>
      <td>28.078216</td>
      <td>0</td>
      <td>4.383256</td>
    </tr>
    <tr>
      <th>21558</th>
      <td>10.797003</td>
      <td>194.379571</td>
      <td>212.663162</td>
      <td>3.683333</td>
      <td>1.591667</td>
      <td>-0.695750</td>
      <td>21.473339</td>
      <td>9.728256</td>
      <td>103.875000</td>
      <td>45.545172</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1950.745470</td>
      <td>0.532006</td>
      <td>0.000000</td>
      <td>0</td>
      <td>-0.021949</td>
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
      <th>18519</th>
      <td>50.307346</td>
      <td>36.569580</td>
      <td>2835.828430</td>
      <td>11.170833</td>
      <td>4.689063</td>
      <td>-0.568571</td>
      <td>5.250933</td>
      <td>4.907627</td>
      <td>26.505635</td>
      <td>21.275108</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.010000</td>
      <td>0.990000</td>
      <td>0.000000</td>
      <td>15827.071106</td>
      <td>0.005740</td>
      <td>6.484813</td>
      <td>1</td>
      <td>10.533318</td>
    </tr>
    <tr>
      <th>12748</th>
      <td>67.845083</td>
      <td>26.702615</td>
      <td>6098.570466</td>
      <td>16.964583</td>
      <td>6.516667</td>
      <td>-0.270462</td>
      <td>5.565469</td>
      <td>3.809935</td>
      <td>33.778871</td>
      <td>19.031124</td>
      <td>...</td>
      <td>23962.089446</td>
      <td>1</td>
      <td>0.250000</td>
      <td>0.400000</td>
      <td>0.350000</td>
      <td>22459.643218</td>
      <td>0.012958</td>
      <td>34.585345</td>
      <td>1</td>
      <td>31.269945</td>
    </tr>
    <tr>
      <th>44339</th>
      <td>52.180348</td>
      <td>11.334142</td>
      <td>14719.075948</td>
      <td>19.141667</td>
      <td>7.269792</td>
      <td>-0.060727</td>
      <td>2.374585</td>
      <td>3.056920</td>
      <td>12.969910</td>
      <td>13.675511</td>
      <td>...</td>
      <td>2056.872703</td>
      <td>1</td>
      <td>0.572727</td>
      <td>0.418182</td>
      <td>0.009091</td>
      <td>90515.935382</td>
      <td>0.709506</td>
      <td>12.554129</td>
      <td>1</td>
      <td>11.210395</td>
    </tr>
    <tr>
      <th>12273</th>
      <td>64.733917</td>
      <td>23.859533</td>
      <td>680.145259</td>
      <td>8.412500</td>
      <td>5.266667</td>
      <td>-0.100167</td>
      <td>4.016969</td>
      <td>4.098531</td>
      <td>21.618080</td>
      <td>18.588007</td>
      <td>...</td>
      <td>14323.633429</td>
      <td>1</td>
      <td>0.060000</td>
      <td>0.040000</td>
      <td>0.900000</td>
      <td>4659.437934</td>
      <td>0.986445</td>
      <td>55.374130</td>
      <td>1</td>
      <td>32.997231</td>
    </tr>
    <tr>
      <th>1136</th>
      <td>61.431050</td>
      <td>28.594912</td>
      <td>1239.800099</td>
      <td>11.885417</td>
      <td>6.933854</td>
      <td>-0.423500</td>
      <td>8.682745</td>
      <td>4.543843</td>
      <td>39.348646</td>
      <td>18.118345</td>
      <td>...</td>
      <td>18235.404637</td>
      <td>1</td>
      <td>0.110000</td>
      <td>0.180000</td>
      <td>0.710000</td>
      <td>9714.089711</td>
      <td>0.000000</td>
      <td>35.862069</td>
      <td>1</td>
      <td>17.184288</td>
    </tr>
  </tbody>
</table>
<p>9951 rows × 21 columns</p>
</div>




```python
# join_test_dfs = join_test_dfs.reset_index(drop=True)
```

### Compare performance of M1 with combined model


```python
# Calculate RMSE in total

mse_combined_model = mean_squared_error(
    join_test_dfs["percent_houses_damaged"], join_test_dfs["predicted_percent_damage"]
)
rmse_combined_model = np.sqrt(mse_combined_model)


print(fg.red + f"RMSE_in_total(combined_model): {rmse_combined_model:.2f}" + fg.rs)
print(f"RMSE_in_total(M1_model): {rmseM1:.2f}")
```

    [31mRMSE_in_total(combined_model): 3.20[39m
    RMSE_in_total(M1_model): 3.08



```python
# Calculate RMSE per bin

y_join = join_test_dfs["percent_houses_damaged"]
y_pred_join = join_test_dfs["predicted_percent_damage"]

bin_index_test = np.digitize(y_join, bins=bins_eval)

RSME_combined_model = np.zeros(len(bins_eval) - 1)

for bin_num in range(1, len(bins_eval)):

    mse_combined_model = mean_squared_error(
        y_join[bin_index_test == bin_num],
        y_pred_join[bin_index_test == bin_num],
    )
    RSME_combined_model[bin_num - 1] = np.sqrt(mse_combined_model)

    print(
        fg.red
        + f"RMSE_combined_model [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {RSME_combined_model[bin_num-1]:.2f}"
        + fg.rs
    )

    print(
        f"RMSE_M1_model       [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {RSME_test_model1[bin_num-1]:.2f}"
    )
    print("\n")
```

    [31mRMSE_combined_model [0,1): 1.55[39m
    RMSE_M1_model       [0,1): 1.17


    [31mRMSE_combined_model [1,10): 5.56[39m
    RMSE_M1_model       [1,10): 4.54


    [31mRMSE_combined_model [10,20): 9.10[39m
    RMSE_M1_model       [10,20): 9.31


    [31mRMSE_combined_model [20,50): 17.79[39m
    RMSE_M1_model       [20,50): 19.75


    [31mRMSE_combined_model [50,101): 32.35[39m
    RMSE_M1_model       [50,101): 33.02

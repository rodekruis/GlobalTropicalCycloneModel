# Regression Model (Data Resampling)

We utilized the SMOTE technique to address the class imbalance in the data by oversampling the minority class. 
In order to estimate the accuracy of a resampled regression model, we created a binary target variable to serve as an auxiliary variable for resampling the training data. The binary target variable was used solely for the purpose of resampling the data and was ignored during the estimation of the root mean squared error (RMSE) for the new resampled dataset.


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
from collections import defaultdict, Counter
import statistics

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import colorama
from colorama import Fore

from utils import get_training_dataset
```

    /Users/mersedehkooshki/opt/anaconda3/envs/global-storm/lib/python3.8/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      from pandas import MultiIndex, Int64Index



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
# For simplicity in the later steps of the code, I removed last feature from last column and insert it before target
first_column = df.pop("percent_houses_damaged_5years")
df.insert(20, "percent_houses_damaged_5years", first_column)
```


```python
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)
```


```python
# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```


```python
# define a threshold to separate target into damaged and not_damaged
thres = 10.0

for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] >= thres:
        df.at[i, "binary_damage"] = 1
    else:
        df.at[i, "binary_damage"] = 0

df["binary_damage"] = df["binary_damage"].astype("int")
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
      <th>mean_slope</th>
      <th>...</th>
      <th>mean_elev</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>binary_damage</th>
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
      <td>-0.213039</td>
      <td>1.018526</td>
      <td>...</td>
      <td>5.762712</td>
      <td>3445.709753</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>12.799127</td>
      <td>8602.645832</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>8.833333</td>
      <td>5084.012925</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.990000</td>
      <td>197.339034</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>17.530431</td>
      <td>55607.865950</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.310000</td>
      <td>0.690000</td>
      <td>4970.477311</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>31.931338</td>
      <td>35529.342507</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.770000</td>
      <td>0.230000</td>
      <td>12408.594656</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>3.790141</td>
      <td>...</td>
      <td>36.304688</td>
      <td>21559.003490</td>
      <td>1</td>
      <td>0.08</td>
      <td>0.080000</td>
      <td>0.840000</td>
      <td>17619.701390</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>3.532580</td>
      <td>...</td>
      <td>65.687266</td>
      <td>12591.742022</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.420000</td>
      <td>0.580000</td>
      <td>5623.069564</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>4.444498</td>
      <td>...</td>
      <td>37.414996</td>
      <td>19740.596834</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.109091</td>
      <td>0.890909</td>
      <td>5912.671746</td>
      <td>0.015207</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>5.816195</td>
      <td>...</td>
      <td>105.812452</td>
      <td>26363.303778</td>
      <td>1</td>
      <td>0.03</td>
      <td>0.250000</td>
      <td>0.720000</td>
      <td>11254.164413</td>
      <td>0.020806</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>6.730992</td>
      <td>...</td>
      <td>89.696703</td>
      <td>9359.492382</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.070000</td>
      <td>0.930000</td>
      <td>3188.718115</td>
      <td>0.001050</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>141258 rows × 23 columns</p>
</div>




```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)].reset_index(drop=True)
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
      <th>...</th>
      <th>mean_elev</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>binary_damage</th>
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
      <td>...</td>
      <td>42.218750</td>
      <td>5303.659490</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>...</td>
      <td>72.283154</td>
      <td>61015.543599</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.14</td>
      <td>0.86</td>
      <td>276.871504</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>...</td>
      <td>102.215198</td>
      <td>66707.438070</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>0.89</td>
      <td>448.539453</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>...</td>
      <td>58.988877</td>
      <td>53841.050168</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>2101.708435</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>...</td>
      <td>111.386527</td>
      <td>87378.257957</td>
      <td>1</td>
      <td>0.07</td>
      <td>0.46</td>
      <td>0.47</td>
      <td>11632.726327</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# Define bin
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
print(samples_per_bin2)
print(binsP2)
```

    [38901  7232  2552   925   144]
    [0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]



```python
# Check the bins' intervalls
df["binary_damage"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    48685
    (9e-05, 1.0]        1069
    (1.0, 10.0]            0
    (10.0, 50.0]           0
    (50.0, 101.0]          0
    Name: binary_damage, dtype: int64




```python
df["percent_houses_damaged"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    38901
    (9e-05, 1.0]        7232
    (1.0, 10.0]         2552
    (10.0, 50.0]         925
    (50.0, 101.0]        144
    Name: percent_houses_damaged, dtype: int64




```python
bin_index2 = np.digitize(df["percent_houses_damaged"], bins=binsP2)
```


```python
y_input_strat = bin_index2
```


```python
""" We keep the continuous target in the dataset, since we only want to use binary target to resample dataset and 
after that we will remove it and use continuous target as the main target 
"""
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
    "percent_houses_damaged",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["binary_damage"]
```


    Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
           'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
           'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
           'water', 'total_pop', 'percent_houses_damaged_5years',
           'percent_houses_damaged'],
          dtype='object')



```python
X_train, X_test, y_train, y_test = train_test_split(
    X, df["binary_damage"], stratify=y_input_strat, test_size=0.2
)
```


```python
# Check train data before resampling
print(Counter(y_train))
```

    Counter({0: 38948, 1: 855})



```python
# Create an oversampled training data
smote = SMOTE()
# random_state=101
X_train, y_train = smote.fit_resample(X_train, y_train)
```


```python
# Check train data after resampling
print(Counter(y_train))
```

    Counter({0: 38948, 1: 38948})



```python
# Insert X_train into a df
df_train = pd.DataFrame(X_train)
df_train
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.091462</td>
      <td>267.564260</td>
      <td>1888.035999</td>
      <td>7.241667</td>
      <td>4.452604</td>
      <td>-0.322769</td>
      <td>14.272665</td>
      <td>7.231711</td>
      <td>67.528004</td>
      <td>32.019718</td>
      <td>396.047333</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.040000</td>
      <td>0.960000</td>
      <td>0.000000</td>
      <td>12861.061324</td>
      <td>0.000000</td>
      <td>0.003449</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.752962</td>
      <td>204.558764</td>
      <td>3600.495119</td>
      <td>0.795833</td>
      <td>0.740625</td>
      <td>-0.514833</td>
      <td>12.137599</td>
      <td>7.983075</td>
      <td>60.112006</td>
      <td>34.609163</td>
      <td>304.289529</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.230000</td>
      <td>0.770000</td>
      <td>0.000000</td>
      <td>23051.899417</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>46.597243</td>
      <td>6.121964</td>
      <td>6433.875299</td>
      <td>9.081250</td>
      <td>5.507812</td>
      <td>-0.186000</td>
      <td>3.046346</td>
      <td>3.198107</td>
      <td>16.860475</td>
      <td>14.204030</td>
      <td>131.276416</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.090000</td>
      <td>0.910000</td>
      <td>0.000000</td>
      <td>31042.079993</td>
      <td>0.000000</td>
      <td>8.057337</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16.448354</td>
      <td>214.405180</td>
      <td>15.155830</td>
      <td>0.791667</td>
      <td>0.286458</td>
      <td>-0.622500</td>
      <td>15.823127</td>
      <td>7.900435</td>
      <td>76.502681</td>
      <td>34.260587</td>
      <td>731.268288</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>879.075902</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>14.153690</td>
      <td>261.538266</td>
      <td>14023.409634</td>
      <td>19.052083</td>
      <td>5.804688</td>
      <td>0.120800</td>
      <td>3.805607</td>
      <td>4.080335</td>
      <td>18.551360</td>
      <td>17.664696</td>
      <td>213.776928</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.540000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>77622.651795</td>
      <td>0.353812</td>
      <td>0.000000</td>
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
    </tr>
    <tr>
      <th>77891</th>
      <td>52.184393</td>
      <td>43.073901</td>
      <td>603.521967</td>
      <td>17.185023</td>
      <td>7.664748</td>
      <td>-0.215479</td>
      <td>9.959980</td>
      <td>5.449229</td>
      <td>46.489727</td>
      <td>23.732881</td>
      <td>92.608417</td>
      <td>9037.678125</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.097259</td>
      <td>0.902741</td>
      <td>1833.428817</td>
      <td>0.699718</td>
      <td>39.686053</td>
    </tr>
    <tr>
      <th>77892</th>
      <td>59.963897</td>
      <td>20.913103</td>
      <td>14198.433042</td>
      <td>19.667743</td>
      <td>10.068214</td>
      <td>0.212390</td>
      <td>1.664777</td>
      <td>2.594430</td>
      <td>9.490436</td>
      <td>11.775097</td>
      <td>22.516089</td>
      <td>10176.087806</td>
      <td>1</td>
      <td>0.632858</td>
      <td>0.312375</td>
      <td>0.054767</td>
      <td>97699.536233</td>
      <td>0.176051</td>
      <td>12.491949</td>
    </tr>
    <tr>
      <th>77893</th>
      <td>55.319867</td>
      <td>14.517573</td>
      <td>4948.324447</td>
      <td>34.261990</td>
      <td>16.515017</td>
      <td>0.230219</td>
      <td>3.369494</td>
      <td>2.916289</td>
      <td>17.850594</td>
      <td>14.011837</td>
      <td>56.404645</td>
      <td>15266.441029</td>
      <td>1</td>
      <td>0.143589</td>
      <td>0.167680</td>
      <td>0.688731</td>
      <td>22812.874039</td>
      <td>0.059657</td>
      <td>27.022900</td>
    </tr>
    <tr>
      <th>77894</th>
      <td>40.940624</td>
      <td>3.437480</td>
      <td>1077.478451</td>
      <td>11.954477</td>
      <td>5.602410</td>
      <td>-0.596445</td>
      <td>6.330113</td>
      <td>4.686753</td>
      <td>34.260929</td>
      <td>20.172359</td>
      <td>136.720915</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.000041</td>
      <td>0.999959</td>
      <td>0.000000</td>
      <td>8119.557321</td>
      <td>0.000053</td>
      <td>11.232933</td>
    </tr>
    <tr>
      <th>77895</th>
      <td>60.585341</td>
      <td>1.689753</td>
      <td>2654.446779</td>
      <td>11.227807</td>
      <td>5.517647</td>
      <td>-0.476329</td>
      <td>9.516706</td>
      <td>7.192640</td>
      <td>46.136482</td>
      <td>31.122338</td>
      <td>201.564474</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.087390</td>
      <td>0.912610</td>
      <td>0.000000</td>
      <td>19710.921151</td>
      <td>0.005914</td>
      <td>52.503305</td>
    </tr>
  </tbody>
</table>
<p>77896 rows × 19 columns</p>
</div>




```python
# Insert test set into a df
df_test = pd.DataFrame(X_test)
df_test
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>6816</th>
      <td>44.048350</td>
      <td>61.786077</td>
      <td>814.542348</td>
      <td>5.829167</td>
      <td>2.683854</td>
      <td>-0.454500</td>
      <td>21.786844</td>
      <td>8.494699</td>
      <td>106.836137</td>
      <td>35.940904</td>
      <td>1348.503859</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.05</td>
      <td>0.950000</td>
      <td>0.000000</td>
      <td>4218.317238</td>
      <td>0.000000</td>
      <td>0.073563</td>
    </tr>
    <tr>
      <th>14330</th>
      <td>34.975646</td>
      <td>72.507191</td>
      <td>17081.650715</td>
      <td>5.935417</td>
      <td>3.106250</td>
      <td>0.098400</td>
      <td>1.069686</td>
      <td>1.063322</td>
      <td>6.487904</td>
      <td>4.885841</td>
      <td>54.366915</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.40</td>
      <td>0.600000</td>
      <td>0.000000</td>
      <td>83849.176804</td>
      <td>0.000159</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4035</th>
      <td>8.559898</td>
      <td>259.887791</td>
      <td>6802.568528</td>
      <td>5.604167</td>
      <td>2.353125</td>
      <td>-0.051875</td>
      <td>2.987586</td>
      <td>4.980522</td>
      <td>15.012676</td>
      <td>22.327500</td>
      <td>90.466566</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.30</td>
      <td>0.700000</td>
      <td>0.000000</td>
      <td>31044.811399</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>46217</th>
      <td>8.608691</td>
      <td>69.435436</td>
      <td>2844.830584</td>
      <td>4.158333</td>
      <td>3.081250</td>
      <td>-0.213333</td>
      <td>0.582979</td>
      <td>0.373018</td>
      <td>4.351621</td>
      <td>1.966492</td>
      <td>4.763869</td>
      <td>9627.413274</td>
      <td>1</td>
      <td>0.08</td>
      <td>0.000000</td>
      <td>0.920000</td>
      <td>20552.931756</td>
      <td>0.048214</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3920</th>
      <td>16.340823</td>
      <td>2.839519</td>
      <td>16065.139352</td>
      <td>10.943750</td>
      <td>6.406771</td>
      <td>0.102920</td>
      <td>0.891633</td>
      <td>0.762243</td>
      <td>5.746411</td>
      <td>3.327205</td>
      <td>23.140974</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.56</td>
      <td>0.440000</td>
      <td>0.000000</td>
      <td>73292.845633</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
    </tr>
    <tr>
      <th>2876</th>
      <td>23.048018</td>
      <td>218.276908</td>
      <td>5647.655383</td>
      <td>5.081250</td>
      <td>3.626562</td>
      <td>-0.469800</td>
      <td>8.637702</td>
      <td>7.042663</td>
      <td>42.553258</td>
      <td>31.234748</td>
      <td>342.079981</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.19</td>
      <td>0.810000</td>
      <td>0.000000</td>
      <td>36805.207329</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7914</th>
      <td>13.740308</td>
      <td>299.626696</td>
      <td>2363.977108</td>
      <td>1.277083</td>
      <td>0.690104</td>
      <td>0.103750</td>
      <td>3.299979</td>
      <td>4.458310</td>
      <td>17.104493</td>
      <td>20.096613</td>
      <td>23.764076</td>
      <td>13312.672371</td>
      <td>1</td>
      <td>0.16</td>
      <td>0.130000</td>
      <td>0.710000</td>
      <td>23067.075843</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34101</th>
      <td>9.093119</td>
      <td>288.706071</td>
      <td>1056.625095</td>
      <td>0.554167</td>
      <td>0.228646</td>
      <td>-0.308857</td>
      <td>8.594823</td>
      <td>6.453575</td>
      <td>42.721060</td>
      <td>28.579737</td>
      <td>50.229820</td>
      <td>29180.717993</td>
      <td>1</td>
      <td>0.01</td>
      <td>0.270000</td>
      <td>0.720000</td>
      <td>8824.907015</td>
      <td>0.002958</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>39677</th>
      <td>52.629388</td>
      <td>23.155755</td>
      <td>1653.440526</td>
      <td>8.487500</td>
      <td>5.494792</td>
      <td>-0.068800</td>
      <td>9.129843</td>
      <td>5.777991</td>
      <td>45.061471</td>
      <td>24.800281</td>
      <td>82.043315</td>
      <td>36086.646548</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.290909</td>
      <td>0.709091</td>
      <td>9664.064570</td>
      <td>0.522671</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>41364</th>
      <td>12.878433</td>
      <td>235.678687</td>
      <td>1742.336683</td>
      <td>8.533333</td>
      <td>4.361806</td>
      <td>-0.468000</td>
      <td>12.208841</td>
      <td>6.636289</td>
      <td>57.184593</td>
      <td>26.941660</td>
      <td>267.487467</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>6890.718918</td>
      <td>0.302255</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>9951 rows × 19 columns</p>
</div>




```python
# Show histogram of damage for train data
df_train.hist(column="percent_houses_damaged", figsize=(4, 3))
plt.title("percent_houses_damaged for train_data")

# Show histogram of damage for test data
df_test.hist(column="percent_houses_damaged", figsize=(4, 3))
plt.title("percent_houses_damaged for test_data")
```




    Text(0.5, 1.0, 'percent_houses_damaged for test_data')




    
![png](output_21_1.png)
    



    
![png](output_21_2.png)
    



```python
# We use this features to train regression model
features_new = [
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
```


```python
# Split X and y from train dataframe
X_train = df_train[features_new]
display(X_train.columns)
y_train = df_train["percent_houses_damaged"]
```


    Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
           'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
           'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
           'water', 'total_pop', 'percent_houses_damaged_5years'],
          dtype='object')



```python
# Split X and y from test dataframe
X_test = df_test[features_new]
display(X_test.columns)
y_test = df_test["percent_houses_damaged"]
```


    Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
           'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
           'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
           'water', 'total_pop', 'percent_houses_damaged_5years'],
          dtype='object')



```python
sc = preprocessing.StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```


```python
X_train = X_train_sc
X_test = X_test_sc
y_train = y_train
y_test = y_test
```


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


print(f"LREG Root mean squared error(training): {rmse_train_LREG:.2f}")
print(f"LREG Root mean squared error(test): {rmse_LREG:.2f}")


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

print(f"RMSE_train_in_total: {rmse_train:.2f}")
print(f"RMSE_test_in_total: {rmse:.2f}")


# Calculate RMSE per bins

bin_index_test = np.digitize(y_test, bins=binsP2)
bin_index_train = np.digitize(y_train, bins=binsP2)


for bin_num in range(1, 6):

    # Estimation of RMSE for train data
    mse_train_idx = mean_squared_error(
        y_train[bin_index_train == bin_num],
        y_pred_train_clipped[bin_index_train == bin_num],
    )
    rmse_train = np.sqrt(mse_train_idx)

    # Estimation of RMSE for test data
    mse_idx = mean_squared_error(
        y_test[bin_index_test == bin_num], y_pred_clipped[bin_index_test == bin_num]
    )
    rmse = np.sqrt(mse_idx)

    print(f"RMSE_train: {rmse_train:.2f}")
    print(f"RMSE_test: {rmse:.2f}")
```

    [21:51:17] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.616
    Model:                                OLS   Adj. R-squared:                  0.616
    Method:                     Least Squares   F-statistic:                     7356.
    Date:                    Wed, 22 Mar 2023   Prob (F-statistic):               0.00
    Time:                            21:51:22   Log-Likelihood:            -2.9724e+05
    No. Observations:                   77896   AIC:                         5.945e+05
    Df Residuals:                       77878   BIC:                         5.947e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const         14.4614      0.041    351.698      0.000      14.381      14.542
    x1            16.1189      0.075    214.567      0.000      15.972      16.266
    x2             3.2518      0.076     42.701      0.000       3.103       3.401
    x3             0.4538      0.145      3.131      0.002       0.170       0.738
    x4             1.9351      0.103     18.743      0.000       1.733       2.137
    x5            -1.7601      0.101    -17.472      0.000      -1.958      -1.563
    x6             0.0751      0.067      1.121      0.262      -0.056       0.206
    x7            -8.2655      0.764    -10.824      0.000      -9.762      -6.769
    x8             0.2073      0.357      0.581      0.561      -0.492       0.907
    x9             7.1466      0.725      9.860      0.000       5.726       8.567
    x10           -0.0323      0.312     -0.103      0.918      -0.644       0.580
    x11           -0.1445      0.085     -1.695      0.090      -0.312       0.023
    x12            0.6583      0.049     13.382      0.000       0.562       0.755
    x13           -1.0258      0.075    -13.698      0.000      -1.173      -0.879
    x14        -3.264e+12   4.47e+12     -0.731      0.465    -1.2e+13    5.49e+12
    x15        -6.081e+12   8.32e+12     -0.731      0.465   -2.24e+13    1.02e+13
    x16        -6.278e+12   8.59e+12     -0.731      0.465   -2.31e+13    1.06e+13
    x17           -0.3668      0.144     -2.551      0.011      -0.649      -0.085
    x18            0.4351      0.040     10.986      0.000       0.357       0.513
    ==============================================================================
    Omnibus:                    20014.673   Durbin-Watson:                   1.903
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):            82746.608
    Skew:                           1.219   Prob(JB):                         0.00
    Kurtosis:                       7.422   Cond. No.                     7.79e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 7.41e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    LREG Root mean squared error(training): 10.99
    LREG Root mean squared error(test): 7.90
    RMSE_train_in_total: 8.08
    RMSE_test_in_total: 5.17
    RMSE_train: 3.03
    RMSE_test: 2.96
    RMSE_train: 6.95
    RMSE_test: 6.98
    RMSE_train: 11.66
    RMSE_test: 11.74
    RMSE_train: 8.51
    RMSE_test: 10.78
    RMSE_train: 21.62
    RMSE_test: 35.09


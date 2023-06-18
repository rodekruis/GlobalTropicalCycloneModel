## 2SG-Global+
#### Converting from grid-based to municipality-based


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
import statistics
import warnings


from math import sqrt
from collections import defaultdict
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

from utils import get_training_dataset, weight_file
```

    pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.



```python
# Hide all warnings
warnings.filterwarnings("ignore")
```


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
      <th>strong_roof_strong_wall</th>
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
      <td>22.580645</td>
      <td>...</td>
      <td>2.699781</td>
      <td>5.762712</td>
      <td>3445.709753</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>2.639401</td>
      <td>...</td>
      <td>4.585088</td>
      <td>12.799127</td>
      <td>8602.645832</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>2.639401</td>
      <td>...</td>
      <td>1.527495</td>
      <td>8.833333</td>
      <td>5084.012925</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.990000</td>
      <td>197.339034</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>2.639401</td>
      <td>...</td>
      <td>11.677657</td>
      <td>17.530431</td>
      <td>55607.865950</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.310000</td>
      <td>0.690000</td>
      <td>4970.477311</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>2.639401</td>
      <td>...</td>
      <td>17.074011</td>
      <td>31.931338</td>
      <td>35529.342507</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.770000</td>
      <td>0.230000</td>
      <td>12408.594656</td>
      <td>0.0</td>
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
      <td>44.762048</td>
      <td>...</td>
      <td>18.012771</td>
      <td>36.304688</td>
      <td>21559.003490</td>
      <td>1</td>
      <td>0.08</td>
      <td>0.080000</td>
      <td>0.840000</td>
      <td>17619.701390</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>44.762048</td>
      <td>...</td>
      <td>13.163042</td>
      <td>65.687266</td>
      <td>12591.742022</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.420000</td>
      <td>0.580000</td>
      <td>5623.069564</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>25.078318</td>
      <td>...</td>
      <td>10.901755</td>
      <td>37.414996</td>
      <td>19740.596834</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.109091</td>
      <td>0.890909</td>
      <td>5912.671746</td>
      <td>0.0</td>
      <td>0.015207</td>
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
      <td>16.796996</td>
      <td>...</td>
      <td>17.917650</td>
      <td>105.812452</td>
      <td>26363.303778</td>
      <td>1</td>
      <td>0.03</td>
      <td>0.250000</td>
      <td>0.720000</td>
      <td>11254.164413</td>
      <td>0.0</td>
      <td>0.020806</td>
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
      <td>17.464316</td>
      <td>...</td>
      <td>17.010867</td>
      <td>89.696703</td>
      <td>9359.492382</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.070000</td>
      <td>0.930000</td>
      <td>3188.718115</td>
      <td>0.0</td>
      <td>0.001050</td>
    </tr>
  </tbody>
</table>
<p>141258 rows × 31 columns</p>
</div>




```python
# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```


```python
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)
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
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
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
      <th>strong_roof_strong_wall</th>
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
      <th>138</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>8284</td>
      <td>12.460039</td>
      <td>275.018491</td>
      <td>0.670833</td>
      <td>0.313021</td>
      <td>0.479848</td>
      <td>-0.213039</td>
      <td>31.336503</td>
      <td>...</td>
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
      <th>139</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>8286</td>
      <td>11.428974</td>
      <td>297.027578</td>
      <td>0.929167</td>
      <td>0.343229</td>
      <td>55.649739</td>
      <td>0.206000</td>
      <td>23.447758</td>
      <td>...</td>
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
      <th>148</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>8450</td>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>0.716667</td>
      <td>0.424479</td>
      <td>8.157414</td>
      <td>-0.636000</td>
      <td>31.336503</td>
      <td>...</td>
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
      <th>149</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>8451</td>
      <td>12.511864</td>
      <td>273.639330</td>
      <td>0.568750</td>
      <td>0.336979</td>
      <td>88.292015</td>
      <td>-0.227500</td>
      <td>31.336503</td>
      <td>...</td>
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
      <th>150</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>8452</td>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>0.589583</td>
      <td>0.290625</td>
      <td>962.766739</td>
      <td>-0.299667</td>
      <td>23.546053</td>
      <td>...</td>
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
<p>5 rows × 31 columns</p>
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
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
      <th>strong_roof_salvage_wall</th>
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
      <th>138</th>
      <td>DURIAN</td>
      <td>12.460039</td>
      <td>275.018491</td>
      <td>0.670833</td>
      <td>0.313021</td>
      <td>0.479848</td>
      <td>-0.213039</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>...</td>
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
      <th>139</th>
      <td>DURIAN</td>
      <td>11.428974</td>
      <td>297.027578</td>
      <td>0.929167</td>
      <td>0.343229</td>
      <td>55.649739</td>
      <td>0.206000</td>
      <td>23.447758</td>
      <td>23.591571</td>
      <td>0.037516</td>
      <td>...</td>
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
      <th>148</th>
      <td>DURIAN</td>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>0.716667</td>
      <td>0.424479</td>
      <td>8.157414</td>
      <td>-0.636000</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>...</td>
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
      <th>149</th>
      <td>DURIAN</td>
      <td>12.511864</td>
      <td>273.639330</td>
      <td>0.568750</td>
      <td>0.336979</td>
      <td>88.292015</td>
      <td>-0.227500</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>...</td>
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
      <th>150</th>
      <td>DURIAN</td>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>0.589583</td>
      <td>0.290625</td>
      <td>962.766739</td>
      <td>-0.299667</td>
      <td>23.546053</td>
      <td>23.660429</td>
      <td>0.037576</td>
      <td>...</td>
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
<p>5 rows × 29 columns</p>
</div>



```python
# Specify features
features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "rwi",
    # "strong_roof_strong_wall",
    # "strong_roof_light_wall",
    # "strong_roof_salvage_wall",
    # "light_roof_strong_wall",
    # "light_roof_light_wall",
    # "light_roof_salvage_wall",
    # "salvaged_roof_strong_wall",
    # "salvaged_roof_light_wall",
    # "salvaged_roof_salvage_wall",
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
X = df_data[features]
display(X.columns)
y = df_data["percent_houses_damaged"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```


    Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
           'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
           'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
           'water', 'total_pop', 'percent_houses_damaged_5years'],
          dtype='object')



```python
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df_data["percent_houses_damaged"], bins=bins2)
```


```python
# Define range of for loop
num_exp = 20

# Define number of bins
num_bins = len(bins2)
```


```python
M1_RMSE_lst = defaultdict(list)
Combined_RMSE_lst = defaultdict(list)
```


```python
# Define empty list to save RMSE in combined model
test_RMSE_lst = np.zeros(num_exp)
test_RMSE_bin = np.zeros((num_exp, num_bins))

# Define empty list to save RMSE in model1
test_RMSE_lst_M1 = np.zeros(num_exp)
test_RMSE_bin_M1 = np.zeros((num_exp, num_bins))
```


```python
# Defin two lists to save RMSE and Average Error

RMSE = defaultdict(list)
AVE = defaultdict(list)
```


```python
for run_ix in range(num_exp):

    bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
    y_input_strat = bin_index2

    X = df_data[features]
    y = df_data["percent_houses_damaged"]

    # Define train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, df_data["percent_houses_damaged"], test_size=0.2, stratify=y_input_strat
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
        verbosity=0,
        eval_metric=["rmse", "logloss"],
        random_state=0,
    )

    eval_set = [(X_train, y_train)]
    xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Make prediction on train and test data
    y_pred_train = xgb.predict(X_train)
    y_pred = xgb.predict(X_test)

    # Calculate RMSE in total
    mse_train_idx = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train_idx)

    mse_idx = mean_squared_error(y_test, y_pred)
    rmseM1 = np.sqrt(mse_idx)

    # Add total RMSE of Model1 to the list
    test_RMSE_lst_M1[run_ix] = rmseM1

    # Calculate RMSE per bins
    bin_index_test = np.digitize(y_test, bins=binsP2)
    bin_index_train = np.digitize(y_train, bins=binsP2)

    RSME_test_model1 = np.zeros(num_bins - 1)

    for bin_num in range(1, num_bins):

        # Estimation of RMSE for train data
        mse_train_idx = mean_squared_error(
            y_train[bin_index_train == bin_num],
            y_pred_train[bin_index_train == bin_num],
        )
        rmse_train = np.sqrt(mse_train_idx)

        # Estimation of RMSE for test data
        mse_idx = mean_squared_error(
            y_test[bin_index_test == bin_num], y_pred[bin_index_test == bin_num]
        )

        RSME_test_model1 = np.sqrt(mse_idx)

        # Add RMSE of Model1 to the list of each bin
        test_RMSE_bin_M1[run_ix, bin_num] = RSME_test_model1
        M1_RMSE_lst[bin_num].append(RSME_test_model1)

    ## Second step is to train XGBoost Binary model for same train data

    # Define a threshold to separate target into damaged and not_damaged
    thres = 10.0
    y_test_bool = y_test >= thres
    y_train_bool = y_train >= thres
    y_test_bin = (y_test_bool) * 1
    y_train_bin = (y_train_bool) * 1

    sum(y_train_bin)

    # Define undersampling strategy
    under = RandomUnderSampler(sampling_strategy=0.1)
    # Fit and apply the transform
    X_train_us, y_train_us = under.fit_resample(X_train, y_train_bin)

    # Use XGBClassifier as a Machine Learning model to fit the data
    xgb_model = XGBClassifier(eval_metric=["error", "logloss"])

    eval_set = [(X_train, y_train_bin)]
    xgb_model.fit(
        X_train_us,
        y_train_us,
        eval_set=eval_set,
        verbose=False,
    )

    # Make prediction on test data and print Confusion Matrix
    y_pred_test = xgb_model.predict(X_test)
    cm = confusion_matrix(y_test_bin, y_pred_test)

    # Make prediction on train data and print Confusion Matrix
    y_pred_train = xgb_model.predict(X_train)
    cm = confusion_matrix(y_train_bin, y_pred_train)

    reduced_df = X_train.copy()

    reduced_df["percent_houses_damaged"] = y_train.values
    reduced_df["predicted_value"] = y_pred_train

    fliterd_df = reduced_df[reduced_df.predicted_value == 1]

    ### Third step is to train XGBoost regression model for this reduced train data (including damg>10.0%)
    bin_index2 = np.digitize(fliterd_df["percent_houses_damaged"], bins=binsP2)
    y_input_strat = bin_index2

    # Split X and y from dataframe features
    X_r = fliterd_df[features]
    y_r = fliterd_df["percent_houses_damaged"]

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
        verbosity=0,
        eval_metric=["rmse", "logloss"],
        random_state=0,
    )

    eval_set = [(X_r, y_r)]
    xgbR_model = xgbR.fit(X_r, y_r, eval_set=eval_set, verbose=False)

    # Make prediction on train and global test data
    y_pred_r = xgbR.predict(X_r)
    y_pred_test_total = xgbR.predict(X_test)

    # Calculate RMSE in total

    mse_train_idxR = mean_squared_error(y_r, y_pred_r)
    rmse_trainR = np.sqrt(mse_train_idxR)

    mse_idxR = mean_squared_error(y_test, y_pred_test_total)
    rmseR = np.sqrt(mse_idxR)

    # Calculate RMSE per bins
    bin_index_r = np.digitize(y_r, bins=binsP2)

    RSME_test_model1R = np.zeros(num_bins - 1)

    for bin_num in range(1, num_bins):

        # Estimation of RMSE for train data
        mse_train_idxR = mean_squared_error(
            y_r[bin_index_r == bin_num], y_pred_r[bin_index_r == bin_num]
        )
        rmse_trainR = np.sqrt(mse_train_idxR)

        # Estimation of RMSE for test data
        mse_idxR = mean_squared_error(
            y_test[bin_index_test == bin_num],
            y_pred_test_total[bin_index_test == bin_num],
        )
        RSME_test_model1R[bin_num - 1] = np.sqrt(mse_idxR)

    #### Last step is to add model combination (model M1 with model MR)

    # Check the result of classifier for test set
    reduced_test_df = X_test.copy()

    # joined X_test with countinous target and binary predicted values
    reduced_test_df["percent_houses_damaged"] = y_test.values
    reduced_test_df["predicted_value"] = y_pred_test

    # damaged prediction
    fliterd_test_df1 = reduced_test_df[reduced_test_df.predicted_value == 1]

    # not damaged prediction
    fliterd_test_df0 = reduced_test_df[reduced_test_df.predicted_value == 0]

    # Use X0 and X1 for the M1 and MR models' predictions
    X1 = fliterd_test_df1[features]
    X0 = fliterd_test_df0[features]

    # For the output equal to 1 apply MR to evaluate the performance
    y1_pred = xgbR.predict(X1)
    y1_pred = y1_pred.clip(0, 100)
    y1 = fliterd_test_df1["percent_houses_damaged"]

    # For the output equal to 0 apply M1 to evaluate the performance
    y0_pred = xgb.predict(X0)
    y0_pred = y0_pred.clip(0, 100)
    y0 = fliterd_test_df0["percent_houses_damaged"]

    fliterd_test_df0["predicted_percent_damage"] = y0_pred
    fliterd_test_df1["predicted_percent_damage"] = y1_pred

    # Join two dataframes together
    join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])

    y_join = join_test_dfs["percent_houses_damaged"]
    y_pred_join = join_test_dfs["predicted_percent_damage"]

    pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])
    pred_df["y_all"] = y_join
    pred_df["y_pred_all"] = y_pred_join

    # bin_index = np.digitize(pred_df["y_all"], bins=binsP2)

    # Join data with y_all and y_all_pred
    df_data_w_pred = pd.merge(pred_df, df_data, left_index=True, right_index=True)
    # Join data with grid_point_id typhoon_year
    df_data_w_pred_grid = pd.merge(
        df[["grid_point_id", "typhoon_year"]],
        df_data_w_pred,
        left_index=True,
        right_index=True,
    )
    df_data_w_pred_grid.sort_values("y_pred_all", ascending=False)

    # join with weights df
    join_df = df_data_w_pred_grid.merge(df_weight, on="grid_point_id", how="left")

    # Indicate where values are valid and not missing
    join_df = join_df.loc[join_df["weight"].notna()]

    # Multiply weight by y_all and y_pred_all
    join_df["weight*y_pred*houses"] = (
        join_df["y_pred_all"] * join_df["weight"] * join_df["total_houses"] / 100
    )
    join_df["weight*y*houses"] = (
        join_df["y_all"] * join_df["weight"] * join_df["total_houses"] / 100
    )
    join_df["weight*houses"] = join_df["weight"] * join_df["total_houses"]

    join_df.sort_values("y_pred_all", ascending=False)

    # Groupby by municipality and typhoon_name with sum as the aggregation function
    agg_df = join_df.groupby(["ADM3_PCODE", "typhoon_name", "typhoon_year"]).agg("sum")

    # Normalize by the sum of the weights
    agg_df["y_pred_norm"] = (
        agg_df["weight*y_pred*houses"] / agg_df["weight*houses"] * 100
    )
    agg_df["y_norm"] = agg_df["weight*y*houses"] / agg_df["weight*houses"] * 100

    # Drop not required column y and y_pred before multiplying by weight
    agg_df.drop("y_all", axis=1, inplace=True)
    agg_df.drop("y_pred_all", axis=1, inplace=True)

    # Remove rows with NaN after normalization
    final_df = agg_df.dropna()

    # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based
    rmse = sqrt(mean_squared_error(final_df["y_norm"], final_df["y_pred_norm"]))
    ave = (final_df["y_pred_norm"] - final_df["y_norm"]).sum() / len(final_df["y_norm"])

    print(f"RMSE for grid_based model: {rmse:.2f}")
    print(f"Average Error for grid_based model: {ave:.2f}")

    RMSE["all"].append(rmse)
    AVE["all"].append(ave)

    bin_index = np.digitize(final_df["y_norm"], bins=binsP2)

    for bin_num in range(1, 6):

        mse_idx = mean_squared_error(
            final_df["y_norm"][bin_index == bin_num],
            final_df["y_pred_norm"][bin_index == bin_num],
        )
        rmse = np.sqrt(mse_idx)

        ave = (
            final_df["y_pred_norm"][bin_index == bin_num]
            - final_df["y_norm"][bin_index == bin_num]
        ).sum() / len(final_df["y_norm"][bin_index == bin_num])

        RMSE[bin_num].append(rmse)
        AVE[bin_num].append(ave)
```

    RMSE for grid_based model: 2.92
    Average Error for grid_based model: 0.05
    RMSE for grid_based model: 2.75
    Average Error for grid_based model: 0.11
    RMSE for grid_based model: 2.73
    Average Error for grid_based model: 0.09
    RMSE for grid_based model: 2.77
    Average Error for grid_based model: 0.13
    RMSE for grid_based model: 2.65
    Average Error for grid_based model: 0.14
    RMSE for grid_based model: 2.68
    Average Error for grid_based model: 0.13
    RMSE for grid_based model: 2.67
    Average Error for grid_based model: 0.12
    RMSE for grid_based model: 2.76
    Average Error for grid_based model: 0.11
    RMSE for grid_based model: 2.84
    Average Error for grid_based model: 0.14
    RMSE for grid_based model: 2.78
    Average Error for grid_based model: 0.09
    RMSE for grid_based model: 2.75
    Average Error for grid_based model: 0.07
    RMSE for grid_based model: 2.65
    Average Error for grid_based model: 0.13
    RMSE for grid_based model: 2.80
    Average Error for grid_based model: 0.16
    RMSE for grid_based model: 2.74
    Average Error for grid_based model: 0.11
    RMSE for grid_based model: 2.85
    Average Error for grid_based model: 0.11
    RMSE for grid_based model: 3.10
    Average Error for grid_based model: 0.11
    RMSE for grid_based model: 2.67
    Average Error for grid_based model: 0.06
    RMSE for grid_based model: 2.98
    Average Error for grid_based model: 0.08
    RMSE for grid_based model: 2.71
    Average Error for grid_based model: 0.12
    RMSE for grid_based model: 2.72
    Average Error for grid_based model: 0.11



```python
# Define a function to plot RMSEs
def rmse_ave_mean(rmse, ave):

    # Mean of RMSE and Standard deviation
    m_rmse = statistics.mean(rmse)
    sd_rmse = statistics.stdev(rmse)

    m_ave = statistics.mean(ave)
    sd_ave = statistics.stdev(ave)

    print(f"mean_RMSE: {m_rmse:.2f}")
    print(f"stdev_RMSE: {sd_rmse:.2f}")

    print(f"mean_average_error: {m_ave:.2f}")
    print(f"stdev_average_error: {sd_ave:.2f}")
```


```python
print("RMSE and Average Error in total", "\n")
rmse_ave_mean(RMSE["all"], AVE["all"])
```

    RMSE and Average Error in total 
    
    mean_RMSE: 2.78
    stdev_RMSE: 0.11
    mean_average_error: 0.11
    stdev_average_error: 0.03



```python
for bin_num in range(1, 6):

    print(f"\n RMSE and Average Error per bin {bin_num}\n")
    rmse_ave_mean(RMSE[bin_num], AVE[bin_num])
```

    
     RMSE and Average Error per bin 1
    
    mean_RMSE: 0.72
    stdev_RMSE: 0.07
    mean_average_error: 0.13
    stdev_average_error: 0.01
    
     RMSE and Average Error per bin 2
    
    mean_RMSE: 1.89
    stdev_RMSE: 0.12
    mean_average_error: 0.63
    stdev_average_error: 0.04
    
     RMSE and Average Error per bin 3
    
    mean_RMSE: 5.38
    stdev_RMSE: 0.50
    mean_average_error: 0.86
    stdev_average_error: 0.28
    
     RMSE and Average Error per bin 4
    
    mean_RMSE: 12.64
    stdev_RMSE: 0.85
    mean_average_error: -4.83
    stdev_average_error: 1.15
    
     RMSE and Average Error per bin 5
    
    mean_RMSE: 32.54
    stdev_RMSE: 3.95
    mean_average_error: -25.45
    stdev_average_error: 3.95



```python

```

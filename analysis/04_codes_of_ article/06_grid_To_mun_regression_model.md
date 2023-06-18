# Converting from grid-based to municipality-based 
#### GridGlobal,  GridGlobal+


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
global_features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    "rainfall_max_24h",
    # "rwi",
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
    # "urban",
    # "rural",
    # "water",
    # "total_pop",
    # "percent_houses_damaged_5years",
]

global_plus_features = [
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
# Ask the user whether to use global features set or global+ features set
feature_set = int(input("Enter 1 for global features, 2 for global+ features: "))
```

    Enter 1 for global features, 2 for global+ features: 2



```python
if feature_set == 1:
    features = global_features
    print(len(features))

elif feature_set == 2:
    features = global_plus_features
    print(len(features))

else:
    print("Invalid input. Please enter 1 or 2")
```

    18



```python
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
bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
```


```python
y_input_strat = bin_index2
```


```python
# Defin two lists to save RMSE and Average Error

RMSE = defaultdict(list)
AVE = defaultdict(list)
```


```python
for i in range(20):
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

    eval_set = [(X_train, y_train)]
    xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    print(est2.summary())

    y_pred_all = xgb.predict(X_test)
    y_pred_all_clipped = y_pred_all.clip(0, 100)

    pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])
    pred_df["y_all"] = y_test
    pred_df["y_pred_all"] = y_pred_all_clipped

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

    [15:37:22] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.203
    Model:                                OLS   Adj. R-squared:                  0.203
    Method:                     Least Squares   F-statistic:                     596.2
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:24   Log-Likelihood:            -1.1613e+05
    No. Observations:                   39803   AIC:                         2.323e+05
    Df Residuals:                       39785   BIC:                         2.325e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8178      0.022     36.443      0.000       0.774       0.862
    x1             2.7983      0.034     82.502      0.000       2.732       2.865
    x2             0.9527      0.034     27.945      0.000       0.886       1.020
    x3             0.0040      0.081      0.049      0.961      -0.155       0.163
    x4             0.4835      0.058      8.287      0.000       0.369       0.598
    x5            -0.4025      0.058     -6.887      0.000      -0.517      -0.288
    x6            -0.0955      0.037     -2.547      0.011      -0.169      -0.022
    x7            -1.3419      0.448     -2.998      0.003      -2.219      -0.465
    x8            -0.2602      0.208     -1.254      0.210      -0.667       0.147
    x9             1.2774      0.431      2.964      0.003       0.433       2.122
    x10            0.3228      0.183      1.763      0.078      -0.036       0.682
    x11           -0.1349      0.046     -2.944      0.003      -0.225      -0.045
    x12            0.1069      0.029      3.744      0.000       0.051       0.163
    x13           -0.0193      0.040     -0.484      0.629      -0.097       0.059
    x14           -0.0472      0.032     -1.459      0.145      -0.111       0.016
    x15           -0.0220      0.025     -0.877      0.380      -0.071       0.027
    x16            0.0497      0.020      2.450      0.014       0.010       0.090
    x17           -0.0170      0.079     -0.215      0.830      -0.172       0.138
    x18            0.0905      0.022      4.092      0.000       0.047       0.134
    ==============================================================================
    Omnibus:                    57748.429   Durbin-Watson:                   1.991
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22311433.558
    Skew:                           8.806   Prob(JB):                         0.00
    Kurtosis:                     117.643   Cond. No.                     1.30e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.38e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.67
    Average Error for grid_based model: 0.01
    [15:37:24] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.205
    Model:                                OLS   Adj. R-squared:                  0.205
    Method:                     Least Squares   F-statistic:                     604.5
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:26   Log-Likelihood:            -1.1593e+05
    No. Observations:                   39803   AIC:                         2.319e+05
    Df Residuals:                       39785   BIC:                         2.321e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8222      0.022     36.824      0.000       0.778       0.866
    x1             2.8124      0.034     83.373      0.000       2.746       2.879
    x2             0.9564      0.034     28.225      0.000       0.890       1.023
    x3             0.0152      0.083      0.184      0.854      -0.147       0.177
    x4             0.5255      0.058      9.033      0.000       0.411       0.639
    x5            -0.4538      0.058     -7.794      0.000      -0.568      -0.340
    x6            -0.1121      0.037     -3.004      0.003      -0.185      -0.039
    x7            -0.9347      0.446     -2.094      0.036      -1.810      -0.060
    x8            -0.2663      0.207     -1.289      0.197      -0.671       0.139
    x9             0.9117      0.430      2.120      0.034       0.069       1.755
    x10            0.2962      0.182      1.626      0.104      -0.061       0.653
    x11           -0.1330      0.046     -2.916      0.004      -0.222      -0.044
    x12            0.1065      0.029      3.717      0.000       0.050       0.163
    x13           -0.0032      0.040     -0.082      0.935      -0.081       0.074
    x14           -0.0070      0.032     -0.218      0.827      -0.070       0.056
    x15           -0.0204      0.025     -0.820      0.412      -0.069       0.028
    x16            0.0246      0.020      1.220      0.223      -0.015       0.064
    x17           -0.0400      0.082     -0.490      0.624      -0.200       0.120
    x18            0.0792      0.022      3.623      0.000       0.036       0.122
    ==============================================================================
    Omnibus:                    57598.928   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21861456.101
    Skew:                           8.768   Prob(JB):                         0.00
    Kurtosis:                     116.465   Cond. No.                     1.79e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 7.3e-26. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.88
    Average Error for grid_based model: -0.02
    [15:37:26] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.200
    Model:                                OLS   Adj. R-squared:                  0.200
    Method:                     Least Squares   F-statistic:                     586.0
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:28   Log-Likelihood:            -1.1685e+05
    No. Observations:                   39803   AIC:                         2.337e+05
    Df Residuals:                       39785   BIC:                         2.339e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8246      0.023     36.087      0.000       0.780       0.869
    x1             2.8371      0.035     82.201      0.000       2.769       2.905
    x2             0.9625      0.035     27.687      0.000       0.894       1.031
    x3             0.0177      0.085      0.209      0.835      -0.148       0.184
    x4             0.4475      0.059      7.523      0.000       0.331       0.564
    x5            -0.3981      0.059     -6.696      0.000      -0.515      -0.282
    x6            -0.0951      0.038     -2.498      0.012      -0.170      -0.020
    x7            -1.2152      0.456     -2.663      0.008      -2.110      -0.321
    x8            -0.1863      0.212     -0.880      0.379      -0.601       0.229
    x9             1.1865      0.440      2.699      0.007       0.325       2.048
    x10            0.2294      0.187      1.229      0.219      -0.136       0.595
    x11           -0.1451      0.047     -3.121      0.002      -0.236      -0.054
    x12            0.1178      0.029      4.025      0.000       0.060       0.175
    x13           -0.0078      0.041     -0.193      0.847      -0.088       0.072
    x14           -0.0265      0.033     -0.808      0.419      -0.091       0.038
    x15           -0.0119      0.025     -0.466      0.641      -0.062       0.038
    x16            0.0275      0.021      1.329      0.184      -0.013       0.068
    x17           -0.0417      0.083     -0.500      0.617      -0.205       0.122
    x18            0.0825      0.022      3.734      0.000       0.039       0.126
    ==============================================================================
    Omnibus:                    58308.756   Durbin-Watson:                   2.003
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23526409.718
    Skew:                           8.963   Prob(JB):                         0.00
    Kurtosis:                     120.747   Cond. No.                     1.51e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.04e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.26
    Average Error for grid_based model: 0.06
    [15:37:28] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.206
    Model:                                OLS   Adj. R-squared:                  0.205
    Method:                     Least Squares   F-statistic:                     606.1
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:30   Log-Likelihood:            -1.1694e+05
    No. Observations:                   39803   AIC:                         2.339e+05
    Df Residuals:                       39785   BIC:                         2.341e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8490      0.029     29.228      0.000       0.792       0.906
    x1             2.8772      0.034     83.445      0.000       2.810       2.945
    x2             0.9720      0.035     28.055      0.000       0.904       1.040
    x3             0.0248      0.082      0.302      0.762      -0.136       0.186
    x4             0.5858      0.060      9.773      0.000       0.468       0.703
    x5            -0.5200      0.060     -8.702      0.000      -0.637      -0.403
    x6            -0.1122      0.039     -2.910      0.004      -0.188      -0.037
    x7            -1.1937      0.458     -2.607      0.009      -2.091      -0.296
    x8            -0.1146      0.214     -0.537      0.591      -0.533       0.304
    x9             1.1263      0.440      2.560      0.010       0.264       1.989
    x10            0.1801      0.188      0.959      0.338      -0.188       0.548
    x11           -0.1406      0.048     -2.915      0.004      -0.235      -0.046
    x12            0.1101      0.030      3.655      0.000       0.051       0.169
    x13           -0.0433      0.043     -1.016      0.309      -0.127       0.040
    x14         -5.55e+11   2.76e+12     -0.201      0.840   -5.96e+12    4.85e+12
    x15        -9.522e+11   4.73e+12     -0.201      0.840   -1.02e+13    8.31e+12
    x16         -9.47e+11    4.7e+12     -0.201      0.840   -1.02e+13    8.27e+12
    x17           -0.0344      0.081     -0.427      0.669      -0.193       0.124
    x18            0.0834      0.023      3.577      0.000       0.038       0.129
    ==============================================================================
    Omnibus:                    57694.520   Durbin-Watson:                   2.014
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22253956.558
    Skew:                           8.790   Prob(JB):                         0.00
    Kurtosis:                     117.496   Cond. No.                     7.65e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 4.01e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.77
    Average Error for grid_based model: 0.06
    [15:37:30] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.204
    Model:                                OLS   Adj. R-squared:                  0.204
    Method:                     Least Squares   F-statistic:                     601.0
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:32   Log-Likelihood:            -1.1653e+05
    No. Observations:                   39803   AIC:                         2.331e+05
    Df Residuals:                       39785   BIC:                         2.333e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8293      0.023     36.490      0.000       0.785       0.874
    x1             2.8487      0.034     82.878      0.000       2.781       2.916
    x2             0.9802      0.035     28.393      0.000       0.913       1.048
    x3             0.0229      0.082      0.278      0.781      -0.139       0.184
    x4             0.5859      0.060      9.840      0.000       0.469       0.703
    x5            -0.4999      0.059     -8.422      0.000      -0.616      -0.384
    x6            -0.0837      0.041     -2.053      0.040      -0.164      -0.004
    x7            -1.1941      0.457     -2.614      0.009      -2.090      -0.299
    x8            -0.2456      0.209     -1.175      0.240      -0.655       0.164
    x9             1.1392      0.440      2.592      0.010       0.278       2.001
    x10            0.2771      0.184      1.502      0.133      -0.085       0.639
    x11           -0.1032      0.047     -2.215      0.027      -0.195      -0.012
    x12            0.0934      0.029      3.211      0.001       0.036       0.150
    x13            0.0135      0.041      0.330      0.741      -0.067       0.094
    x14        -2.595e+12   3.52e+12     -0.737      0.461   -9.49e+12     4.3e+12
    x15        -4.453e+12   6.04e+12     -0.737      0.461   -1.63e+13    7.38e+12
    x16        -4.428e+12   6.01e+12     -0.737      0.461   -1.62e+13    7.34e+12
    x17           -0.0374      0.081     -0.461      0.645      -0.197       0.122
    x18            0.1026      0.026      3.970      0.000       0.052       0.153
    ==============================================================================
    Omnibus:                    57814.246   Durbin-Watson:                   2.008
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22468768.056
    Skew:                           8.824   Prob(JB):                         0.00
    Kurtosis:                     118.050   Cond. No.                     9.86e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 2.41e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.89
    Average Error for grid_based model: 0.07
    [15:37:33] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.203
    Model:                                OLS   Adj. R-squared:                  0.203
    Method:                     Least Squares   F-statistic:                     597.7
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:35   Log-Likelihood:            -1.1606e+05
    No. Observations:                   39803   AIC:                         2.322e+05
    Df Residuals:                       39785   BIC:                         2.323e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8229      0.022     36.736      0.000       0.779       0.867
    x1             2.7920      0.034     82.461      0.000       2.726       2.858
    x2             0.9513      0.034     27.965      0.000       0.885       1.018
    x3             0.0422      0.085      0.497      0.619      -0.124       0.209
    x4             0.5647      0.059      9.650      0.000       0.450       0.679
    x5            -0.4785      0.058     -8.197      0.000      -0.593      -0.364
    x6            -0.1044      0.037     -2.794      0.005      -0.178      -0.031
    x7            -1.3288      0.447     -2.974      0.003      -2.205      -0.453
    x8            -0.2598      0.206     -1.260      0.208      -0.664       0.144
    x9             1.2787      0.430      2.971      0.003       0.435       2.122
    x10            0.3090      0.182      1.699      0.089      -0.047       0.665
    x11           -0.1281      0.045     -2.824      0.005      -0.217      -0.039
    x12            0.1448      0.029      5.054      0.000       0.089       0.201
    x13           -0.0658      0.040     -1.656      0.098      -0.144       0.012
    x14           -0.0250      0.032     -0.774      0.439      -0.088       0.038
    x15           -0.0286      0.025     -1.148      0.251      -0.077       0.020
    x16            0.0434      0.020      2.146      0.032       0.004       0.083
    x17           -0.0593      0.086     -0.689      0.491      -0.228       0.109
    x18            0.0740      0.022      3.370      0.001       0.031       0.117
    ==============================================================================
    Omnibus:                    57506.611   Durbin-Watson:                   2.013
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21617154.671
    Skew:                           8.744   Prob(JB):                         0.00
    Kurtosis:                     115.821   Cond. No.                     1.43e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.15e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.82
    Average Error for grid_based model: 0.05
    [15:37:35] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.205
    Model:                                OLS   Adj. R-squared:                  0.204
    Method:                     Least Squares   F-statistic:                     601.9
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:37   Log-Likelihood:            -1.1647e+05
    No. Observations:                   39803   AIC:                         2.330e+05
    Df Residuals:                       39785   BIC:                         2.331e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8302      0.023     36.687      0.000       0.786       0.875
    x1             2.8455      0.034     83.293      0.000       2.779       2.912
    x2             0.9642      0.034     28.111      0.000       0.897       1.031
    x3             0.0308      0.081      0.382      0.702      -0.127       0.189
    x4             0.5622      0.059      9.525      0.000       0.447       0.678
    x5            -0.5051      0.059     -8.585      0.000      -0.620      -0.390
    x6            -0.0984      0.038     -2.599      0.009      -0.173      -0.024
    x7            -0.7401      0.451     -1.640      0.101      -1.625       0.145
    x8            -0.2843      0.208     -1.370      0.171      -0.691       0.123
    x9             0.7391      0.435      1.699      0.089      -0.114       1.592
    x10            0.3053      0.183      1.669      0.095      -0.053       0.664
    x11           -0.1447      0.046     -3.144      0.002      -0.235      -0.055
    x12            0.1155      0.029      3.998      0.000       0.059       0.172
    x13            0.0025      0.040      0.062      0.950      -0.077       0.082
    x14           -0.0193      0.033     -0.592      0.554      -0.083       0.045
    x15           -0.0024      0.025     -0.095      0.924      -0.052       0.047
    x16            0.0137      0.021      0.668      0.504      -0.027       0.054
    x17           -0.0452      0.080     -0.569      0.570      -0.201       0.111
    x18            0.0919      0.023      4.010      0.000       0.047       0.137
    ==============================================================================
    Omnibus:                    57598.359   Durbin-Watson:                   2.004
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21774302.274
    Skew:                           8.770   Prob(JB):                         0.00
    Kurtosis:                     116.233   Cond. No.                     1.10e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.91e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.65
    Average Error for grid_based model: 0.02
    [15:37:37] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.201
    Model:                                OLS   Adj. R-squared:                  0.201
    Method:                     Least Squares   F-statistic:                     590.0
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:39   Log-Likelihood:            -1.1654e+05
    No. Observations:                   39803   AIC:                         2.331e+05
    Df Residuals:                       39785   BIC:                         2.333e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8293      0.023     36.572      0.000       0.785       0.874
    x1             2.8220      0.034     82.079      0.000       2.755       2.889
    x2             0.9585      0.034     27.808      0.000       0.891       1.026
    x3             0.0614      0.085      0.720      0.472      -0.106       0.229
    x4             0.5569      0.059      9.415      0.000       0.441       0.673
    x5            -0.4730      0.059     -8.000      0.000      -0.589      -0.357
    x6            -0.1080      0.038     -2.845      0.004      -0.182      -0.034
    x7            -1.2888      0.454     -2.837      0.005      -2.179      -0.398
    x8            -0.1673      0.210     -0.797      0.425      -0.579       0.244
    x9             1.2251      0.438      2.800      0.005       0.368       2.083
    x10            0.2346      0.185      1.267      0.205      -0.128       0.598
    x11           -0.1370      0.047     -2.945      0.003      -0.228      -0.046
    x12            0.0919      0.029      3.168      0.002       0.035       0.149
    x13           -0.0215      0.040     -0.532      0.595      -0.100       0.058
    x14           -0.0356      0.033     -1.086      0.277      -0.100       0.029
    x15           -0.0146      0.025     -0.578      0.563      -0.064       0.035
    x16            0.0356      0.021      1.732      0.083      -0.005       0.076
    x17           -0.0643      0.084     -0.765      0.444      -0.229       0.100
    x18            0.0720      0.022      3.238      0.001       0.028       0.116
    ==============================================================================
    Omnibus:                    58333.215   Durbin-Watson:                   2.002
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23967727.520
    Skew:                           8.961   Prob(JB):                         0.00
    Kurtosis:                     121.872   Cond. No.                     1.15e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.78e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.70
    Average Error for grid_based model: 0.08
    [15:37:39] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.207
    Model:                                OLS   Adj. R-squared:                  0.207
    Method:                     Least Squares   F-statistic:                     611.7
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:41   Log-Likelihood:            -1.1662e+05
    No. Observations:                   39803   AIC:                         2.333e+05
    Df Residuals:                       39785   BIC:                         2.334e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8394      0.025     32.922      0.000       0.789       0.889
    x1             2.8746      0.034     83.623      0.000       2.807       2.942
    x2             0.9833      0.035     28.466      0.000       0.916       1.051
    x3             0.0170      0.084      0.203      0.839      -0.147       0.181
    x4             0.6258      0.059     10.540      0.000       0.509       0.742
    x5            -0.5528      0.059     -9.311      0.000      -0.669      -0.436
    x6            -0.0993      0.038     -2.584      0.010      -0.175      -0.024
    x7            -1.5348      0.455     -3.371      0.001      -2.427      -0.642
    x8            -0.1352      0.214     -0.632      0.528      -0.555       0.284
    x9             1.4568      0.437      3.332      0.001       0.600       2.314
    x10            0.2203      0.188      1.170      0.242      -0.149       0.589
    x11           -0.1158      0.053     -2.199      0.028      -0.219      -0.013
    x12            0.1335      0.029      4.572      0.000       0.076       0.191
    x13           -0.0420      0.048     -0.870      0.384      -0.137       0.053
    x14        -1.102e+12   2.53e+12     -0.436      0.663   -6.06e+12    3.86e+12
    x15         -1.89e+12   4.34e+12     -0.436      0.663   -1.04e+13    6.62e+12
    x16         -1.88e+12   4.32e+12     -0.436      0.663   -1.03e+13    6.58e+12
    x17           -0.0425      0.083     -0.515      0.607      -0.204       0.119
    x18            0.0999      0.025      4.035      0.000       0.051       0.148
    ==============================================================================
    Omnibus:                    57564.554   Durbin-Watson:                   1.993
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21851615.621
    Skew:                           8.757   Prob(JB):                         0.00
    Kurtosis:                     116.442   Cond. No.                     7.07e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 4.68e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.68
    Average Error for grid_based model: 0.03
    [15:37:41] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.208
    Model:                                OLS   Adj. R-squared:                  0.208
    Method:                     Least Squares   F-statistic:                     616.0
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:43   Log-Likelihood:            -1.1625e+05
    No. Observations:                   39803   AIC:                         2.325e+05
    Df Residuals:                       39785   BIC:                         2.327e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8311      0.023     36.927      0.000       0.787       0.875
    x1             2.8791      0.034     84.585      0.000       2.812       2.946
    x2             0.9802      0.034     28.609      0.000       0.913       1.047
    x3             0.0120      0.081      0.147      0.883      -0.148       0.172
    x4             0.5266      0.059      8.984      0.000       0.412       0.642
    x5            -0.4986      0.059     -8.488      0.000      -0.614      -0.383
    x6            -0.1138      0.037     -3.042      0.002      -0.187      -0.040
    x7            -1.1110      0.449     -2.476      0.013      -1.991      -0.232
    x8            -0.2514      0.207     -1.212      0.226      -0.658       0.155
    x9             1.0361      0.432      2.397      0.017       0.189       1.883
    x10            0.3145      0.183      1.720      0.085      -0.044       0.673
    x11           -0.1094      0.046     -2.400      0.016      -0.199      -0.020
    x12            0.1099      0.029      3.828      0.000       0.054       0.166
    x13           -0.0302      0.040     -0.754      0.451      -0.109       0.048
    x14           -0.0249      0.032     -0.771      0.441      -0.088       0.038
    x15           -0.0206      0.025     -0.824      0.410      -0.070       0.028
    x16            0.0354      0.020      1.737      0.082      -0.005       0.075
    x17           -0.0336      0.080     -0.419      0.675      -0.191       0.124
    x18            0.1107      0.025      4.481      0.000       0.062       0.159
    ==============================================================================
    Omnibus:                    57795.556   Durbin-Watson:                   2.007
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22693483.293
    Skew:                           8.813   Prob(JB):                         0.00
    Kurtosis:                     118.641   Cond. No.                     1.23e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.55e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.94
    Average Error for grid_based model: -0.01
    [15:37:43] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     592.2
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:46   Log-Likelihood:            -1.1653e+05
    No. Observations:                   39803   AIC:                         2.331e+05
    Df Residuals:                       39785   BIC:                         2.332e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8278      0.023     36.522      0.000       0.783       0.872
    x1             2.8186      0.034     82.398      0.000       2.752       2.886
    x2             0.9592      0.034     27.898      0.000       0.892       1.027
    x3             0.0440      0.083      0.530      0.596      -0.119       0.207
    x4             0.5385      0.059      9.115      0.000       0.423       0.654
    x5            -0.4618      0.059     -7.829      0.000      -0.577      -0.346
    x6            -0.1065      0.038     -2.814      0.005      -0.181      -0.032
    x7            -1.3377      0.453     -2.956      0.003      -2.225      -0.451
    x8            -0.2562      0.208     -1.229      0.219      -0.665       0.152
    x9             1.2677      0.436      2.909      0.004       0.413       2.122
    x10            0.3159      0.184      1.720      0.085      -0.044       0.676
    x11           -0.1214      0.046     -2.622      0.009      -0.212      -0.031
    x12            0.0927      0.029      3.213      0.001       0.036       0.149
    x13           -0.0163      0.040     -0.403      0.687      -0.095       0.063
    x14           -0.0440      0.033     -1.346      0.178      -0.108       0.020
    x15           -0.0175      0.025     -0.693      0.488      -0.067       0.032
    x16            0.0434      0.021      2.115      0.034       0.003       0.084
    x17           -0.0530      0.081     -0.652      0.514      -0.212       0.106
    x18            0.0766      0.023      3.372      0.001       0.032       0.121
    ==============================================================================
    Omnibus:                    58245.752   Durbin-Watson:                   1.987
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23646900.535
    Skew:                           8.939   Prob(JB):                         0.00
    Kurtosis:                     121.062   Cond. No.                     1.35e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.29e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.62
    Average Error for grid_based model: 0.03
    [15:37:46] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.207
    Model:                                OLS   Adj. R-squared:                  0.207
    Method:                     Least Squares   F-statistic:                     612.8
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:48   Log-Likelihood:            -1.1627e+05
    No. Observations:                   39803   AIC:                         2.326e+05
    Df Residuals:                       39785   BIC:                         2.327e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8353      0.023     37.091      0.000       0.791       0.879
    x1             2.8699      0.034     84.263      0.000       2.803       2.937
    x2             0.9760      0.034     28.550      0.000       0.909       1.043
    x3             0.0532      0.083      0.638      0.523      -0.110       0.217
    x4             0.5078      0.059      8.584      0.000       0.392       0.624
    x5            -0.4543      0.059     -7.697      0.000      -0.570      -0.339
    x6            -0.0825      0.038     -2.189      0.029      -0.156      -0.009
    x7            -0.9260      0.451     -2.054      0.040      -1.810      -0.042
    x8            -0.1715      0.208     -0.825      0.410      -0.579       0.236
    x9             0.8376      0.434      1.928      0.054      -0.014       1.689
    x10            0.2421      0.184      1.318      0.187      -0.118       0.602
    x11           -0.1209      0.046     -2.646      0.008      -0.210      -0.031
    x12            0.0975      0.029      3.398      0.001       0.041       0.154
    x13           -0.0396      0.040     -0.987      0.323      -0.118       0.039
    x14           -0.0600      0.032     -1.852      0.064      -0.124       0.003
    x15           -0.0163      0.025     -0.651      0.515      -0.066       0.033
    x16            0.0516      0.020      2.529      0.011       0.012       0.092
    x17           -0.0540      0.083     -0.648      0.517      -0.217       0.109
    x18            0.0920      0.023      4.039      0.000       0.047       0.137
    ==============================================================================
    Omnibus:                    57640.257   Durbin-Watson:                   1.964
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22198883.459
    Skew:                           8.774   Prob(JB):                         0.00
    Kurtosis:                     117.356   Cond. No.                     1.05e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 2.14e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.89
    Average Error for grid_based model: 0.02
    [15:37:48] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.203
    Model:                                OLS   Adj. R-squared:                  0.203
    Method:                     Least Squares   F-statistic:                     596.6
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:50   Log-Likelihood:            -1.1659e+05
    No. Observations:                   39803   AIC:                         2.332e+05
    Df Residuals:                       39785   BIC:                         2.334e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8311      0.023     36.612      0.000       0.787       0.876
    x1             2.8380      0.034     82.724      0.000       2.771       2.905
    x2             0.9633      0.034     27.947      0.000       0.896       1.031
    x3             0.0586      0.085      0.687      0.492      -0.109       0.226
    x4             0.5777      0.059      9.768      0.000       0.462       0.694
    x5            -0.5179      0.059     -8.760      0.000      -0.634      -0.402
    x6            -0.0986      0.038     -2.591      0.010      -0.173      -0.024
    x7            -1.0433      0.456     -2.287      0.022      -1.937      -0.149
    x8            -0.2109      0.210     -1.006      0.314      -0.622       0.200
    x9             0.9706      0.440      2.208      0.027       0.109       1.832
    x10            0.2645      0.185      1.430      0.153      -0.098       0.627
    x11           -0.1045      0.046     -2.253      0.024      -0.195      -0.014
    x12            0.1034      0.029      3.578      0.000       0.047       0.160
    x13            0.0103      0.040      0.255      0.799      -0.069       0.089
    x14           -0.0385      0.033     -1.178      0.239      -0.102       0.026
    x15            0.0014      0.025      0.054      0.957      -0.048       0.051
    x16            0.0212      0.021      1.030      0.303      -0.019       0.062
    x17           -0.0713      0.085     -0.837      0.403      -0.238       0.096
    x18            0.0986      0.025      4.000      0.000       0.050       0.147
    ==============================================================================
    Omnibus:                    58307.193   Durbin-Watson:                   2.005
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23930379.782
    Skew:                           8.953   Prob(JB):                         0.00
    Kurtosis:                     121.780   Cond. No.                     1.14e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.81e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.56
    Average Error for grid_based model: 0.05
    [15:37:50] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.205
    Model:                                OLS   Adj. R-squared:                  0.205
    Method:                     Least Squares   F-statistic:                     603.0
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:52   Log-Likelihood:            -1.1661e+05
    No. Observations:                   39803   AIC:                         2.333e+05
    Df Residuals:                       39785   BIC:                         2.334e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8387      0.023     36.652      0.000       0.794       0.884
    x1             2.8438      0.034     82.847      0.000       2.776       2.911
    x2             0.9713      0.035     28.096      0.000       0.904       1.039
    x3             0.0347      0.087      0.400      0.689      -0.136       0.205
    x4             0.5977      0.059     10.114      0.000       0.482       0.714
    x5            -0.5120      0.059     -8.668      0.000      -0.628      -0.396
    x6            -0.1110      0.041     -2.675      0.007      -0.192      -0.030
    x7            -1.1501      0.453     -2.538      0.011      -2.038      -0.262
    x8            -0.2844      0.211     -1.351      0.177      -0.697       0.128
    x9             1.1017      0.437      2.523      0.012       0.246       1.958
    x10            0.3363      0.186      1.812      0.070      -0.027       0.700
    x11           -0.1371      0.047     -2.929      0.003      -0.229      -0.045
    x12            0.1343      0.032      4.182      0.000       0.071       0.197
    x13           -0.0267      0.049     -0.545      0.586      -0.123       0.069
    x14         1.252e+12   2.81e+12      0.445      0.656   -4.26e+12    6.77e+12
    x15         2.147e+12   4.83e+12      0.445      0.656   -7.31e+12    1.16e+13
    x16         2.135e+12    4.8e+12      0.445      0.656   -7.27e+12    1.15e+13
    x17           -0.0416      0.086     -0.484      0.628      -0.210       0.127
    x18            0.0861      0.023      3.690      0.000       0.040       0.132
    ==============================================================================
    Omnibus:                    57721.095   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22205488.571
    Skew:                           8.800   Prob(JB):                         0.00
    Kurtosis:                     117.366   Cond. No.                     7.87e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 3.79e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.03
    [15:37:52] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.204
    Model:                                OLS   Adj. R-squared:                  0.204
    Method:                     Least Squares   F-statistic:                     599.9
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:54   Log-Likelihood:            -1.1683e+05
    No. Observations:                   39803   AIC:                         2.337e+05
    Df Residuals:                       39785   BIC:                         2.338e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8260      0.029     28.328      0.000       0.769       0.883
    x1             2.8671      0.035     82.740      0.000       2.799       2.935
    x2             0.9772      0.035     28.149      0.000       0.909       1.045
    x3             0.0238      0.085      0.282      0.778      -0.142       0.190
    x4             0.6345      0.059     10.665      0.000       0.518       0.751
    x5            -0.5734      0.059     -9.664      0.000      -0.690      -0.457
    x6            -0.0917      0.041     -2.264      0.024      -0.171      -0.012
    x7            -1.1000      0.457     -2.409      0.016      -1.995      -0.205
    x8            -0.2704      0.210     -1.285      0.199      -0.683       0.142
    x9             1.0599      0.440      2.408      0.016       0.197       1.923
    x10            0.3179      0.185      1.714      0.087      -0.046       0.681
    x11           -0.1387      0.047     -2.976      0.003      -0.230      -0.047
    x12            0.1315      0.029      4.487      0.000       0.074       0.189
    x13           -0.0115      0.041     -0.280      0.779      -0.092       0.069
    x14        -1.354e+12   2.31e+12     -0.586      0.558   -5.88e+12    3.17e+12
    x15        -2.323e+12   3.96e+12     -0.586      0.558   -1.01e+13    5.44e+12
    x16        -2.311e+12   3.94e+12     -0.586      0.558      -1e+13    5.41e+12
    x17           -0.0323      0.083     -0.388      0.698      -0.195       0.131
    x18            0.0836      0.022      3.745      0.000       0.040       0.127
    ==============================================================================
    Omnibus:                    57944.247   Durbin-Watson:                   1.988
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22792477.021
    Skew:                           8.859   Prob(JB):                         0.00
    Kurtosis:                     118.885   Cond. No.                     6.42e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 5.68e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.64
    Average Error for grid_based model: 0.02
    [15:37:54] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.203
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     595.1
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:57   Log-Likelihood:            -1.1637e+05
    No. Observations:                   39803   AIC:                         2.328e+05
    Df Residuals:                       39785   BIC:                         2.329e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8383      0.029     29.359      0.000       0.782       0.894
    x1             2.8136      0.034     82.546      0.000       2.747       2.880
    x2             0.9560      0.034     27.846      0.000       0.889       1.023
    x3             0.0282      0.082      0.344      0.731      -0.132       0.189
    x4             0.5745      0.059      9.775      0.000       0.459       0.690
    x5            -0.5030      0.059     -8.571      0.000      -0.618      -0.388
    x6            -0.1000      0.038     -2.644      0.008      -0.174      -0.026
    x7            -1.1647      0.452     -2.579      0.010      -2.050      -0.279
    x8            -0.2289      0.206     -1.109      0.267      -0.633       0.176
    x9             1.1521      0.435      2.648      0.008       0.299       2.005
    x10            0.2401      0.182      1.320      0.187      -0.116       0.597
    x11           -0.1463      0.046     -3.168      0.002      -0.237      -0.056
    x12            0.1167      0.032      3.662      0.000       0.054       0.179
    x13           -0.0112      0.042     -0.265      0.791      -0.094       0.072
    x14         2.437e+12   2.94e+12      0.828      0.408   -3.33e+12    8.21e+12
    x15         4.181e+12   5.05e+12      0.828      0.408   -5.72e+12    1.41e+13
    x16         4.158e+12   5.02e+12      0.828      0.408   -5.69e+12     1.4e+13
    x17           -0.0463      0.081     -0.575      0.566      -0.204       0.112
    x18            0.0762      0.022      3.477      0.001       0.033       0.119
    ==============================================================================
    Omnibus:                    58237.702   Durbin-Watson:                   2.004
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23520970.950
    Skew:                           8.939   Prob(JB):                         0.00
    Kurtosis:                     120.740   Cond. No.                     8.32e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 3.41e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.86
    Average Error for grid_based model: -0.01
    [15:37:57] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     592.0
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:37:59   Log-Likelihood:            -1.1651e+05
    No. Observations:                   39803   AIC:                         2.330e+05
    Df Residuals:                       39785   BIC:                         2.332e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8239      0.023     36.370      0.000       0.780       0.868
    x1             2.8208      0.034     82.405      0.000       2.754       2.888
    x2             0.9595      0.034     27.890      0.000       0.892       1.027
    x3             0.0075      0.089      0.084      0.933      -0.166       0.181
    x4             0.5111      0.059      8.727      0.000       0.396       0.626
    x5            -0.4489      0.059     -7.672      0.000      -0.564      -0.334
    x6            -0.1034      0.038     -2.735      0.006      -0.177      -0.029
    x7            -1.4357      0.452     -3.173      0.002      -2.322      -0.549
    x8            -0.1833      0.207     -0.884      0.377      -0.590       0.223
    x9             1.3893      0.436      3.185      0.001       0.534       2.244
    x10            0.2265      0.183      1.239      0.215      -0.132       0.585
    x11           -0.1329      0.046     -2.918      0.004      -0.222      -0.044
    x12            0.1139      0.029      3.915      0.000       0.057       0.171
    x13           -0.0239      0.040     -0.593      0.553      -0.103       0.055
    x14           -0.0486      0.033     -1.488      0.137      -0.113       0.015
    x15           -0.0092      0.025     -0.366      0.714      -0.059       0.040
    x16            0.0378      0.020      1.842      0.065      -0.002       0.078
    x17           -0.0151      0.091     -0.166      0.868      -0.193       0.163
    x18            0.0801      0.022      3.612      0.000       0.037       0.124
    ==============================================================================
    Omnibus:                    58284.167   Durbin-Watson:                   2.014
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23795651.342
    Skew:                           8.949   Prob(JB):                         0.00
    Kurtosis:                     121.439   Cond. No.                     1.19e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.64e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.52
    Average Error for grid_based model: 0.04
    [15:37:59] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.206
    Model:                                OLS   Adj. R-squared:                  0.206
    Method:                     Least Squares   F-statistic:                     608.7
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:38:01   Log-Likelihood:            -1.1602e+05
    No. Observations:                   39803   AIC:                         2.321e+05
    Df Residuals:                       39785   BIC:                         2.322e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8231      0.022     36.779      0.000       0.779       0.867
    x1             2.8456      0.034     83.903      0.000       2.779       2.912
    x2             0.9762      0.034     28.620      0.000       0.909       1.043
    x3             0.0104      0.084      0.124      0.901      -0.154       0.174
    x4             0.5543      0.059      9.445      0.000       0.439       0.669
    x5            -0.5020      0.058     -8.591      0.000      -0.617      -0.387
    x6            -0.0792      0.038     -2.111      0.035      -0.153      -0.006
    x7            -0.9853      0.447     -2.203      0.028      -1.862      -0.109
    x8            -0.1237      0.206     -0.600      0.548      -0.528       0.280
    x9             0.9373      0.431      2.176      0.030       0.093       1.782
    x10            0.1705      0.182      0.938      0.348      -0.186       0.527
    x11           -0.1365      0.045     -3.000      0.003      -0.226      -0.047
    x12            0.1221      0.029      4.274      0.000       0.066       0.178
    x13           -0.0541      0.040     -1.360      0.174      -0.132       0.024
    x14           -0.0534      0.032     -1.658      0.097      -0.116       0.010
    x15           -0.0187      0.025     -0.747      0.455      -0.068       0.030
    x16            0.0500      0.020      2.465      0.014       0.010       0.090
    x17           -0.0206      0.084     -0.244      0.807      -0.186       0.145
    x18            0.0974      0.022      4.483      0.000       0.055       0.140
    ==============================================================================
    Omnibus:                    56981.123   Durbin-Watson:                   2.010
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         20304362.725
    Skew:                           8.606   Prob(JB):                         0.00
    Kurtosis:                     112.301   Cond. No.                     1.28e+15
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 1.43e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 3.13
    Average Error for grid_based model: -0.00
    [15:38:01] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.209
    Model:                                OLS   Adj. R-squared:                  0.208
    Method:                     Least Squares   F-statistic:                     617.6
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:38:03   Log-Likelihood:            -1.1560e+05
    No. Observations:                   39803   AIC:                         2.312e+05
    Df Residuals:                       39785   BIC:                         2.314e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8324      0.023     36.690      0.000       0.788       0.877
    x1             2.8249      0.033     84.395      0.000       2.759       2.891
    x2             0.9605      0.034     28.532      0.000       0.895       1.026
    x3             0.0547      0.081      0.677      0.498      -0.104       0.213
    x4             0.5114      0.058      8.885      0.000       0.399       0.624
    x5            -0.4456      0.058     -7.729      0.000      -0.559      -0.333
    x6            -0.1480      0.044     -3.388      0.001      -0.234      -0.062
    x7            -1.1266      0.442     -2.549      0.011      -1.993      -0.260
    x8            -0.3645      0.206     -1.767      0.077      -0.769       0.040
    x9             1.0706      0.425      2.518      0.012       0.237       1.904
    x10            0.4120      0.182      2.262      0.024       0.055       0.769
    x11           -0.1309      0.045     -2.886      0.004      -0.220      -0.042
    x12            0.0983      0.029      3.443      0.001       0.042       0.154
    x13           -0.0089      0.040     -0.222      0.824      -0.088       0.070
    x14        -4.572e+12   2.74e+12     -1.668      0.095   -9.94e+12    7.99e+11
    x15        -7.844e+12    4.7e+12     -1.668      0.095   -1.71e+13    1.37e+12
    x16        -7.801e+12   4.68e+12     -1.668      0.095    -1.7e+13    1.36e+12
    x17           -0.0509      0.079     -0.642      0.521      -0.206       0.104
    x18            0.0763      0.022      3.548      0.000       0.034       0.118
    ==============================================================================
    Omnibus:                    56684.137   Durbin-Watson:                   1.992
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         19588016.379
    Skew:                           8.529   Prob(JB):                         0.00
    Kurtosis:                     110.331   Cond. No.                     7.85e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 3.79e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 3.09
    Average Error for grid_based model: 0.01
    [15:38:03] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.201
    Model:                                OLS   Adj. R-squared:                  0.201
    Method:                     Least Squares   F-statistic:                     589.4
    Date:                    Tue, 13 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:38:05   Log-Likelihood:            -1.1688e+05
    No. Observations:                   39803   AIC:                         2.338e+05
    Df Residuals:                       39785   BIC:                         2.339e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8379      0.029     29.085      0.000       0.781       0.894
    x1             2.8403      0.035     82.145      0.000       2.773       2.908
    x2             0.9446      0.035     27.205      0.000       0.877       1.013
    x3             0.0338      0.090      0.377      0.706      -0.142       0.210
    x4             0.5073      0.060      8.508      0.000       0.390       0.624
    x5            -0.4642      0.060     -7.791      0.000      -0.581      -0.347
    x6            -0.1530      0.041     -3.764      0.000      -0.233      -0.073
    x7            -1.4102      0.459     -3.070      0.002      -2.310      -0.510
    x8            -0.1599      0.209     -0.764      0.445      -0.570       0.250
    x9             1.3439      0.442      3.042      0.002       0.478       2.210
    x10            0.2271      0.184      1.233      0.218      -0.134       0.588
    x11           -0.1511      0.049     -3.101      0.002      -0.247      -0.056
    x12            0.0798      0.029      2.751      0.006       0.023       0.137
    x13           -0.0207      0.042     -0.497      0.619      -0.102       0.061
    x14        -8.437e+11   2.31e+12     -0.365      0.715   -5.38e+12    3.69e+12
    x15        -1.448e+12   3.97e+12     -0.365      0.715   -9.23e+12    6.33e+12
    x16         -1.44e+12   3.95e+12     -0.365      0.715   -9.18e+12     6.3e+12
    x17           -0.0322      0.090     -0.358      0.721      -0.209       0.144
    x18            0.0812      0.022      3.645      0.000       0.038       0.125
    ==============================================================================
    Omnibus:                    58244.199   Durbin-Watson:                   2.009
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23538200.817
    Skew:                           8.941   Prob(JB):                         0.00
    Kurtosis:                     120.784   Cond. No.                     6.42e+14
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 5.67e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    RMSE for grid_based model: 2.65
    Average Error for grid_based model: 0.02



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
    
    mean_RMSE: 2.74
    stdev_RMSE: 0.20
    mean_average_error: 0.03
    stdev_average_error: 0.03



```python
for bin_num in range(1, 6):

    print(f"\n RMSE and Average Error per bin {bin_num}\n")
    rmse_ave_mean(RMSE[bin_num], AVE[bin_num])
```

    
     RMSE and Average Error per bin 1
    
    mean_RMSE: 0.60
    stdev_RMSE: 0.07
    mean_average_error: 0.13
    stdev_average_error: 0.01
    
     RMSE and Average Error per bin 2
    
    mean_RMSE: 1.43
    stdev_RMSE: 0.13
    mean_average_error: 0.55
    stdev_average_error: 0.04
    
     RMSE and Average Error per bin 3
    
    mean_RMSE: 4.65
    stdev_RMSE: 0.44
    mean_average_error: 0.06
    stdev_average_error: 0.20
    
     RMSE and Average Error per bin 4
    
    mean_RMSE: 14.06
    stdev_RMSE: 0.84
    mean_average_error: -6.84
    stdev_average_error: 1.17
    
     RMSE and Average Error per bin 5
    
    mean_RMSE: 32.71
    stdev_RMSE: 5.18
    mean_average_error: -24.74
    stdev_average_error: 4.84



```python

```

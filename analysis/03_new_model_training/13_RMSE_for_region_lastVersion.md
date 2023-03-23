# RMSE estimation for region(adm1)

We used a weight and adm3_area files to join weight and adm_1(region) and adm_3(municipality) to our main dataset.
We prepared a dataframe that represents real and damaged value per region(ADM1). Then we train our model(XGBoost Reduced Overfitting) to this input data while we splitted five typhoons(randomly selected) as the test set and the rest of them as the train set.
The final goal is to estimate the difference between real and predicted damage value per region with respect to each typhoon, to check how the model performs for a wide area.


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

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics

from utils import get_training_dataset, weight_file
```

    /Users/mersedehkooshki/opt/anaconda3/envs/global-storm/lib/python3.8/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      from pandas import MultiIndex, Int64Index



```python
# Read csv file and import to df
df = get_training_dataset()

# Move target to be the last column for simplicity
df = df.reindex(
    columns=[col for col in df.columns if col != "percent_houses_damaged"]
    + ["percent_houses_damaged"]
)

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
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
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
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.990000</td>
      <td>197.339034</td>
      <td>0.000000</td>
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
      <td>0.00</td>
      <td>0.310000</td>
      <td>0.690000</td>
      <td>4970.477311</td>
      <td>0.000000</td>
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
      <td>0.00</td>
      <td>0.770000</td>
      <td>0.230000</td>
      <td>12408.594656</td>
      <td>0.000000</td>
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
      <td>18.012771</td>
      <td>36.304688</td>
      <td>21559.003490</td>
      <td>1</td>
      <td>0.08</td>
      <td>0.080000</td>
      <td>0.840000</td>
      <td>17619.701390</td>
      <td>0.000000</td>
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
      <td>3.532580</td>
      <td>...</td>
      <td>13.163042</td>
      <td>65.687266</td>
      <td>12591.742022</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.420000</td>
      <td>0.580000</td>
      <td>5623.069564</td>
      <td>0.000000</td>
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
      <td>4.444498</td>
      <td>...</td>
      <td>10.901755</td>
      <td>37.414996</td>
      <td>19740.596834</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.109091</td>
      <td>0.890909</td>
      <td>5912.671746</td>
      <td>0.015207</td>
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
      <td>5.816195</td>
      <td>...</td>
      <td>17.917650</td>
      <td>105.812452</td>
      <td>26363.303778</td>
      <td>1</td>
      <td>0.03</td>
      <td>0.250000</td>
      <td>0.720000</td>
      <td>11254.164413</td>
      <td>0.020806</td>
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
      <td>6.730992</td>
      <td>...</td>
      <td>17.010867</td>
      <td>89.696703</td>
      <td>9359.492382</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.070000</td>
      <td>0.930000</td>
      <td>3188.718115</td>
      <td>0.001050</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>141258 rows × 22 columns</p>
</div>




```python
# df.loc[df["typhoon_name"] == "GONI"]
```


```python
# Fill the missing values of RWI with mean value
df["rwi"].fillna(df["rwi"].mean(), inplace=True)
```


```python
# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```


```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df.reset_index(drop=True, inplace=True)
df = df.drop(columns=["typhoon_year"])
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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>...</th>
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
      <td>DURIAN</td>
      <td>8284</td>
      <td>12.460039</td>
      <td>275.018491</td>
      <td>0.670833</td>
      <td>0.313021</td>
      <td>0.479848</td>
      <td>-0.213039</td>
      <td>12.896581</td>
      <td>7.450346</td>
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
      <th>1</th>
      <td>DURIAN</td>
      <td>8286</td>
      <td>11.428974</td>
      <td>297.027578</td>
      <td>0.929167</td>
      <td>0.343229</td>
      <td>55.649739</td>
      <td>0.206000</td>
      <td>14.070741</td>
      <td>6.514647</td>
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
      <th>2</th>
      <td>DURIAN</td>
      <td>8450</td>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>0.716667</td>
      <td>0.424479</td>
      <td>8.157414</td>
      <td>-0.636000</td>
      <td>19.758682</td>
      <td>10.940700</td>
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
      <th>3</th>
      <td>DURIAN</td>
      <td>8451</td>
      <td>12.511864</td>
      <td>273.639330</td>
      <td>0.568750</td>
      <td>0.336979</td>
      <td>88.292015</td>
      <td>-0.227500</td>
      <td>11.499097</td>
      <td>6.901584</td>
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
      <th>4</th>
      <td>DURIAN</td>
      <td>8452</td>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>0.589583</td>
      <td>0.290625</td>
      <td>962.766739</td>
      <td>-0.299667</td>
      <td>13.866633</td>
      <td>6.528689</td>
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
<p>5 rows × 21 columns</p>
</div>




```python
# Define bins for data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
```


```python
# Check the bins' intervalls
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
# Use MinMaxScaler function for data standardization (it normalaize data in range of [0,1] and not negative values)

# Separate typhoon from other features
dfs = np.split(df, [2], axis=1)
dfa = np.split(dfs[1], [14], axis=1)
# print(dfs[0], dfs[1], dfa[0], dfa[1])

# Standardaize data
scaler = MinMaxScaler().fit(dfa[0])
X1 = scaler.transform(dfa[0])
Xnew = pd.DataFrame(X1)
Xnew_per_pred = pd.DataFrame(X1)
display(Xnew)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.168433</td>
      <td>0.879044</td>
      <td>0.010371</td>
      <td>0.009660</td>
      <td>2.334511e-07</td>
      <td>0.330964</td>
      <td>0.421621</td>
      <td>0.526082</td>
      <td>0.478800</td>
      <td>0.413882</td>
      <td>0.028344</td>
      <td>0.031312</td>
      <td>1.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.154462</td>
      <td>0.949392</td>
      <td>0.014378</td>
      <td>0.010594</td>
      <td>9.712553e-05</td>
      <td>0.504983</td>
      <td>0.460007</td>
      <td>0.460011</td>
      <td>0.440663</td>
      <td>0.304474</td>
      <td>0.043049</td>
      <td>0.360224</td>
      <td>1.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.176799</td>
      <td>0.839346</td>
      <td>0.011082</td>
      <td>0.013105</td>
      <td>1.371717e-05</td>
      <td>0.155316</td>
      <td>0.645959</td>
      <td>0.772542</td>
      <td>0.670175</td>
      <td>0.649623</td>
      <td>0.057690</td>
      <td>0.393828</td>
      <td>1.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.169135</td>
      <td>0.874636</td>
      <td>0.008788</td>
      <td>0.010400</td>
      <td>1.544535e-04</td>
      <td>0.324958</td>
      <td>0.375933</td>
      <td>0.487333</td>
      <td>0.383667</td>
      <td>0.380232</td>
      <td>0.036547</td>
      <td>0.317867</td>
      <td>1.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.161895</td>
      <td>0.909926</td>
      <td>0.009111</td>
      <td>0.008968</td>
      <td>1.690249e-03</td>
      <td>0.294989</td>
      <td>0.453334</td>
      <td>0.461002</td>
      <td>0.421247</td>
      <td>0.310462</td>
      <td>0.062176</td>
      <td>0.515864</td>
      <td>1.0</td>
      <td>0.07</td>
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
    </tr>
    <tr>
      <th>49749</th>
      <td>0.108159</td>
      <td>0.816770</td>
      <td>0.034958</td>
      <td>0.032650</td>
      <td>7.311641e-05</td>
      <td>0.111296</td>
      <td>0.327556</td>
      <td>0.563018</td>
      <td>0.331492</td>
      <td>0.433746</td>
      <td>0.026779</td>
      <td>0.011391</td>
      <td>1.0</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>49750</th>
      <td>0.102816</td>
      <td>0.852281</td>
      <td>0.033634</td>
      <td>0.028851</td>
      <td>2.854586e-03</td>
      <td>0.379331</td>
      <td>0.129550</td>
      <td>0.275946</td>
      <td>0.129081</td>
      <td>0.208269</td>
      <td>0.019943</td>
      <td>0.271370</td>
      <td>1.0</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>49751</th>
      <td>0.096754</td>
      <td>0.887792</td>
      <td>0.029724</td>
      <td>0.025744</td>
      <td>1.096340e-03</td>
      <td>0.498339</td>
      <td>0.129227</td>
      <td>0.261104</td>
      <td>0.138776</td>
      <td>0.204722</td>
      <td>0.016272</td>
      <td>0.087050</td>
      <td>1.0</td>
      <td>0.01</td>
    </tr>
    <tr>
      <th>49752</th>
      <td>0.092212</td>
      <td>0.923300</td>
      <td>0.058092</td>
      <td>0.035882</td>
      <td>3.178534e-05</td>
      <td>0.286545</td>
      <td>0.132543</td>
      <td>0.177055</td>
      <td>0.156689</td>
      <td>0.151533</td>
      <td>0.015221</td>
      <td>0.031742</td>
      <td>1.0</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>49753</th>
      <td>0.092815</td>
      <td>0.923297</td>
      <td>0.064748</td>
      <td>0.038826</td>
      <td>2.127484e-04</td>
      <td>0.315199</td>
      <td>0.179078</td>
      <td>0.324393</td>
      <td>0.188511</td>
      <td>0.254262</td>
      <td>0.025869</td>
      <td>0.057267</td>
      <td>1.0</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>49754 rows × 14 columns</p>
</div>



```python
dfa[1] = dfa[1].astype(float)
```


```python
Xnew = pd.concat([Xnew.reset_index(drop=True), dfa[1].reset_index(drop=True)], axis=1)
Xnew
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
      <th>13</th>
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
      <td>0.168433</td>
      <td>0.879044</td>
      <td>0.010371</td>
      <td>0.009660</td>
      <td>2.334511e-07</td>
      <td>0.330964</td>
      <td>0.421621</td>
      <td>0.526082</td>
      <td>0.478800</td>
      <td>0.413882</td>
      <td>0.028344</td>
      <td>0.031312</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.154462</td>
      <td>0.949392</td>
      <td>0.014378</td>
      <td>0.010594</td>
      <td>9.712553e-05</td>
      <td>0.504983</td>
      <td>0.460007</td>
      <td>0.460011</td>
      <td>0.440663</td>
      <td>0.304474</td>
      <td>0.043049</td>
      <td>0.360224</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.140000</td>
      <td>0.860000</td>
      <td>276.871504</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.176799</td>
      <td>0.839346</td>
      <td>0.011082</td>
      <td>0.013105</td>
      <td>1.371717e-05</td>
      <td>0.155316</td>
      <td>0.645959</td>
      <td>0.772542</td>
      <td>0.670175</td>
      <td>0.649623</td>
      <td>0.057690</td>
      <td>0.393828</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.110000</td>
      <td>0.890000</td>
      <td>448.539453</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.169135</td>
      <td>0.874636</td>
      <td>0.008788</td>
      <td>0.010400</td>
      <td>1.544535e-04</td>
      <td>0.324958</td>
      <td>0.375933</td>
      <td>0.487333</td>
      <td>0.383667</td>
      <td>0.380232</td>
      <td>0.036547</td>
      <td>0.317867</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.120000</td>
      <td>0.880000</td>
      <td>2101.708435</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.161895</td>
      <td>0.909926</td>
      <td>0.009111</td>
      <td>0.008968</td>
      <td>1.690249e-03</td>
      <td>0.294989</td>
      <td>0.453334</td>
      <td>0.461002</td>
      <td>0.421247</td>
      <td>0.310462</td>
      <td>0.062176</td>
      <td>0.515864</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.460000</td>
      <td>0.470000</td>
      <td>11632.726327</td>
      <td>0.000000</td>
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
      <th>49749</th>
      <td>0.108159</td>
      <td>0.816770</td>
      <td>0.034958</td>
      <td>0.032650</td>
      <td>7.311641e-05</td>
      <td>0.111296</td>
      <td>0.327556</td>
      <td>0.563018</td>
      <td>0.331492</td>
      <td>0.433746</td>
      <td>0.026779</td>
      <td>0.011391</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.990000</td>
      <td>330.215768</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49750</th>
      <td>0.102816</td>
      <td>0.852281</td>
      <td>0.033634</td>
      <td>0.028851</td>
      <td>2.854586e-03</td>
      <td>0.379331</td>
      <td>0.129550</td>
      <td>0.275946</td>
      <td>0.129081</td>
      <td>0.208269</td>
      <td>0.019943</td>
      <td>0.271370</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.190000</td>
      <td>0.760000</td>
      <td>5409.607943</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49751</th>
      <td>0.096754</td>
      <td>0.887792</td>
      <td>0.029724</td>
      <td>0.025744</td>
      <td>1.096340e-03</td>
      <td>0.498339</td>
      <td>0.129227</td>
      <td>0.261104</td>
      <td>0.138776</td>
      <td>0.204722</td>
      <td>0.016272</td>
      <td>0.087050</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.970000</td>
      <td>5378.401365</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49752</th>
      <td>0.092212</td>
      <td>0.923300</td>
      <td>0.058092</td>
      <td>0.035882</td>
      <td>3.178534e-05</td>
      <td>0.286545</td>
      <td>0.132543</td>
      <td>0.177055</td>
      <td>0.156689</td>
      <td>0.151533</td>
      <td>0.015221</td>
      <td>0.031742</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.027273</td>
      <td>0.972727</td>
      <td>914.677196</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49753</th>
      <td>0.092815</td>
      <td>0.923297</td>
      <td>0.064748</td>
      <td>0.038826</td>
      <td>2.127484e-04</td>
      <td>0.315199</td>
      <td>0.179078</td>
      <td>0.324393</td>
      <td>0.188511</td>
      <td>0.254262</td>
      <td>0.025869</td>
      <td>0.057267</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.980000</td>
      <td>1466.117288</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>49754 rows × 19 columns</p>
</div>




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
```


```python
# Add the features to the columns' headers after standardization
i = 0
for feature in features:
    Xnew = Xnew.rename(columns={i: feature})
    i += 1

Xnew = pd.concat([dfs[0].reset_index(drop=True), Xnew.reset_index(drop=True)], axis=1)
Xnew
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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>...</th>
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
      <td>DURIAN</td>
      <td>8284</td>
      <td>0.168433</td>
      <td>0.879044</td>
      <td>0.010371</td>
      <td>0.009660</td>
      <td>2.334511e-07</td>
      <td>0.330964</td>
      <td>0.421621</td>
      <td>0.526082</td>
      <td>...</td>
      <td>0.413882</td>
      <td>0.028344</td>
      <td>0.031312</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN</td>
      <td>8286</td>
      <td>0.154462</td>
      <td>0.949392</td>
      <td>0.014378</td>
      <td>0.010594</td>
      <td>9.712553e-05</td>
      <td>0.504983</td>
      <td>0.460007</td>
      <td>0.460011</td>
      <td>...</td>
      <td>0.304474</td>
      <td>0.043049</td>
      <td>0.360224</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.140000</td>
      <td>0.860000</td>
      <td>276.871504</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>8450</td>
      <td>0.176799</td>
      <td>0.839346</td>
      <td>0.011082</td>
      <td>0.013105</td>
      <td>1.371717e-05</td>
      <td>0.155316</td>
      <td>0.645959</td>
      <td>0.772542</td>
      <td>...</td>
      <td>0.649623</td>
      <td>0.057690</td>
      <td>0.393828</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.110000</td>
      <td>0.890000</td>
      <td>448.539453</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN</td>
      <td>8451</td>
      <td>0.169135</td>
      <td>0.874636</td>
      <td>0.008788</td>
      <td>0.010400</td>
      <td>1.544535e-04</td>
      <td>0.324958</td>
      <td>0.375933</td>
      <td>0.487333</td>
      <td>...</td>
      <td>0.380232</td>
      <td>0.036547</td>
      <td>0.317867</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.120000</td>
      <td>0.880000</td>
      <td>2101.708435</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN</td>
      <td>8452</td>
      <td>0.161895</td>
      <td>0.909926</td>
      <td>0.009111</td>
      <td>0.008968</td>
      <td>1.690249e-03</td>
      <td>0.294989</td>
      <td>0.453334</td>
      <td>0.461002</td>
      <td>...</td>
      <td>0.310462</td>
      <td>0.062176</td>
      <td>0.515864</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.460000</td>
      <td>0.470000</td>
      <td>11632.726327</td>
      <td>0.000000</td>
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
      <th>49749</th>
      <td>MOLAVE</td>
      <td>19306</td>
      <td>0.108159</td>
      <td>0.816770</td>
      <td>0.034958</td>
      <td>0.032650</td>
      <td>7.311641e-05</td>
      <td>0.111296</td>
      <td>0.327556</td>
      <td>0.563018</td>
      <td>...</td>
      <td>0.433746</td>
      <td>0.026779</td>
      <td>0.011391</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.990000</td>
      <td>330.215768</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49750</th>
      <td>MOLAVE</td>
      <td>19307</td>
      <td>0.102816</td>
      <td>0.852281</td>
      <td>0.033634</td>
      <td>0.028851</td>
      <td>2.854586e-03</td>
      <td>0.379331</td>
      <td>0.129550</td>
      <td>0.275946</td>
      <td>...</td>
      <td>0.208269</td>
      <td>0.019943</td>
      <td>0.271370</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.190000</td>
      <td>0.760000</td>
      <td>5409.607943</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49751</th>
      <td>MOLAVE</td>
      <td>19308</td>
      <td>0.096754</td>
      <td>0.887792</td>
      <td>0.029724</td>
      <td>0.025744</td>
      <td>1.096340e-03</td>
      <td>0.498339</td>
      <td>0.129227</td>
      <td>0.261104</td>
      <td>...</td>
      <td>0.204722</td>
      <td>0.016272</td>
      <td>0.087050</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.970000</td>
      <td>5378.401365</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49752</th>
      <td>MOLAVE</td>
      <td>19476</td>
      <td>0.092212</td>
      <td>0.923300</td>
      <td>0.058092</td>
      <td>0.035882</td>
      <td>3.178534e-05</td>
      <td>0.286545</td>
      <td>0.132543</td>
      <td>0.177055</td>
      <td>...</td>
      <td>0.151533</td>
      <td>0.015221</td>
      <td>0.031742</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.027273</td>
      <td>0.972727</td>
      <td>914.677196</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49753</th>
      <td>MOLAVE</td>
      <td>19643</td>
      <td>0.092815</td>
      <td>0.923297</td>
      <td>0.064748</td>
      <td>0.038826</td>
      <td>2.127484e-04</td>
      <td>0.315199</td>
      <td>0.179078</td>
      <td>0.324393</td>
      <td>...</td>
      <td>0.254262</td>
      <td>0.025869</td>
      <td>0.057267</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.980000</td>
      <td>1466.117288</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>49754 rows × 21 columns</p>
</div>




```python
df["typhoon_name"].unique()
```




    array(['DURIAN', 'FENGSHEN', 'KETSANA', 'CONSON', 'NESAT', 'BOPHA',
           'NARI', 'KROSA', 'HAIYAN', 'USAGI', 'UTOR', 'JANGMI', 'KALMAEGI',
           'RAMMASUN', 'HAGUPIT', 'FUNG-WONG', 'LINGLING', 'MUJIGAE', 'MELOR',
           'NOUL', 'GONI', 'LINFA', 'KOPPU', 'MEKKHALA', 'HAIMA', 'TOKAGE',
           'MERANTI', 'NOCK-TEN', 'SARIKA', 'MANGKHUT', 'YUTU', 'KAMMURI',
           'NAKRI', 'PHANFONE', 'SAUDEL', 'VAMCO', 'VONGFONG', 'MOLAVE'],
          dtype=object)




```python
# Define a test_list (including 5 typhoons) randomly were chosen
test_list_1 = ["FENGSHEN", "DURIAN", "NESAT", "VONGFONG", "MOLAVE"]

test_list_2 = ["YUTU", "KAMMURI", "SARIKA", "TOKAGE", "LINGLING"]

test_list_3 = ["SAUDEL", "MANGKHUT", "HAIMA", "BOPHA", "KETSANA"]

test_list_4 = ["GONI", "LINFA", "NOCK-TEN", "NOUL", "JANGMI"]

test_list_5 = ["NAKRI", "UTOR", "HAIYAN", "RAMMASUN", "CONSON"]

test_list_6 = ["PHANFONE", "VAMCO", "KOPPU", "FUNG-WONG", "HAGUPIT"]

test_list_7 = ["MEKKHALA", "NARI", "KROSA", "USAGI", "KALMAEGI"]
```


```python
# Extract the column of unique ids
grid_id = df["grid_point_id"]
```


```python
df_test = pd.DataFrame(
    Xnew,
    columns=[
        "typhoon_name",
        "grid_point_id",
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
    ],
)

df_test = Xnew[Xnew["typhoon_name"] == test_list_3[4]]
df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[3]])
df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[2]])
df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[1]])
df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[0]])

Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[4]], inplace=True)
Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[3]], inplace=True)
Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[2]], inplace=True)
Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[1]], inplace=True)
Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[0]], inplace=True)

display(df_test)
df_train = Xnew
display(df_train)
```

    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_32733/1046774399.py:29: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[3]])
    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_32733/1046774399.py:30: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[2]])
    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_32733/1046774399.py:31: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[1]])
    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_32733/1046774399.py:32: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[0]])



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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>...</th>
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
      <th>3804</th>
      <td>KETSANA</td>
      <td>9233</td>
      <td>0.228997</td>
      <td>0.250719</td>
      <td>0.044070</td>
      <td>0.044935</td>
      <td>0.000753</td>
      <td>0.462625</td>
      <td>0.069010</td>
      <td>0.090805</td>
      <td>...</td>
      <td>0.065630</td>
      <td>0.020180</td>
      <td>0.031687</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.92</td>
      <td>3893.053124</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3805</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>0.141129</td>
      <td>0.060336</td>
      <td>0.091281</td>
      <td>1.0</td>
      <td>0.10</td>
      <td>0.60</td>
      <td>0.30</td>
      <td>13238.460497</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3806</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>0.214061</td>
      <td>0.058423</td>
      <td>0.104104</td>
      <td>1.0</td>
      <td>0.28</td>
      <td>0.59</td>
      <td>0.13</td>
      <td>21410.246051</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3807</th>
      <td>KETSANA</td>
      <td>9236</td>
      <td>0.242485</td>
      <td>0.144241</td>
      <td>0.126329</td>
      <td>0.139054</td>
      <td>0.008478</td>
      <td>0.307074</td>
      <td>0.147618</td>
      <td>0.257436</td>
      <td>...</td>
      <td>0.184985</td>
      <td>0.033297</td>
      <td>0.091221</td>
      <td>1.0</td>
      <td>0.24</td>
      <td>0.62</td>
      <td>0.14</td>
      <td>27185.054763</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3808</th>
      <td>KETSANA</td>
      <td>9237</td>
      <td>0.244396</td>
      <td>0.108748</td>
      <td>0.143194</td>
      <td>0.143739</td>
      <td>0.005643</td>
      <td>0.320598</td>
      <td>0.088132</td>
      <td>0.140876</td>
      <td>...</td>
      <td>0.103623</td>
      <td>0.028936</td>
      <td>0.140126</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.89</td>
      <td>0.11</td>
      <td>9535.117048</td>
      <td>0.000000</td>
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
      <th>43270</th>
      <td>SAUDEL</td>
      <td>18795</td>
      <td>0.061937</td>
      <td>0.877058</td>
      <td>0.091758</td>
      <td>0.077746</td>
      <td>0.004009</td>
      <td>0.358942</td>
      <td>0.094337</td>
      <td>0.197095</td>
      <td>...</td>
      <td>0.158586</td>
      <td>0.015329</td>
      <td>0.265042</td>
      <td>1.0</td>
      <td>0.12</td>
      <td>0.16</td>
      <td>0.72</td>
      <td>12112.204272</td>
      <td>1.133269</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>43271</th>
      <td>SAUDEL</td>
      <td>18796</td>
      <td>0.059568</td>
      <td>0.907444</td>
      <td>0.090530</td>
      <td>0.066750</td>
      <td>0.008603</td>
      <td>0.548934</td>
      <td>0.043329</td>
      <td>0.115342</td>
      <td>...</td>
      <td>0.095746</td>
      <td>0.012607</td>
      <td>0.089733</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.80</td>
      <td>23128.451605</td>
      <td>0.922223</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>43272</th>
      <td>SAUDEL</td>
      <td>18797</td>
      <td>0.053634</td>
      <td>0.938190</td>
      <td>0.076702</td>
      <td>0.057364</td>
      <td>0.000619</td>
      <td>0.329734</td>
      <td>0.066634</td>
      <td>0.119561</td>
      <td>...</td>
      <td>0.088262</td>
      <td>0.013689</td>
      <td>0.077939</td>
      <td>1.0</td>
      <td>0.03</td>
      <td>0.07</td>
      <td>0.90</td>
      <td>361.762983</td>
      <td>1.475799</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>43273</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>0.117307</td>
      <td>0.016159</td>
      <td>0.095031</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>2407.611398</td>
      <td>1.310422</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>43274</th>
      <td>SAUDEL</td>
      <td>18963</td>
      <td>0.060293</td>
      <td>0.889904</td>
      <td>0.057833</td>
      <td>0.050505</td>
      <td>0.001149</td>
      <td>0.406977</td>
      <td>0.085628</td>
      <td>0.101782</td>
      <td>...</td>
      <td>0.084217</td>
      <td>0.013664</td>
      <td>0.064003</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>2750.286411</td>
      <td>0.977414</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>6111 rows × 21 columns</p>
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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>...</th>
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
      <td>DURIAN</td>
      <td>8284</td>
      <td>0.168433</td>
      <td>0.879044</td>
      <td>0.010371</td>
      <td>0.009660</td>
      <td>2.334511e-07</td>
      <td>0.330964</td>
      <td>0.421621</td>
      <td>0.526082</td>
      <td>...</td>
      <td>0.413882</td>
      <td>0.028344</td>
      <td>0.031312</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN</td>
      <td>8286</td>
      <td>0.154462</td>
      <td>0.949392</td>
      <td>0.014378</td>
      <td>0.010594</td>
      <td>9.712553e-05</td>
      <td>0.504983</td>
      <td>0.460007</td>
      <td>0.460011</td>
      <td>...</td>
      <td>0.304474</td>
      <td>0.043049</td>
      <td>0.360224</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.140000</td>
      <td>0.860000</td>
      <td>276.871504</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>8450</td>
      <td>0.176799</td>
      <td>0.839346</td>
      <td>0.011082</td>
      <td>0.013105</td>
      <td>1.371717e-05</td>
      <td>0.155316</td>
      <td>0.645959</td>
      <td>0.772542</td>
      <td>...</td>
      <td>0.649623</td>
      <td>0.057690</td>
      <td>0.393828</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.110000</td>
      <td>0.890000</td>
      <td>448.539453</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN</td>
      <td>8451</td>
      <td>0.169135</td>
      <td>0.874636</td>
      <td>0.008788</td>
      <td>0.010400</td>
      <td>1.544535e-04</td>
      <td>0.324958</td>
      <td>0.375933</td>
      <td>0.487333</td>
      <td>...</td>
      <td>0.380232</td>
      <td>0.036547</td>
      <td>0.317867</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.120000</td>
      <td>0.880000</td>
      <td>2101.708435</td>
      <td>0.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN</td>
      <td>8452</td>
      <td>0.161895</td>
      <td>0.909926</td>
      <td>0.009111</td>
      <td>0.008968</td>
      <td>1.690249e-03</td>
      <td>0.294989</td>
      <td>0.453334</td>
      <td>0.461002</td>
      <td>...</td>
      <td>0.310462</td>
      <td>0.062176</td>
      <td>0.515864</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.460000</td>
      <td>0.470000</td>
      <td>11632.726327</td>
      <td>0.000000</td>
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
      <th>49749</th>
      <td>MOLAVE</td>
      <td>19306</td>
      <td>0.108159</td>
      <td>0.816770</td>
      <td>0.034958</td>
      <td>0.032650</td>
      <td>7.311641e-05</td>
      <td>0.111296</td>
      <td>0.327556</td>
      <td>0.563018</td>
      <td>...</td>
      <td>0.433746</td>
      <td>0.026779</td>
      <td>0.011391</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.990000</td>
      <td>330.215768</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49750</th>
      <td>MOLAVE</td>
      <td>19307</td>
      <td>0.102816</td>
      <td>0.852281</td>
      <td>0.033634</td>
      <td>0.028851</td>
      <td>2.854586e-03</td>
      <td>0.379331</td>
      <td>0.129550</td>
      <td>0.275946</td>
      <td>...</td>
      <td>0.208269</td>
      <td>0.019943</td>
      <td>0.271370</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.190000</td>
      <td>0.760000</td>
      <td>5409.607943</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49751</th>
      <td>MOLAVE</td>
      <td>19308</td>
      <td>0.096754</td>
      <td>0.887792</td>
      <td>0.029724</td>
      <td>0.025744</td>
      <td>1.096340e-03</td>
      <td>0.498339</td>
      <td>0.129227</td>
      <td>0.261104</td>
      <td>...</td>
      <td>0.204722</td>
      <td>0.016272</td>
      <td>0.087050</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.970000</td>
      <td>5378.401365</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49752</th>
      <td>MOLAVE</td>
      <td>19476</td>
      <td>0.092212</td>
      <td>0.923300</td>
      <td>0.058092</td>
      <td>0.035882</td>
      <td>3.178534e-05</td>
      <td>0.286545</td>
      <td>0.132543</td>
      <td>0.177055</td>
      <td>...</td>
      <td>0.151533</td>
      <td>0.015221</td>
      <td>0.031742</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.027273</td>
      <td>0.972727</td>
      <td>914.677196</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49753</th>
      <td>MOLAVE</td>
      <td>19643</td>
      <td>0.092815</td>
      <td>0.923297</td>
      <td>0.064748</td>
      <td>0.038826</td>
      <td>2.127484e-04</td>
      <td>0.315199</td>
      <td>0.179078</td>
      <td>0.324393</td>
      <td>...</td>
      <td>0.254262</td>
      <td>0.025869</td>
      <td>0.057267</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.980000</td>
      <td>1466.117288</td>
      <td>1.143833</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>43643 rows × 21 columns</p>
</div>



```python
df_test["typhoon_name"].unique()
```




    array(['KETSANA', 'BOPHA', 'HAIMA', 'MANGKHUT', 'SAUDEL'], dtype=object)




```python
# Split X and y from dataframe features
X_test = df_test[features]
X_train = df_train[features]

y_train = df_train["percent_houses_damaged"]
y_test = df_test["percent_houses_damaged"]
```


```python
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

    /Users/mersedehkooshki/opt/anaconda3/envs/global-storm/lib/python3.8/site-packages/xgboost/data.py:250: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      elif isinstance(data.columns, (pd.Int64Index, pd.RangeIndex)):


    [18:45:37] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.215
    Model:                                OLS   Adj. R-squared:                  0.215
    Method:                     Least Squares   F-statistic:                     704.8
    Date:                    Thu, 23 Mar 2023   Prob (F-statistic):               0.00
    Time:                            18:45:40   Log-Likelihood:            -1.2620e+05
    No. Observations:                   43643   AIC:                         2.524e+05
    Df Residuals:                       43625   BIC:                         2.526e+05
    Df Model:                              17                                         
    Covariance Type:                nonrobust                                         
    =================================================================================================
                                        coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------------
    const                            -4.2635      0.123    -34.789      0.000      -4.504      -4.023
    wind_speed                       17.6589      0.194     91.138      0.000      17.279      18.039
    track_distance                    3.6361      0.114     31.914      0.000       3.413       3.859
    total_houses                      3.5889      0.564      6.368      0.000       2.484       4.693
    rainfall_max_6h                  -4.2084      0.551     -7.645      0.000      -5.287      -3.129
    rainfall_max_24h                  0.5180      1.565      0.331      0.741      -2.550       3.586
    rwi                              -0.9921      0.241     -4.118      0.000      -1.464      -0.520
    mean_slope                       -7.2427      2.025     -3.576      0.000     -11.212      -3.273
    std_slope                         0.2400      1.022      0.235      0.814      -1.763       2.243
    mean_tri                          7.1147      2.100      3.389      0.001       2.999      11.230
    std_tri                           0.5063      1.191      0.425      0.671      -1.829       2.841
    mean_elev                        -0.3810      0.283     -1.347      0.178      -0.936       0.174
    coast_length                      1.0542      0.255      4.135      0.000       0.554       1.554
    with_coast                        0.0463      0.074      0.626      0.531      -0.099       0.191
    urban                            -1.3979      0.130    -10.718      0.000      -1.654      -1.142
    rural                            -1.5669      0.076    -20.580      0.000      -1.716      -1.418
    water                            -1.2987      0.073    -17.750      0.000      -1.442      -1.155
    total_pop                     -3.988e-07   5.35e-07     -0.746      0.456   -1.45e-06    6.49e-07
    percent_houses_damaged_5years     0.0725      0.017      4.256      0.000       0.039       0.106
    ==============================================================================
    Omnibus:                    62175.634   Durbin-Watson:                   0.641
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22251468.950
    Skew:                           8.520   Prob(JB):                         0.00
    Kurtosis:                     112.298   Cond. No.                     4.50e+18
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 4.81e-23. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.



```python
# Make prediction
y_pred_train = xgb.predict(X_train)
y_pred_train_clipped = y_pred_train.clip(0, 100)

y_pred = xgb.predict(X_test)
y_pred_clipped = y_pred.clip(0, 100)
```


```python
y_pred = y_pred_clipped.tolist()
y_true = df_test["percent_houses_damaged"].tolist()
```


```python
df_test.reset_index(drop=True, inplace=True)
for i in range(len(df_test)):
    df_test.at[i, "y_pred"] = y_pred[i]
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
      <th>typhoon_name</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
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
      <th>y_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KETSANA</td>
      <td>9233</td>
      <td>0.228997</td>
      <td>0.250719</td>
      <td>0.044070</td>
      <td>0.044935</td>
      <td>0.000753</td>
      <td>0.462625</td>
      <td>0.069010</td>
      <td>0.090805</td>
      <td>...</td>
      <td>0.020180</td>
      <td>0.031687</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.92</td>
      <td>3893.053124</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.025106</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>0.060336</td>
      <td>0.091281</td>
      <td>1.0</td>
      <td>0.10</td>
      <td>0.60</td>
      <td>0.30</td>
      <td>13238.460497</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>0.058423</td>
      <td>0.104104</td>
      <td>1.0</td>
      <td>0.28</td>
      <td>0.59</td>
      <td>0.13</td>
      <td>21410.246051</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KETSANA</td>
      <td>9236</td>
      <td>0.242485</td>
      <td>0.144241</td>
      <td>0.126329</td>
      <td>0.139054</td>
      <td>0.008478</td>
      <td>0.307074</td>
      <td>0.147618</td>
      <td>0.257436</td>
      <td>...</td>
      <td>0.033297</td>
      <td>0.091221</td>
      <td>1.0</td>
      <td>0.24</td>
      <td>0.62</td>
      <td>0.14</td>
      <td>27185.054763</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.166181</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KETSANA</td>
      <td>9237</td>
      <td>0.244396</td>
      <td>0.108748</td>
      <td>0.143194</td>
      <td>0.143739</td>
      <td>0.005643</td>
      <td>0.320598</td>
      <td>0.088132</td>
      <td>0.140876</td>
      <td>...</td>
      <td>0.028936</td>
      <td>0.140126</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.89</td>
      <td>0.11</td>
      <td>9535.117048</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.567609</td>
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
      <th>6106</th>
      <td>SAUDEL</td>
      <td>18795</td>
      <td>0.061937</td>
      <td>0.877058</td>
      <td>0.091758</td>
      <td>0.077746</td>
      <td>0.004009</td>
      <td>0.358942</td>
      <td>0.094337</td>
      <td>0.197095</td>
      <td>...</td>
      <td>0.015329</td>
      <td>0.265042</td>
      <td>1.0</td>
      <td>0.12</td>
      <td>0.16</td>
      <td>0.72</td>
      <td>12112.204272</td>
      <td>1.133269</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6107</th>
      <td>SAUDEL</td>
      <td>18796</td>
      <td>0.059568</td>
      <td>0.907444</td>
      <td>0.090530</td>
      <td>0.066750</td>
      <td>0.008603</td>
      <td>0.548934</td>
      <td>0.043329</td>
      <td>0.115342</td>
      <td>...</td>
      <td>0.012607</td>
      <td>0.089733</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.80</td>
      <td>23128.451605</td>
      <td>0.922223</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6108</th>
      <td>SAUDEL</td>
      <td>18797</td>
      <td>0.053634</td>
      <td>0.938190</td>
      <td>0.076702</td>
      <td>0.057364</td>
      <td>0.000619</td>
      <td>0.329734</td>
      <td>0.066634</td>
      <td>0.119561</td>
      <td>...</td>
      <td>0.013689</td>
      <td>0.077939</td>
      <td>1.0</td>
      <td>0.03</td>
      <td>0.07</td>
      <td>0.90</td>
      <td>361.762983</td>
      <td>1.475799</td>
      <td>0.0</td>
      <td>0.008930</td>
    </tr>
    <tr>
      <th>6109</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>0.016159</td>
      <td>0.095031</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>2407.611398</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
    </tr>
    <tr>
      <th>6110</th>
      <td>SAUDEL</td>
      <td>18963</td>
      <td>0.060293</td>
      <td>0.889904</td>
      <td>0.057833</td>
      <td>0.050505</td>
      <td>0.001149</td>
      <td>0.406977</td>
      <td>0.085628</td>
      <td>0.101782</td>
      <td>...</td>
      <td>0.013664</td>
      <td>0.064003</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>2750.286411</td>
      <td>0.977414</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>6111 rows × 22 columns</p>
</div>




```python
# Read a CSV file including grid_id and mun_code and import to a df
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




```python
# join main df to the weight df based on grid_point_id
join_final = df_test.merge(df_weight, on="grid_point_id", how="left")
join_final
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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>...</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>y_pred</th>
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
      <td>KETSANA</td>
      <td>9233</td>
      <td>0.228997</td>
      <td>0.250719</td>
      <td>0.044070</td>
      <td>0.044935</td>
      <td>0.000753</td>
      <td>0.462625</td>
      <td>0.069010</td>
      <td>0.090805</td>
      <td>...</td>
      <td>3893.053124</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.025106</td>
      <td>PH015514000</td>
      <td>9233.0</td>
      <td>119.8E_16.4N</td>
      <td>689</td>
      <td>689</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>13238.460497</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
      <td>PH015508000</td>
      <td>9234.0</td>
      <td>119.8E_16.3N</td>
      <td>1844</td>
      <td>5089</td>
      <td>0.362350</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>13238.460497</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
      <td>PH015514000</td>
      <td>9234.0</td>
      <td>119.8E_16.3N</td>
      <td>3245</td>
      <td>5089</td>
      <td>0.637650</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>21410.246051</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
      <td>PH015501000</td>
      <td>9235.0</td>
      <td>119.8E_16.2N</td>
      <td>1351</td>
      <td>6106</td>
      <td>0.221258</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>21410.246051</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
      <td>PH015508000</td>
      <td>9235.0</td>
      <td>119.8E_16.2N</td>
      <td>4755</td>
      <td>6106</td>
      <td>0.778742</td>
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
      <th>18337</th>
      <td>SAUDEL</td>
      <td>18796</td>
      <td>0.059568</td>
      <td>0.907444</td>
      <td>0.090530</td>
      <td>0.066750</td>
      <td>0.008603</td>
      <td>0.548934</td>
      <td>0.043329</td>
      <td>0.115342</td>
      <td>...</td>
      <td>23128.451605</td>
      <td>0.922223</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>18796.0</td>
      <td>125.5E_12.0N</td>
      <td>4070</td>
      <td>5944</td>
      <td>0.684724</td>
    </tr>
    <tr>
      <th>18338</th>
      <td>SAUDEL</td>
      <td>18797</td>
      <td>0.053634</td>
      <td>0.938190</td>
      <td>0.076702</td>
      <td>0.057364</td>
      <td>0.000619</td>
      <td>0.329734</td>
      <td>0.066634</td>
      <td>0.119561</td>
      <td>...</td>
      <td>361.762983</td>
      <td>1.475799</td>
      <td>0.0</td>
      <td>0.008930</td>
      <td>PH082622000</td>
      <td>18797.0</td>
      <td>125.5E_11.9N</td>
      <td>463</td>
      <td>463</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>18339</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>2407.611398</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
      <td>PH082606000</td>
      <td>18962.0</td>
      <td>125.6E_12.1N</td>
      <td>77</td>
      <td>461</td>
      <td>0.167028</td>
    </tr>
    <tr>
      <th>18340</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>2407.611398</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
      <td>PH082617000</td>
      <td>18962.0</td>
      <td>125.6E_12.1N</td>
      <td>384</td>
      <td>461</td>
      <td>0.832972</td>
    </tr>
    <tr>
      <th>18341</th>
      <td>SAUDEL</td>
      <td>18963</td>
      <td>0.060293</td>
      <td>0.889904</td>
      <td>0.057833</td>
      <td>0.050505</td>
      <td>0.001149</td>
      <td>0.406977</td>
      <td>0.085628</td>
      <td>0.101782</td>
      <td>...</td>
      <td>2750.286411</td>
      <td>0.977414</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>18963.0</td>
      <td>125.6E_12.0N</td>
      <td>731</td>
      <td>731</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>18342 rows × 28 columns</p>
</div>




```python
# Remove all columns between column index 21 to 25
join_final.drop(join_final.iloc[:, 23:27], inplace=True, axis=1)
join_final
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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>...</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>y_pred</th>
      <th>ADM3_PCODE</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KETSANA</td>
      <td>9233</td>
      <td>0.228997</td>
      <td>0.250719</td>
      <td>0.044070</td>
      <td>0.044935</td>
      <td>0.000753</td>
      <td>0.462625</td>
      <td>0.069010</td>
      <td>0.090805</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.92</td>
      <td>3893.053124</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.025106</td>
      <td>PH015514000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.10</td>
      <td>0.60</td>
      <td>0.30</td>
      <td>13238.460497</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
      <td>PH015508000</td>
      <td>0.362350</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.10</td>
      <td>0.60</td>
      <td>0.30</td>
      <td>13238.460497</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
      <td>PH015514000</td>
      <td>0.637650</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.28</td>
      <td>0.59</td>
      <td>0.13</td>
      <td>21410.246051</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
      <td>PH015501000</td>
      <td>0.221258</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.28</td>
      <td>0.59</td>
      <td>0.13</td>
      <td>21410.246051</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
      <td>PH015508000</td>
      <td>0.778742</td>
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
      <th>18337</th>
      <td>SAUDEL</td>
      <td>18796</td>
      <td>0.059568</td>
      <td>0.907444</td>
      <td>0.090530</td>
      <td>0.066750</td>
      <td>0.008603</td>
      <td>0.548934</td>
      <td>0.043329</td>
      <td>0.115342</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.80</td>
      <td>23128.451605</td>
      <td>0.922223</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>0.684724</td>
    </tr>
    <tr>
      <th>18338</th>
      <td>SAUDEL</td>
      <td>18797</td>
      <td>0.053634</td>
      <td>0.938190</td>
      <td>0.076702</td>
      <td>0.057364</td>
      <td>0.000619</td>
      <td>0.329734</td>
      <td>0.066634</td>
      <td>0.119561</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.03</td>
      <td>0.07</td>
      <td>0.90</td>
      <td>361.762983</td>
      <td>1.475799</td>
      <td>0.0</td>
      <td>0.008930</td>
      <td>PH082622000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>18339</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>2407.611398</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
      <td>PH082606000</td>
      <td>0.167028</td>
    </tr>
    <tr>
      <th>18340</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>2407.611398</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
      <td>PH082617000</td>
      <td>0.832972</td>
    </tr>
    <tr>
      <th>18341</th>
      <td>SAUDEL</td>
      <td>18963</td>
      <td>0.060293</td>
      <td>0.889904</td>
      <td>0.057833</td>
      <td>0.050505</td>
      <td>0.001149</td>
      <td>0.406977</td>
      <td>0.085628</td>
      <td>0.101782</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>2750.286411</td>
      <td>0.977414</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
<p>18342 rows × 24 columns</p>
</div>




```python
# Multiply %damg and also %predicted_damg with total_houses and weight
join_final["weight*%damg*houses"] = (
    join_final["weight"]
    * join_final["percent_houses_damaged"]
    * join_final["total_houses"]
) / 100
join_final["weight*%predicted_damg*houses"] = (
    join_final["weight"] * join_final["y_pred"] * join_final["total_houses"]
) / 100

# Multiply total_houses with weight
join_final["weight*houses"] = (join_final["weight"] * join_final["total_houses"]) / 100

join_final
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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>...</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>y_pred</th>
      <th>ADM3_PCODE</th>
      <th>weight</th>
      <th>weight*%damg*houses</th>
      <th>weight*%predicted_damg*houses</th>
      <th>weight*houses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KETSANA</td>
      <td>9233</td>
      <td>0.228997</td>
      <td>0.250719</td>
      <td>0.044070</td>
      <td>0.044935</td>
      <td>0.000753</td>
      <td>0.462625</td>
      <td>0.069010</td>
      <td>0.090805</td>
      <td>...</td>
      <td>0.92</td>
      <td>3893.053124</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.025106</td>
      <td>PH015514000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.106424e-05</td>
      <td>0.000441</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>0.30</td>
      <td>13238.460497</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
      <td>PH015508000</td>
      <td>0.362350</td>
      <td>0.0</td>
      <td>1.625420e-05</td>
      <td>0.000260</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>0.30</td>
      <td>13238.460497</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
      <td>PH015514000</td>
      <td>0.637650</td>
      <td>0.0</td>
      <td>2.860352e-05</td>
      <td>0.000458</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>0.13</td>
      <td>21410.246051</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
      <td>PH015501000</td>
      <td>0.221258</td>
      <td>0.0</td>
      <td>1.908737e-05</td>
      <td>0.000219</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>0.13</td>
      <td>21410.246051</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
      <td>PH015508000</td>
      <td>0.778742</td>
      <td>0.0</td>
      <td>6.718019e-05</td>
      <td>0.000770</td>
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
      <th>18337</th>
      <td>SAUDEL</td>
      <td>18796</td>
      <td>0.059568</td>
      <td>0.907444</td>
      <td>0.090530</td>
      <td>0.066750</td>
      <td>0.008603</td>
      <td>0.548934</td>
      <td>0.043329</td>
      <td>0.115342</td>
      <td>...</td>
      <td>0.80</td>
      <td>23128.451605</td>
      <td>0.922223</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>0.684724</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000620</td>
    </tr>
    <tr>
      <th>18338</th>
      <td>SAUDEL</td>
      <td>18797</td>
      <td>0.053634</td>
      <td>0.938190</td>
      <td>0.076702</td>
      <td>0.057364</td>
      <td>0.000619</td>
      <td>0.329734</td>
      <td>0.066634</td>
      <td>0.119561</td>
      <td>...</td>
      <td>0.90</td>
      <td>361.762983</td>
      <td>1.475799</td>
      <td>0.0</td>
      <td>0.008930</td>
      <td>PH082622000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>6.849179e-06</td>
      <td>0.000767</td>
    </tr>
    <tr>
      <th>18339</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>0.97</td>
      <td>2407.611398</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
      <td>PH082606000</td>
      <td>0.167028</td>
      <td>0.0</td>
      <td>5.274526e-07</td>
      <td>0.000090</td>
    </tr>
    <tr>
      <th>18340</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>0.97</td>
      <td>2407.611398</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
      <td>PH082617000</td>
      <td>0.832972</td>
      <td>0.0</td>
      <td>2.630413e-06</td>
      <td>0.000449</td>
    </tr>
    <tr>
      <th>18341</th>
      <td>SAUDEL</td>
      <td>18963</td>
      <td>0.060293</td>
      <td>0.889904</td>
      <td>0.057833</td>
      <td>0.050505</td>
      <td>0.001149</td>
      <td>0.406977</td>
      <td>0.085628</td>
      <td>0.101782</td>
      <td>...</td>
      <td>0.97</td>
      <td>2750.286411</td>
      <td>0.977414</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000578</td>
    </tr>
  </tbody>
</table>
<p>18342 rows × 27 columns</p>
</div>




```python
# Read CSV file which includes regoin name and code
region_df = pd.read_csv("data/adm3_area.csv", index_col=0)
region_df.head()
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
      <th>Shape_Leng</th>
      <th>Shape_Area</th>
      <th>ADM3_EN</th>
      <th>ADM3_PCODE</th>
      <th>ADM3_REF</th>
      <th>ADM3ALT1EN</th>
      <th>ADM3ALT2EN</th>
      <th>ADM2_EN</th>
      <th>ADM2_PCODE</th>
      <th>ADM1_EN</th>
      <th>ADM1_PCODE</th>
      <th>ADM0_EN</th>
      <th>ADM0_PCODE</th>
      <th>date</th>
      <th>validOn</th>
      <th>validTo</th>
      <th>geometry</th>
      <th>Area</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.601219</td>
      <td>0.063496</td>
      <td>Aborlan</td>
      <td>PH175301000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Palawan</td>
      <td>PH175300000</td>
      <td>Region IV-B</td>
      <td>PH170000000</td>
      <td>Philippines (the)</td>
      <td>PH</td>
      <td>2016-06-30</td>
      <td>2020-05-29</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((13200654.48649568 1032355.1025...</td>
      <td>7.711206e+08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.078749</td>
      <td>0.050232</td>
      <td>Abra de Ilog</td>
      <td>PH175101000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Occidental Mindoro</td>
      <td>PH175100000</td>
      <td>Region IV-B</td>
      <td>PH170000000</td>
      <td>Philippines (the)</td>
      <td>PH</td>
      <td>2016-06-30</td>
      <td>2020-05-29</td>
      <td>NaN</td>
      <td>POLYGON ((13423362.387871413 1479551.980005401...</td>
      <td>6.019146e+08</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.424301</td>
      <td>0.006453</td>
      <td>Abucay</td>
      <td>PH030801000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Bataan</td>
      <td>PH030800000</td>
      <td>Region III</td>
      <td>PH030000000</td>
      <td>Philippines (the)</td>
      <td>PH</td>
      <td>2016-06-30</td>
      <td>2020-05-29</td>
      <td>NaN</td>
      <td>POLYGON ((13413856.918075956 1614138.946940594...</td>
      <td>7.688903e+07</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.566053</td>
      <td>0.011343</td>
      <td>Abulug</td>
      <td>PH021501000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Cagayan</td>
      <td>PH021500000</td>
      <td>Region II</td>
      <td>PH020000000</td>
      <td>Philippines (the)</td>
      <td>PH</td>
      <td>2016-06-30</td>
      <td>2020-05-29</td>
      <td>NaN</td>
      <td>POLYGON ((13518031.78157248 2007651.089252317,...</td>
      <td>1.326682e+08</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.013649</td>
      <td>0.026124</td>
      <td>Abuyog</td>
      <td>PH083701000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Leyte</td>
      <td>PH083700000</td>
      <td>Region VIII</td>
      <td>PH080000000</td>
      <td>Philippines (the)</td>
      <td>PH</td>
      <td>2016-06-30</td>
      <td>2020-05-29</td>
      <td>NaN</td>
      <td>MULTIPOLYGON (((13917924.3505296 1180265.08047...</td>
      <td>3.161752e+08</td>
    </tr>
  </tbody>
</table>
</div>




```python
# join regoin_code column to the main df(join_final) based on mun_code
join_region_df = join_final.merge(
    region_df[["ADM1_EN", "ADM1_PCODE", "ADM3_PCODE"]], on="ADM3_PCODE", how="left"
)
join_region_df
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
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>total_houses</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>rwi</th>
      <th>mean_slope</th>
      <th>std_slope</th>
      <th>...</th>
      <th>percent_houses_damaged_5years</th>
      <th>percent_houses_damaged</th>
      <th>y_pred</th>
      <th>ADM3_PCODE</th>
      <th>weight</th>
      <th>weight*%damg*houses</th>
      <th>weight*%predicted_damg*houses</th>
      <th>weight*houses</th>
      <th>ADM1_EN</th>
      <th>ADM1_PCODE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KETSANA</td>
      <td>9233</td>
      <td>0.228997</td>
      <td>0.250719</td>
      <td>0.044070</td>
      <td>0.044935</td>
      <td>0.000753</td>
      <td>0.462625</td>
      <td>0.069010</td>
      <td>0.090805</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.025106</td>
      <td>PH015514000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>1.106424e-05</td>
      <td>0.000441</td>
      <td>Region I</td>
      <td>PH010000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
      <td>PH015508000</td>
      <td>0.362350</td>
      <td>0.0</td>
      <td>1.625420e-05</td>
      <td>0.000260</td>
      <td>Region I</td>
      <td>PH010000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KETSANA</td>
      <td>9234</td>
      <td>0.236445</td>
      <td>0.215226</td>
      <td>0.071856</td>
      <td>0.077553</td>
      <td>0.005352</td>
      <td>0.324779</td>
      <td>0.084089</td>
      <td>0.189581</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.062428</td>
      <td>PH015514000</td>
      <td>0.637650</td>
      <td>0.0</td>
      <td>2.860352e-05</td>
      <td>0.000458</td>
      <td>Region I</td>
      <td>PH010000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
      <td>PH015501000</td>
      <td>0.221258</td>
      <td>0.0</td>
      <td>1.908737e-05</td>
      <td>0.000219</td>
      <td>Region I</td>
      <td>PH010000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>KETSANA</td>
      <td>9235</td>
      <td>0.239295</td>
      <td>0.179733</td>
      <td>0.098866</td>
      <td>0.114486</td>
      <td>0.006004</td>
      <td>0.308338</td>
      <td>0.139945</td>
      <td>0.291882</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.087257</td>
      <td>PH015508000</td>
      <td>0.778742</td>
      <td>0.0</td>
      <td>6.718019e-05</td>
      <td>0.000770</td>
      <td>Region I</td>
      <td>PH010000000</td>
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
      <th>18337</th>
      <td>SAUDEL</td>
      <td>18796</td>
      <td>0.059568</td>
      <td>0.907444</td>
      <td>0.090530</td>
      <td>0.066750</td>
      <td>0.008603</td>
      <td>0.548934</td>
      <td>0.043329</td>
      <td>0.115342</td>
      <td>...</td>
      <td>0.922223</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>0.684724</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000620</td>
      <td>Region VIII</td>
      <td>PH080000000</td>
    </tr>
    <tr>
      <th>18338</th>
      <td>SAUDEL</td>
      <td>18797</td>
      <td>0.053634</td>
      <td>0.938190</td>
      <td>0.076702</td>
      <td>0.057364</td>
      <td>0.000619</td>
      <td>0.329734</td>
      <td>0.066634</td>
      <td>0.119561</td>
      <td>...</td>
      <td>1.475799</td>
      <td>0.0</td>
      <td>0.008930</td>
      <td>PH082622000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>6.849179e-06</td>
      <td>0.000767</td>
      <td>Region VIII</td>
      <td>PH080000000</td>
    </tr>
    <tr>
      <th>18339</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
      <td>PH082606000</td>
      <td>0.167028</td>
      <td>0.0</td>
      <td>5.274526e-07</td>
      <td>0.000090</td>
      <td>Region VIII</td>
      <td>PH080000000</td>
    </tr>
    <tr>
      <th>18340</th>
      <td>SAUDEL</td>
      <td>18962</td>
      <td>0.062720</td>
      <td>0.858897</td>
      <td>0.053956</td>
      <td>0.054691</td>
      <td>0.001009</td>
      <td>0.372093</td>
      <td>0.106989</td>
      <td>0.150028</td>
      <td>...</td>
      <td>1.310422</td>
      <td>0.0</td>
      <td>0.005853</td>
      <td>PH082617000</td>
      <td>0.832972</td>
      <td>0.0</td>
      <td>2.630413e-06</td>
      <td>0.000449</td>
      <td>Region VIII</td>
      <td>PH080000000</td>
    </tr>
    <tr>
      <th>18341</th>
      <td>SAUDEL</td>
      <td>18963</td>
      <td>0.060293</td>
      <td>0.889904</td>
      <td>0.057833</td>
      <td>0.050505</td>
      <td>0.001149</td>
      <td>0.406977</td>
      <td>0.085628</td>
      <td>0.101782</td>
      <td>...</td>
      <td>0.977414</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000578</td>
      <td>Region VIII</td>
      <td>PH080000000</td>
    </tr>
  </tbody>
</table>
<p>18342 rows × 29 columns</p>
</div>




```python
# Groupby by municipality with sum as the aggregation function
agg_df = join_region_df.groupby(["ADM3_PCODE", "ADM1_PCODE", "typhoon_name"]).agg(
    {
        "weight*%damg*houses": "sum",
        "weight*%predicted_damg*houses": "sum",
        "weight": "sum",
        "weight*houses": "sum",
    }
)
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
      <th>weight*%damg*houses</th>
      <th>weight*%predicted_damg*houses</th>
      <th>weight</th>
      <th>weight*houses</th>
    </tr>
    <tr>
      <th>ADM3_PCODE</th>
      <th>ADM1_PCODE</th>
      <th>typhoon_name</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">PH012801000</th>
      <th rowspan="4" valign="top">PH010000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.571093</td>
      <td>0.000338</td>
    </tr>
    <tr>
      <th>HAIMA</th>
      <td>0.000664</td>
      <td>0.004569</td>
      <td>1.571093</td>
      <td>0.002437</td>
    </tr>
    <tr>
      <th>MANGKHUT</th>
      <td>0.002289</td>
      <td>0.015345</td>
      <td>1.571093</td>
      <td>0.003863</td>
    </tr>
    <tr>
      <th>SAUDEL</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.571093</td>
      <td>0.000650</td>
    </tr>
    <tr>
      <th>PH012802000</th>
      <th>PH010000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.724799</td>
      <td>0.000044</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>PH175902000</th>
      <th>PH170000000</th>
      <th>KETSANA</th>
      <td>0.000000</td>
      <td>0.000388</td>
      <td>4.000000</td>
      <td>0.005455</td>
    </tr>
    <tr>
      <th>PH175905000</th>
      <th>PH170000000</th>
      <th>KETSANA</th>
      <td>0.000000</td>
      <td>0.000329</td>
      <td>1.000000</td>
      <td>0.001521</td>
    </tr>
    <tr>
      <th>PH175907000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.340199</td>
      <td>0.000043</td>
    </tr>
    <tr>
      <th>PH175914000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000004</td>
      <td>2.551962</td>
      <td>0.000365</td>
    </tr>
    <tr>
      <th>PH175915000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000001</td>
      <td>1.692708</td>
      <td>0.000204</td>
    </tr>
  </tbody>
</table>
<p>3223 rows × 4 columns</p>
</div>




```python
# Normalize by the sum of the weights
agg_df["damg_houses_per_mun"] = agg_df["weight*%damg*houses"] / agg_df["weight"]
agg_df["predicted_damg_houses_per_mun"] = (
    agg_df["weight*%predicted_damg*houses"] / agg_df["weight"]
)

agg_df["sum_of_weight_mun"] = agg_df["weight*houses"] / agg_df["weight"]

agg_df.head()
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
      <th>weight*%damg*houses</th>
      <th>weight*%predicted_damg*houses</th>
      <th>weight</th>
      <th>weight*houses</th>
      <th>damg_houses_per_mun</th>
      <th>predicted_damg_houses_per_mun</th>
      <th>sum_of_weight_mun</th>
    </tr>
    <tr>
      <th>ADM3_PCODE</th>
      <th>ADM1_PCODE</th>
      <th>typhoon_name</th>
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
      <th rowspan="4" valign="top">PH012801000</th>
      <th rowspan="4" valign="top">PH010000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.571093</td>
      <td>0.000338</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000215</td>
    </tr>
    <tr>
      <th>HAIMA</th>
      <td>0.000664</td>
      <td>0.004569</td>
      <td>1.571093</td>
      <td>0.002437</td>
      <td>0.000423</td>
      <td>0.002908</td>
      <td>0.001551</td>
    </tr>
    <tr>
      <th>MANGKHUT</th>
      <td>0.002289</td>
      <td>0.015345</td>
      <td>1.571093</td>
      <td>0.003863</td>
      <td>0.001457</td>
      <td>0.009767</td>
      <td>0.002459</td>
    </tr>
    <tr>
      <th>SAUDEL</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.571093</td>
      <td>0.000650</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000414</td>
    </tr>
    <tr>
      <th>PH012802000</th>
      <th>PH010000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.724799</td>
      <td>0.000044</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000061</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Keep only %damg_normalized and %pred_damg_normalized columns
agg_df.drop(agg_df.columns[:4], inplace=True, axis=1)
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
      <th>damg_houses_per_mun</th>
      <th>predicted_damg_houses_per_mun</th>
      <th>sum_of_weight_mun</th>
    </tr>
    <tr>
      <th>ADM3_PCODE</th>
      <th>ADM1_PCODE</th>
      <th>typhoon_name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="4" valign="top">PH012801000</th>
      <th rowspan="4" valign="top">PH010000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000215</td>
    </tr>
    <tr>
      <th>HAIMA</th>
      <td>0.000423</td>
      <td>2.908288e-03</td>
      <td>0.001551</td>
    </tr>
    <tr>
      <th>MANGKHUT</th>
      <td>0.001457</td>
      <td>9.767222e-03</td>
      <td>0.002459</td>
    </tr>
    <tr>
      <th>SAUDEL</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000414</td>
    </tr>
    <tr>
      <th>PH012802000</th>
      <th>PH010000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000061</td>
    </tr>
    <tr>
      <th>...</th>
      <th>...</th>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>PH175902000</th>
      <th>PH170000000</th>
      <th>KETSANA</th>
      <td>0.000000</td>
      <td>9.707278e-05</td>
      <td>0.001364</td>
    </tr>
    <tr>
      <th>PH175905000</th>
      <th>PH170000000</th>
      <th>KETSANA</th>
      <td>0.000000</td>
      <td>3.286993e-04</td>
      <td>0.001521</td>
    </tr>
    <tr>
      <th>PH175907000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000125</td>
    </tr>
    <tr>
      <th>PH175914000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>1.674685e-06</td>
      <td>0.000143</td>
    </tr>
    <tr>
      <th>PH175915000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>6.764880e-07</td>
      <td>0.000121</td>
    </tr>
  </tbody>
</table>
<p>3223 rows × 3 columns</p>
</div>




```python
# Groupby by regin with sum as the aggregation function
agg_df_1 = agg_df.groupby(["ADM1_PCODE", "typhoon_name"]).agg(
    {
        "damg_houses_per_mun": "sum",
        "predicted_damg_houses_per_mun": "sum",
        "sum_of_weight_mun": "sum",
    }
)
agg_df_1.head()
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
      <th>damg_houses_per_mun</th>
      <th>predicted_damg_houses_per_mun</th>
      <th>sum_of_weight_mun</th>
    </tr>
    <tr>
      <th>ADM1_PCODE</th>
      <th>typhoon_name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">PH010000000</th>
      <th>BOPHA</th>
      <td>0.000001</td>
      <td>0.000242</td>
      <td>0.017613</td>
    </tr>
    <tr>
      <th>HAIMA</th>
      <td>0.132145</td>
      <td>0.067877</td>
      <td>0.194996</td>
    </tr>
    <tr>
      <th>KETSANA</th>
      <td>0.000060</td>
      <td>0.006893</td>
      <td>0.111819</td>
    </tr>
    <tr>
      <th>MANGKHUT</th>
      <td>0.127631</td>
      <td>0.164637</td>
      <td>0.301262</td>
    </tr>
    <tr>
      <th>SAUDEL</th>
      <td>0.000000</td>
      <td>0.005398</td>
      <td>0.099009</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rename columns' names
agg_df_1 = agg_df_1.rename(
    columns={
        "damg_houses_per_mun": "damg_houses_per_Region",
        "predicted_damg_houses_per_mun": "predicted_damg_houses_per_Region",
        "sum_of_weight_mun": "sum_of_weight_region",
    }
)

agg_df_1.head()
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
      <th>damg_houses_per_Region</th>
      <th>predicted_damg_houses_per_Region</th>
      <th>sum_of_weight_region</th>
    </tr>
    <tr>
      <th>ADM1_PCODE</th>
      <th>typhoon_name</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">PH010000000</th>
      <th>BOPHA</th>
      <td>0.000001</td>
      <td>0.000242</td>
      <td>0.017613</td>
    </tr>
    <tr>
      <th>HAIMA</th>
      <td>0.132145</td>
      <td>0.067877</td>
      <td>0.194996</td>
    </tr>
    <tr>
      <th>KETSANA</th>
      <td>0.000060</td>
      <td>0.006893</td>
      <td>0.111819</td>
    </tr>
    <tr>
      <th>MANGKHUT</th>
      <td>0.127631</td>
      <td>0.164637</td>
      <td>0.301262</td>
    </tr>
    <tr>
      <th>SAUDEL</th>
      <td>0.000000</td>
      <td>0.005398</td>
      <td>0.099009</td>
    </tr>
  </tbody>
</table>
</div>




```python
agg_df_2 = agg_df_1.reset_index()
agg_df_2.head()
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
      <th>ADM1_PCODE</th>
      <th>typhoon_name</th>
      <th>damg_houses_per_Region</th>
      <th>predicted_damg_houses_per_Region</th>
      <th>sum_of_weight_region</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH010000000</td>
      <td>BOPHA</td>
      <td>0.000001</td>
      <td>0.000242</td>
      <td>0.017613</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH010000000</td>
      <td>HAIMA</td>
      <td>0.132145</td>
      <td>0.067877</td>
      <td>0.194996</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH010000000</td>
      <td>KETSANA</td>
      <td>0.000060</td>
      <td>0.006893</td>
      <td>0.111819</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH010000000</td>
      <td>MANGKHUT</td>
      <td>0.127631</td>
      <td>0.164637</td>
      <td>0.301262</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH010000000</td>
      <td>SAUDEL</td>
      <td>0.000000</td>
      <td>0.005398</td>
      <td>0.099009</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Estimate the percent difference of real and predicted damaged values  (First way)
agg_df_2["Percent_Difference_total_houses_based"] = (
    (agg_df_2["damg_houses_per_Region"] - agg_df_2["predicted_damg_houses_per_Region"])
    / (
        agg_df_2["sum_of_weight_region"]
    )  # (agg_df_2["damg_houses_per_Region"] + np.finfo(float).eps)
) * 100
```


```python
# Estimate the percent difference of real and predicted damaged values (Second way)
difference = (
    agg_df_2["damg_houses_per_Region"] - agg_df_2["predicted_damg_houses_per_Region"]
)
ave = (
    agg_df_2["damg_houses_per_Region"] + agg_df_2["predicted_damg_houses_per_Region"]
) / 2

agg_df_2["Percent_Difference_average_based"] = (abs(difference) / ave) * 100
agg_df_2
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
      <th>ADM1_PCODE</th>
      <th>typhoon_name</th>
      <th>damg_houses_per_Region</th>
      <th>predicted_damg_houses_per_Region</th>
      <th>sum_of_weight_region</th>
      <th>Percent_Difference_total_houses_based</th>
      <th>Percent_Difference_average_based</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH010000000</td>
      <td>BOPHA</td>
      <td>1.094611e-06</td>
      <td>2.424821e-04</td>
      <td>0.017613</td>
      <td>-1.370529</td>
      <td>198.202436</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH010000000</td>
      <td>HAIMA</td>
      <td>1.321450e-01</td>
      <td>6.787738e-02</td>
      <td>0.194996</td>
      <td>32.958484</td>
      <td>64.260414</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH010000000</td>
      <td>KETSANA</td>
      <td>5.969492e-05</td>
      <td>6.892515e-03</td>
      <td>0.111819</td>
      <td>-6.110583</td>
      <td>196.565413</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH010000000</td>
      <td>MANGKHUT</td>
      <td>1.276309e-01</td>
      <td>1.646369e-01</td>
      <td>0.301262</td>
      <td>-12.283663</td>
      <td>25.323341</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH010000000</td>
      <td>SAUDEL</td>
      <td>0.000000e+00</td>
      <td>5.397968e-03</td>
      <td>0.099009</td>
      <td>-5.452019</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PH020000000</td>
      <td>BOPHA</td>
      <td>0.000000e+00</td>
      <td>1.625500e-04</td>
      <td>0.017272</td>
      <td>-0.941131</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PH020000000</td>
      <td>HAIMA</td>
      <td>6.779745e-01</td>
      <td>3.108077e-01</td>
      <td>0.180373</td>
      <td>203.559265</td>
      <td>74.266467</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PH020000000</td>
      <td>KETSANA</td>
      <td>0.000000e+00</td>
      <td>5.698073e-03</td>
      <td>0.107808</td>
      <td>-5.285384</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PH020000000</td>
      <td>MANGKHUT</td>
      <td>4.488461e-01</td>
      <td>6.855844e-01</td>
      <td>0.208631</td>
      <td>-113.472285</td>
      <td>41.736938</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PH020000000</td>
      <td>SAUDEL</td>
      <td>0.000000e+00</td>
      <td>1.325783e-02</td>
      <td>0.102609</td>
      <td>-12.920680</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PH030000000</td>
      <td>BOPHA</td>
      <td>0.000000e+00</td>
      <td>1.351836e-07</td>
      <td>0.000952</td>
      <td>-0.014199</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PH030000000</td>
      <td>HAIMA</td>
      <td>5.615845e-04</td>
      <td>1.618386e-03</td>
      <td>0.058629</td>
      <td>-1.802512</td>
      <td>96.955580</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PH030000000</td>
      <td>KETSANA</td>
      <td>4.561876e-02</td>
      <td>3.885428e-02</td>
      <td>0.283215</td>
      <td>2.388462</td>
      <td>16.015704</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PH030000000</td>
      <td>MANGKHUT</td>
      <td>1.763859e-03</td>
      <td>2.749839e-03</td>
      <td>0.098552</td>
      <td>-1.000464</td>
      <td>43.688339</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PH030000000</td>
      <td>SAUDEL</td>
      <td>0.000000e+00</td>
      <td>2.615451e-02</td>
      <td>0.183101</td>
      <td>-14.284173</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PH040000000</td>
      <td>HAIMA</td>
      <td>8.540490e-07</td>
      <td>4.042321e-05</td>
      <td>0.003866</td>
      <td>-1.023477</td>
      <td>191.723781</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PH040000000</td>
      <td>KETSANA</td>
      <td>5.657947e-01</td>
      <td>4.293850e-01</td>
      <td>0.649474</td>
      <td>21.003093</td>
      <td>27.414090</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PH040000000</td>
      <td>SAUDEL</td>
      <td>6.619053e-05</td>
      <td>7.854769e-03</td>
      <td>0.124175</td>
      <td>-6.272242</td>
      <td>196.657449</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PH050000000</td>
      <td>BOPHA</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000104</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PH050000000</td>
      <td>HAIMA</td>
      <td>9.707499e-06</td>
      <td>2.387181e-04</td>
      <td>0.006650</td>
      <td>-3.443930</td>
      <td>184.369567</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PH050000000</td>
      <td>KETSANA</td>
      <td>0.000000e+00</td>
      <td>2.726016e-02</td>
      <td>0.120149</td>
      <td>-22.688620</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PH050000000</td>
      <td>SAUDEL</td>
      <td>0.000000e+00</td>
      <td>8.920129e-03</td>
      <td>0.084294</td>
      <td>-10.582172</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PH060000000</td>
      <td>BOPHA</td>
      <td>0.000000e+00</td>
      <td>4.755955e-03</td>
      <td>0.041024</td>
      <td>-11.593242</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PH070000000</td>
      <td>BOPHA</td>
      <td>9.247815e-03</td>
      <td>3.863762e-02</td>
      <td>0.104594</td>
      <td>-28.098951</td>
      <td>122.750504</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PH080000000</td>
      <td>BOPHA</td>
      <td>0.000000e+00</td>
      <td>5.986989e-04</td>
      <td>0.053142</td>
      <td>-1.126596</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PH080000000</td>
      <td>KETSANA</td>
      <td>0.000000e+00</td>
      <td>3.275871e-05</td>
      <td>0.008034</td>
      <td>-0.407757</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>PH080000000</td>
      <td>SAUDEL</td>
      <td>0.000000e+00</td>
      <td>4.594349e-04</td>
      <td>0.023955</td>
      <td>-1.917907</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>PH090000000</td>
      <td>BOPHA</td>
      <td>3.402632e-06</td>
      <td>1.274852e-02</td>
      <td>0.040768</td>
      <td>-31.262838</td>
      <td>199.893267</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PH100000000</td>
      <td>BOPHA</td>
      <td>2.946602e-02</td>
      <td>3.148425e-01</td>
      <td>0.106632</td>
      <td>-267.626501</td>
      <td>165.767892</td>
    </tr>
    <tr>
      <th>29</th>
      <td>PH110000000</td>
      <td>BOPHA</td>
      <td>1.282667e+00</td>
      <td>4.729969e-01</td>
      <td>0.080476</td>
      <td>1006.103064</td>
      <td>92.235226</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PH120000000</td>
      <td>BOPHA</td>
      <td>1.225662e-05</td>
      <td>3.456566e-04</td>
      <td>0.035587</td>
      <td>-0.936847</td>
      <td>186.302130</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PH130000000</td>
      <td>KETSANA</td>
      <td>1.540625e-01</td>
      <td>4.026151e-02</td>
      <td>0.138561</td>
      <td>82.130520</td>
      <td>117.125003</td>
    </tr>
    <tr>
      <th>32</th>
      <td>PH130000000</td>
      <td>SAUDEL</td>
      <td>0.000000e+00</td>
      <td>4.083420e-04</td>
      <td>0.025589</td>
      <td>-1.595789</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PH140000000</td>
      <td>BOPHA</td>
      <td>0.000000e+00</td>
      <td>3.389807e-06</td>
      <td>0.017597</td>
      <td>-0.019263</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PH140000000</td>
      <td>HAIMA</td>
      <td>1.828150e-01</td>
      <td>1.457814e-01</td>
      <td>0.166236</td>
      <td>22.277784</td>
      <td>22.540475</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PH140000000</td>
      <td>KETSANA</td>
      <td>1.000339e-06</td>
      <td>7.081980e-04</td>
      <td>0.064493</td>
      <td>-1.096554</td>
      <td>199.435792</td>
    </tr>
    <tr>
      <th>36</th>
      <td>PH140000000</td>
      <td>MANGKHUT</td>
      <td>1.200781e-01</td>
      <td>1.608988e-01</td>
      <td>0.191586</td>
      <td>-21.306796</td>
      <td>29.056292</td>
    </tr>
    <tr>
      <th>37</th>
      <td>PH140000000</td>
      <td>SAUDEL</td>
      <td>0.000000e+00</td>
      <td>3.397131e-03</td>
      <td>0.065647</td>
      <td>-5.174829</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>38</th>
      <td>PH150000000</td>
      <td>BOPHA</td>
      <td>1.002135e-02</td>
      <td>2.831747e-02</td>
      <td>0.089236</td>
      <td>-20.503124</td>
      <td>95.444403</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PH160000000</td>
      <td>BOPHA</td>
      <td>6.576090e-01</td>
      <td>1.206925e-01</td>
      <td>0.091569</td>
      <td>586.353550</td>
      <td>137.971322</td>
    </tr>
    <tr>
      <th>40</th>
      <td>PH170000000</td>
      <td>BOPHA</td>
      <td>8.893061e-03</td>
      <td>2.152328e-02</td>
      <td>0.019978</td>
      <td>-63.221954</td>
      <td>83.048895</td>
    </tr>
    <tr>
      <th>41</th>
      <td>PH170000000</td>
      <td>KETSANA</td>
      <td>0.000000e+00</td>
      <td>1.062324e-02</td>
      <td>0.066912</td>
      <td>-15.876545</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>PH170000000</td>
      <td>SAUDEL</td>
      <td>0.000000e+00</td>
      <td>6.259374e-04</td>
      <td>0.011757</td>
      <td>-5.324003</td>
      <td>200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
agg_df_2 = agg_df_2[
    [
        "ADM1_PCODE",
        "typhoon_name",
        "Percent_Difference_total_houses_based",
        "Percent_Difference_average_based",
    ]
]
```


```python
df_sorted = agg_df_2.sort_values(by=["typhoon_name"], ascending=-True).reset_index(
    drop=True
)
df_sorted
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
      <th>ADM1_PCODE</th>
      <th>typhoon_name</th>
      <th>Percent_Difference_total_houses_based</th>
      <th>Percent_Difference_average_based</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH010000000</td>
      <td>BOPHA</td>
      <td>-1.370529</td>
      <td>198.202436</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH120000000</td>
      <td>BOPHA</td>
      <td>-0.936847</td>
      <td>186.302130</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH110000000</td>
      <td>BOPHA</td>
      <td>1006.103064</td>
      <td>92.235226</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH100000000</td>
      <td>BOPHA</td>
      <td>-267.626501</td>
      <td>165.767892</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH170000000</td>
      <td>BOPHA</td>
      <td>-63.221954</td>
      <td>83.048895</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PH020000000</td>
      <td>BOPHA</td>
      <td>-0.941131</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PH090000000</td>
      <td>BOPHA</td>
      <td>-31.262838</td>
      <td>199.893267</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PH080000000</td>
      <td>BOPHA</td>
      <td>-1.126596</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PH070000000</td>
      <td>BOPHA</td>
      <td>-28.098951</td>
      <td>122.750504</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PH160000000</td>
      <td>BOPHA</td>
      <td>586.353550</td>
      <td>137.971322</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PH030000000</td>
      <td>BOPHA</td>
      <td>-0.014199</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PH060000000</td>
      <td>BOPHA</td>
      <td>-11.593242</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PH050000000</td>
      <td>BOPHA</td>
      <td>0.000000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PH150000000</td>
      <td>BOPHA</td>
      <td>-20.503124</td>
      <td>95.444403</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PH140000000</td>
      <td>BOPHA</td>
      <td>-0.019263</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PH050000000</td>
      <td>HAIMA</td>
      <td>-3.443930</td>
      <td>184.369567</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PH140000000</td>
      <td>HAIMA</td>
      <td>22.277784</td>
      <td>22.540475</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PH040000000</td>
      <td>HAIMA</td>
      <td>-1.023477</td>
      <td>191.723781</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PH010000000</td>
      <td>HAIMA</td>
      <td>32.958484</td>
      <td>64.260414</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PH030000000</td>
      <td>HAIMA</td>
      <td>-1.802512</td>
      <td>96.955580</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PH020000000</td>
      <td>HAIMA</td>
      <td>203.559265</td>
      <td>74.266467</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PH140000000</td>
      <td>KETSANA</td>
      <td>-1.096554</td>
      <td>199.435792</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PH130000000</td>
      <td>KETSANA</td>
      <td>82.130520</td>
      <td>117.125003</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PH010000000</td>
      <td>KETSANA</td>
      <td>-6.110583</td>
      <td>196.565413</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PH040000000</td>
      <td>KETSANA</td>
      <td>21.003093</td>
      <td>27.414090</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PH080000000</td>
      <td>KETSANA</td>
      <td>-0.407757</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>PH170000000</td>
      <td>KETSANA</td>
      <td>-15.876545</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>PH050000000</td>
      <td>KETSANA</td>
      <td>-22.688620</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PH030000000</td>
      <td>KETSANA</td>
      <td>2.388462</td>
      <td>16.015704</td>
    </tr>
    <tr>
      <th>29</th>
      <td>PH020000000</td>
      <td>KETSANA</td>
      <td>-5.285384</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PH140000000</td>
      <td>MANGKHUT</td>
      <td>-21.306796</td>
      <td>29.056292</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PH020000000</td>
      <td>MANGKHUT</td>
      <td>-113.472285</td>
      <td>41.736938</td>
    </tr>
    <tr>
      <th>32</th>
      <td>PH010000000</td>
      <td>MANGKHUT</td>
      <td>-12.283663</td>
      <td>25.323341</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PH030000000</td>
      <td>MANGKHUT</td>
      <td>-1.000464</td>
      <td>43.688339</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PH140000000</td>
      <td>SAUDEL</td>
      <td>-5.174829</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PH050000000</td>
      <td>SAUDEL</td>
      <td>-10.582172</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>PH080000000</td>
      <td>SAUDEL</td>
      <td>-1.917907</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>PH040000000</td>
      <td>SAUDEL</td>
      <td>-6.272242</td>
      <td>196.657449</td>
    </tr>
    <tr>
      <th>38</th>
      <td>PH030000000</td>
      <td>SAUDEL</td>
      <td>-14.284173</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PH020000000</td>
      <td>SAUDEL</td>
      <td>-12.920680</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>PH010000000</td>
      <td>SAUDEL</td>
      <td>-5.452019</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>PH130000000</td>
      <td>SAUDEL</td>
      <td>-1.595789</td>
      <td>200.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>PH170000000</td>
      <td>SAUDEL</td>
      <td>-5.324003</td>
      <td>200.000000</td>
    </tr>
  </tbody>
</table>
</div>



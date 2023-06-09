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
      <td>22.580645</td>
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
      <td>2.639401</td>
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
      <td>2.639401</td>
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
      <td>2.639401</td>
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
      <td>2.639401</td>
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
<p>5 rows × 31 columns</p>
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
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
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
      <td>31.336503</td>
      <td>29.117802</td>
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
      <td>23.447758</td>
      <td>23.591571</td>
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
      <td>31.336503</td>
      <td>29.117802</td>
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
      <td>31.336503</td>
      <td>29.117802</td>
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
      <td>23.546053</td>
      <td>23.660429</td>
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
<p>5 rows × 30 columns</p>
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
dfa = np.split(dfs[1], [27], axis=1)
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
      <th>...</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
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
      <td>0.314832</td>
      <td>0.337448</td>
      <td>0.008392</td>
      <td>0.015702</td>
      <td>...</td>
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
    </tr>
    <tr>
      <th>1</th>
      <td>0.154462</td>
      <td>0.949392</td>
      <td>0.014378</td>
      <td>0.010594</td>
      <td>9.712553e-05</td>
      <td>0.504983</td>
      <td>0.233919</td>
      <td>0.273404</td>
      <td>0.007450</td>
      <td>0.035236</td>
      <td>...</td>
      <td>0.440663</td>
      <td>0.304474</td>
      <td>0.043049</td>
      <td>0.360224</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.140000</td>
      <td>0.860000</td>
      <td>0.000086</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.176799</td>
      <td>0.839346</td>
      <td>0.011082</td>
      <td>0.013105</td>
      <td>1.371717e-05</td>
      <td>0.155316</td>
      <td>0.314832</td>
      <td>0.337448</td>
      <td>0.008392</td>
      <td>0.015702</td>
      <td>...</td>
      <td>0.670175</td>
      <td>0.649623</td>
      <td>0.057690</td>
      <td>0.393828</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.110000</td>
      <td>0.890000</td>
      <td>0.000139</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.169135</td>
      <td>0.874636</td>
      <td>0.008788</td>
      <td>0.010400</td>
      <td>1.544535e-04</td>
      <td>0.324958</td>
      <td>0.314832</td>
      <td>0.337448</td>
      <td>0.008392</td>
      <td>0.015702</td>
      <td>...</td>
      <td>0.383667</td>
      <td>0.380232</td>
      <td>0.036547</td>
      <td>0.317867</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.120000</td>
      <td>0.880000</td>
      <td>0.000654</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.161895</td>
      <td>0.909926</td>
      <td>0.009111</td>
      <td>0.008968</td>
      <td>1.690249e-03</td>
      <td>0.294989</td>
      <td>0.234927</td>
      <td>0.274202</td>
      <td>0.007462</td>
      <td>0.034992</td>
      <td>...</td>
      <td>0.421247</td>
      <td>0.310462</td>
      <td>0.062176</td>
      <td>0.515864</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.460000</td>
      <td>0.470000</td>
      <td>0.003618</td>
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
      <th>49749</th>
      <td>0.108159</td>
      <td>0.816770</td>
      <td>0.034958</td>
      <td>0.032650</td>
      <td>7.311641e-05</td>
      <td>0.111296</td>
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.331492</td>
      <td>0.433746</td>
      <td>0.026779</td>
      <td>0.011391</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.990000</td>
      <td>0.000103</td>
      <td>0.009097</td>
    </tr>
    <tr>
      <th>49750</th>
      <td>0.102816</td>
      <td>0.852281</td>
      <td>0.033634</td>
      <td>0.028851</td>
      <td>2.854586e-03</td>
      <td>0.379331</td>
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.129081</td>
      <td>0.208269</td>
      <td>0.019943</td>
      <td>0.271370</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.190000</td>
      <td>0.760000</td>
      <td>0.001682</td>
      <td>0.009097</td>
    </tr>
    <tr>
      <th>49751</th>
      <td>0.096754</td>
      <td>0.887792</td>
      <td>0.029724</td>
      <td>0.025744</td>
      <td>1.096340e-03</td>
      <td>0.498339</td>
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.138776</td>
      <td>0.204722</td>
      <td>0.016272</td>
      <td>0.087050</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.970000</td>
      <td>0.001673</td>
      <td>0.009097</td>
    </tr>
    <tr>
      <th>49752</th>
      <td>0.092212</td>
      <td>0.923300</td>
      <td>0.058092</td>
      <td>0.035882</td>
      <td>3.178534e-05</td>
      <td>0.286545</td>
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.156689</td>
      <td>0.151533</td>
      <td>0.015221</td>
      <td>0.031742</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.027273</td>
      <td>0.972727</td>
      <td>0.000284</td>
      <td>0.009097</td>
    </tr>
    <tr>
      <th>49753</th>
      <td>0.092815</td>
      <td>0.923297</td>
      <td>0.064748</td>
      <td>0.038826</td>
      <td>2.127484e-04</td>
      <td>0.315199</td>
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.188511</td>
      <td>0.254262</td>
      <td>0.025869</td>
      <td>0.057267</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.980000</td>
      <td>0.000456</td>
      <td>0.009097</td>
    </tr>
  </tbody>
</table>
<p>49754 rows × 27 columns</p>
</div>



```python
# All df without target column
dfa[0]
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
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
      <th>strong_roof_salvage_wall</th>
      <th>light_roof_strong_wall</th>
      <th>...</th>
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>12.460039</td>
      <td>275.018491</td>
      <td>0.670833</td>
      <td>0.313021</td>
      <td>0.479848</td>
      <td>-0.213039</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>0.507132</td>
      <td>...</td>
      <td>74.625539</td>
      <td>34.62955</td>
      <td>42.21875</td>
      <td>5303.65949</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11.428974</td>
      <td>297.027578</td>
      <td>0.929167</td>
      <td>0.343229</td>
      <td>55.649739</td>
      <td>0.206</td>
      <td>23.447758</td>
      <td>23.591571</td>
      <td>0.037516</td>
      <td>1.137998</td>
      <td>...</td>
      <td>68.681417</td>
      <td>25.475388</td>
      <td>72.283154</td>
      <td>61015.543599</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.14</td>
      <td>0.86</td>
      <td>276.871504</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>0.716667</td>
      <td>0.424479</td>
      <td>8.157414</td>
      <td>-0.636</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>0.507132</td>
      <td>...</td>
      <td>104.453163</td>
      <td>54.353996</td>
      <td>102.215198</td>
      <td>66707.43807</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.11</td>
      <td>0.89</td>
      <td>448.539453</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>12.511864</td>
      <td>273.63933</td>
      <td>0.56875</td>
      <td>0.336979</td>
      <td>88.292015</td>
      <td>-0.2275</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>0.507132</td>
      <td>...</td>
      <td>59.798108</td>
      <td>31.814048</td>
      <td>58.988877</td>
      <td>53841.050168</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>2101.708435</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>0.589583</td>
      <td>0.290625</td>
      <td>962.766739</td>
      <td>-0.299667</td>
      <td>23.546053</td>
      <td>23.660429</td>
      <td>0.037576</td>
      <td>1.130137</td>
      <td>...</td>
      <td>65.65528</td>
      <td>25.976413</td>
      <td>111.386527</td>
      <td>87378.257957</td>
      <td>1</td>
      <td>0.07</td>
      <td>0.46</td>
      <td>0.47</td>
      <td>11632.726327</td>
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
      <td>8.011792</td>
      <td>255.535258</td>
      <td>2.25625</td>
      <td>1.056771</td>
      <td>41.979062</td>
      <td>-0.742</td>
      <td>34.903986</td>
      <td>60.634178</td>
      <td>1.444247</td>
      <td>0.177505</td>
      <td>...</td>
      <td>51.666178</td>
      <td>36.291573</td>
      <td>39.018519</td>
      <td>1929.419748</td>
      <td>1</td>
      <td>0.01</td>
      <td>0.0</td>
      <td>0.99</td>
      <td>330.215768</td>
      <td>1.143833</td>
    </tr>
    <tr>
      <th>49750</th>
      <td>7.61746</td>
      <td>266.645258</td>
      <td>2.170833</td>
      <td>0.933854</td>
      <td>1625.734579</td>
      <td>-0.096571</td>
      <td>34.903986</td>
      <td>60.634178</td>
      <td>1.444247</td>
      <td>0.177505</td>
      <td>...</td>
      <td>20.11842</td>
      <td>17.425889</td>
      <td>25.042969</td>
      <td>45965.284119</td>
      <td>1</td>
      <td>0.05</td>
      <td>0.19</td>
      <td>0.76</td>
      <td>5409.607943</td>
      <td>1.143833</td>
    </tr>
    <tr>
      <th>49751</th>
      <td>7.170117</td>
      <td>277.755258</td>
      <td>1.91875</td>
      <td>0.833333</td>
      <td>624.597557</td>
      <td>0.19</td>
      <td>34.903986</td>
      <td>60.634178</td>
      <td>1.444247</td>
      <td>0.177505</td>
      <td>...</td>
      <td>21.62959</td>
      <td>17.129093</td>
      <td>17.537129</td>
      <td>14744.712453</td>
      <td>1</td>
      <td>0.01</td>
      <td>0.02</td>
      <td>0.97</td>
      <td>5378.401365</td>
      <td>1.143833</td>
    </tr>
    <tr>
      <th>49752</th>
      <td>6.834925</td>
      <td>288.864374</td>
      <td>3.747917</td>
      <td>1.16131</td>
      <td>18.445345</td>
      <td>-0.32</td>
      <td>34.903986</td>
      <td>60.634178</td>
      <td>1.444247</td>
      <td>0.177505</td>
      <td>...</td>
      <td>24.42143</td>
      <td>12.678785</td>
      <td>15.389474</td>
      <td>5376.583753</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.027273</td>
      <td>0.972727</td>
      <td>914.677196</td>
      <td>1.143833</td>
    </tr>
    <tr>
      <th>49753</th>
      <td>6.879427</td>
      <td>288.863491</td>
      <td>4.177083</td>
      <td>1.256548</td>
      <td>121.484861</td>
      <td>-0.251</td>
      <td>34.903986</td>
      <td>60.634178</td>
      <td>1.444247</td>
      <td>0.177505</td>
      <td>...</td>
      <td>29.381264</td>
      <td>21.274151</td>
      <td>37.157791</td>
      <td>9700.043352</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.02</td>
      <td>0.98</td>
      <td>1466.117288</td>
      <td>1.143833</td>
    </tr>
  </tbody>
</table>
<p>49754 rows × 27 columns</p>
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
      <th>...</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
      <th>26</th>
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
      <td>0.314832</td>
      <td>0.337448</td>
      <td>0.008392</td>
      <td>0.015702</td>
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
      <td>0.154462</td>
      <td>0.949392</td>
      <td>0.014378</td>
      <td>0.010594</td>
      <td>9.712553e-05</td>
      <td>0.504983</td>
      <td>0.233919</td>
      <td>0.273404</td>
      <td>0.007450</td>
      <td>0.035236</td>
      <td>...</td>
      <td>0.304474</td>
      <td>0.043049</td>
      <td>0.360224</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.140000</td>
      <td>0.860000</td>
      <td>0.000086</td>
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
      <td>0.314832</td>
      <td>0.337448</td>
      <td>0.008392</td>
      <td>0.015702</td>
      <td>...</td>
      <td>0.649623</td>
      <td>0.057690</td>
      <td>0.393828</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.110000</td>
      <td>0.890000</td>
      <td>0.000139</td>
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
      <td>0.314832</td>
      <td>0.337448</td>
      <td>0.008392</td>
      <td>0.015702</td>
      <td>...</td>
      <td>0.380232</td>
      <td>0.036547</td>
      <td>0.317867</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.120000</td>
      <td>0.880000</td>
      <td>0.000654</td>
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
      <td>0.234927</td>
      <td>0.274202</td>
      <td>0.007462</td>
      <td>0.034992</td>
      <td>...</td>
      <td>0.310462</td>
      <td>0.062176</td>
      <td>0.515864</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.460000</td>
      <td>0.470000</td>
      <td>0.003618</td>
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
      <td>0.108159</td>
      <td>0.816770</td>
      <td>0.034958</td>
      <td>0.032650</td>
      <td>7.311641e-05</td>
      <td>0.111296</td>
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.433746</td>
      <td>0.026779</td>
      <td>0.011391</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.990000</td>
      <td>0.000103</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.208269</td>
      <td>0.019943</td>
      <td>0.271370</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.190000</td>
      <td>0.760000</td>
      <td>0.001682</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.204722</td>
      <td>0.016272</td>
      <td>0.087050</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.970000</td>
      <td>0.001673</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.151533</td>
      <td>0.015221</td>
      <td>0.031742</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.027273</td>
      <td>0.972727</td>
      <td>0.000284</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>0.286799</td>
      <td>0.005496</td>
      <td>...</td>
      <td>0.254262</td>
      <td>0.025869</td>
      <td>0.057267</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.980000</td>
      <td>0.000456</td>
      <td>0.009097</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>49754 rows × 28 columns</p>
</div>




```python
features = [
    "wind_speed",
    "track_distance",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "total_houses",
    "rwi",
    "strong_roof_strong_wall",
    "strong_roof_light_wall",
    "strong_roof_salvage_wall",
    "light_roof_strong_wall",
    "light_roof_light_wall",
    "light_roof_salvage_wall",
    "salvaged_roof_strong_wall",
    "salvaged_roof_light_wall",
    "salvaged_roof_salvage_wall",
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
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
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
      <td>0.314832</td>
      <td>0.337448</td>
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
      <td>0.233919</td>
      <td>0.273404</td>
      <td>...</td>
      <td>0.304474</td>
      <td>0.043049</td>
      <td>0.360224</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.140000</td>
      <td>0.860000</td>
      <td>0.000086</td>
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
      <td>0.314832</td>
      <td>0.337448</td>
      <td>...</td>
      <td>0.649623</td>
      <td>0.057690</td>
      <td>0.393828</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.110000</td>
      <td>0.890000</td>
      <td>0.000139</td>
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
      <td>0.314832</td>
      <td>0.337448</td>
      <td>...</td>
      <td>0.380232</td>
      <td>0.036547</td>
      <td>0.317867</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.120000</td>
      <td>0.880000</td>
      <td>0.000654</td>
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
      <td>0.234927</td>
      <td>0.274202</td>
      <td>...</td>
      <td>0.310462</td>
      <td>0.062176</td>
      <td>0.515864</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.460000</td>
      <td>0.470000</td>
      <td>0.003618</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.433746</td>
      <td>0.026779</td>
      <td>0.011391</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.990000</td>
      <td>0.000103</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.208269</td>
      <td>0.019943</td>
      <td>0.271370</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.190000</td>
      <td>0.760000</td>
      <td>0.001682</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.204722</td>
      <td>0.016272</td>
      <td>0.087050</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.970000</td>
      <td>0.001673</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.151533</td>
      <td>0.015221</td>
      <td>0.031742</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.027273</td>
      <td>0.972727</td>
      <td>0.000284</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.254262</td>
      <td>0.025869</td>
      <td>0.057267</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.980000</td>
      <td>0.000456</td>
      <td>0.009097</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>49754 rows × 30 columns</p>
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
        "rainfall_max_6h",
        "rainfall_max_24h",
        "total_houses",
        "rwi",
        "strong_roof_strong_wall",
        "strong_roof_light_wall",
        "strong_roof_salvage_wall",
        "light_roof_strong_wall",
        "light_roof_light_wall",
        "light_roof_salvage_wall",
        "salvaged_roof_strong_wall",
        "salvaged_roof_light_wall",
        "salvaged_roof_salvage_wall",
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

    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_30127/354225951.py:38: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[3]])
    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_30127/354225951.py:39: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[2]])
    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_30127/354225951.py:40: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
      df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[1]])
    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_30127/354225951.py:41: FutureWarning: The frame.append method is deprecated and will be removed from pandas in a future version. Use pandas.concat instead.
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
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
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
      <td>0.657219</td>
      <td>0.284772</td>
      <td>...</td>
      <td>0.065630</td>
      <td>0.020180</td>
      <td>0.031687</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.92</td>
      <td>0.001211</td>
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
      <td>0.657960</td>
      <td>0.278739</td>
      <td>...</td>
      <td>0.141129</td>
      <td>0.060336</td>
      <td>0.091281</td>
      <td>1.0</td>
      <td>0.10</td>
      <td>0.60</td>
      <td>0.30</td>
      <td>0.004117</td>
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
      <td>0.646976</td>
      <td>0.266692</td>
      <td>...</td>
      <td>0.214061</td>
      <td>0.058423</td>
      <td>0.104104</td>
      <td>1.0</td>
      <td>0.28</td>
      <td>0.59</td>
      <td>0.13</td>
      <td>0.006658</td>
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
      <td>0.605574</td>
      <td>0.265275</td>
      <td>...</td>
      <td>0.184985</td>
      <td>0.033297</td>
      <td>0.091221</td>
      <td>1.0</td>
      <td>0.24</td>
      <td>0.62</td>
      <td>0.14</td>
      <td>0.008454</td>
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
      <td>0.637688</td>
      <td>0.231547</td>
      <td>...</td>
      <td>0.103623</td>
      <td>0.028936</td>
      <td>0.140126</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.89</td>
      <td>0.11</td>
      <td>0.002965</td>
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
      <td>0.292576</td>
      <td>0.487645</td>
      <td>...</td>
      <td>0.158586</td>
      <td>0.015329</td>
      <td>0.265042</td>
      <td>1.0</td>
      <td>0.12</td>
      <td>0.16</td>
      <td>0.72</td>
      <td>0.003767</td>
      <td>0.009013</td>
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
      <td>0.325208</td>
      <td>0.513887</td>
      <td>...</td>
      <td>0.095746</td>
      <td>0.012607</td>
      <td>0.089733</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.80</td>
      <td>0.007193</td>
      <td>0.007335</td>
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
      <td>0.473382</td>
      <td>0.475945</td>
      <td>...</td>
      <td>0.088262</td>
      <td>0.013689</td>
      <td>0.077939</td>
      <td>1.0</td>
      <td>0.03</td>
      <td>0.07</td>
      <td>0.90</td>
      <td>0.000113</td>
      <td>0.011737</td>
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
      <td>0.293731</td>
      <td>0.446105</td>
      <td>...</td>
      <td>0.117307</td>
      <td>0.016159</td>
      <td>0.095031</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>0.000749</td>
      <td>0.010422</td>
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
      <td>0.291560</td>
      <td>0.524191</td>
      <td>...</td>
      <td>0.084217</td>
      <td>0.013664</td>
      <td>0.064003</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>0.000855</td>
      <td>0.007773</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>6111 rows × 30 columns</p>
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
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
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
      <td>0.314832</td>
      <td>0.337448</td>
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
      <td>0.233919</td>
      <td>0.273404</td>
      <td>...</td>
      <td>0.304474</td>
      <td>0.043049</td>
      <td>0.360224</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.140000</td>
      <td>0.860000</td>
      <td>0.000086</td>
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
      <td>0.314832</td>
      <td>0.337448</td>
      <td>...</td>
      <td>0.649623</td>
      <td>0.057690</td>
      <td>0.393828</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.110000</td>
      <td>0.890000</td>
      <td>0.000139</td>
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
      <td>0.314832</td>
      <td>0.337448</td>
      <td>...</td>
      <td>0.380232</td>
      <td>0.036547</td>
      <td>0.317867</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.120000</td>
      <td>0.880000</td>
      <td>0.000654</td>
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
      <td>0.234927</td>
      <td>0.274202</td>
      <td>...</td>
      <td>0.310462</td>
      <td>0.062176</td>
      <td>0.515864</td>
      <td>1.0</td>
      <td>0.07</td>
      <td>0.460000</td>
      <td>0.470000</td>
      <td>0.003618</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.433746</td>
      <td>0.026779</td>
      <td>0.011391</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.990000</td>
      <td>0.000103</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.208269</td>
      <td>0.019943</td>
      <td>0.271370</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.190000</td>
      <td>0.760000</td>
      <td>0.001682</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.204722</td>
      <td>0.016272</td>
      <td>0.087050</td>
      <td>1.0</td>
      <td>0.01</td>
      <td>0.020000</td>
      <td>0.970000</td>
      <td>0.001673</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.151533</td>
      <td>0.015221</td>
      <td>0.031742</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.027273</td>
      <td>0.972727</td>
      <td>0.000284</td>
      <td>0.009097</td>
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
      <td>0.351423</td>
      <td>0.702693</td>
      <td>...</td>
      <td>0.254262</td>
      <td>0.025869</td>
      <td>0.057267</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.020000</td>
      <td>0.980000</td>
      <td>0.000456</td>
      <td>0.009097</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>43643 rows × 30 columns</p>
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


    [15:36:29] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.229
    Model:                                OLS   Adj. R-squared:                  0.228
    Method:                     Least Squares   F-statistic:                     479.1
    Date:                    Fri, 09 Jun 2023   Prob (F-statistic):               0.00
    Time:                            15:36:34   Log-Likelihood:            -1.2583e+05
    No. Observations:                   43643   AIC:                         2.517e+05
    Df Residuals:                       43615   BIC:                         2.520e+05
    Df Model:                              27                                         
    Covariance Type:                nonrobust                                         
    =================================================================================================
                                        coef    std err          t      P>|t|      [0.025      0.975]
    -------------------------------------------------------------------------------------------------
    const                         -4.905e+10   3.97e+11     -0.124      0.902   -8.27e+11    7.29e+11
    wind_speed                       17.4519      0.193     90.623      0.000      17.074      17.829
    track_distance                    3.4347      0.113     30.286      0.000       3.212       3.657
    rainfall_max_6h                   3.6435      0.559      6.514      0.000       2.547       4.740
    rainfall_max_24h                 -4.2706      0.546     -7.817      0.000      -5.341      -3.200
    total_houses                     -0.1335      1.553     -0.086      0.932      -3.178       2.911
    rwi                              -0.0095      0.256     -0.037      0.971      -0.511       0.492
    strong_roof_strong_wall           7.0063      3.442      2.035      0.042       0.259      13.753
    strong_roof_light_wall            8.4713      3.055      2.773      0.006       2.483      14.460
    strong_roof_salvage_wall          5.9594      0.404     14.736      0.000       5.167       6.752
    light_roof_strong_wall            4.1391      1.158      3.574      0.000       1.869       6.409
    light_roof_light_wall             6.9589      2.968      2.345      0.019       1.142      12.776
    light_roof_salvage_wall          -1.5073      0.875     -1.722      0.085      -3.223       0.209
    salvaged_roof_strong_wall         0.9716      0.565      1.718      0.086      -0.137       2.080
    salvaged_roof_light_wall         -0.4198      0.863     -0.486      0.627      -2.112       1.272
    salvaged_roof_salvage_wall        0.0442      0.391      0.113      0.910      -0.722       0.810
    mean_slope                       -0.9250      2.046     -0.452      0.651      -4.935       3.085
    std_slope                        -1.5320      1.024     -1.496      0.135      -3.539       0.475
    mean_tri                          0.7997      2.122      0.377      0.706      -3.360       4.959
    std_tri                           2.5473      1.194      2.134      0.033       0.207       4.887
    mean_elev                        -0.3685      0.287     -1.282      0.200      -0.932       0.195
    coast_length                      0.6033      0.256      2.359      0.018       0.102       1.105
    with_coast                       -0.0761      0.074     -1.025      0.305      -0.222       0.069
    urban                          4.905e+10   3.97e+11      0.124      0.902   -7.29e+11    8.27e+11
    rural                          4.905e+10   3.97e+11      0.124      0.902   -7.29e+11    8.27e+11
    water                          4.905e+10   3.97e+11      0.124      0.902   -7.29e+11    8.27e+11
    total_pop                        -0.2064      1.706     -0.121      0.904      -3.549       3.136
    percent_houses_damaged_5years    -3.4584      2.176     -1.589      0.112      -7.724       0.807
    ==============================================================================
    Omnibus:                    61282.876   Durbin-Watson:                   0.648
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         20701380.945
    Skew:                           8.299   Prob(JB):                         0.00
    Kurtosis:                     108.397   Cond. No.                     6.66e+13
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The smallest eigenvalue is 2.97e-23. This might indicate that there are
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
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
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
      <td>0.657219</td>
      <td>0.284772</td>
      <td>...</td>
      <td>0.020180</td>
      <td>0.031687</td>
      <td>1.0</td>
      <td>0.05</td>
      <td>0.03</td>
      <td>0.92</td>
      <td>0.001211</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>0.657960</td>
      <td>0.278739</td>
      <td>...</td>
      <td>0.060336</td>
      <td>0.091281</td>
      <td>1.0</td>
      <td>0.10</td>
      <td>0.60</td>
      <td>0.30</td>
      <td>0.004117</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.024520</td>
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
      <td>0.646976</td>
      <td>0.266692</td>
      <td>...</td>
      <td>0.058423</td>
      <td>0.104104</td>
      <td>1.0</td>
      <td>0.28</td>
      <td>0.59</td>
      <td>0.13</td>
      <td>0.006658</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.042715</td>
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
      <td>0.605574</td>
      <td>0.265275</td>
      <td>...</td>
      <td>0.033297</td>
      <td>0.091221</td>
      <td>1.0</td>
      <td>0.24</td>
      <td>0.62</td>
      <td>0.14</td>
      <td>0.008454</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.209506</td>
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
      <td>0.637688</td>
      <td>0.231547</td>
      <td>...</td>
      <td>0.028936</td>
      <td>0.140126</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.89</td>
      <td>0.11</td>
      <td>0.002965</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.029664</td>
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
      <td>0.292576</td>
      <td>0.487645</td>
      <td>...</td>
      <td>0.015329</td>
      <td>0.265042</td>
      <td>1.0</td>
      <td>0.12</td>
      <td>0.16</td>
      <td>0.72</td>
      <td>0.003767</td>
      <td>0.009013</td>
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
      <td>0.325208</td>
      <td>0.513887</td>
      <td>...</td>
      <td>0.012607</td>
      <td>0.089733</td>
      <td>1.0</td>
      <td>0.14</td>
      <td>0.06</td>
      <td>0.80</td>
      <td>0.007193</td>
      <td>0.007335</td>
      <td>0.0</td>
      <td>0.032423</td>
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
      <td>0.473382</td>
      <td>0.475945</td>
      <td>...</td>
      <td>0.013689</td>
      <td>0.077939</td>
      <td>1.0</td>
      <td>0.03</td>
      <td>0.07</td>
      <td>0.90</td>
      <td>0.000113</td>
      <td>0.011737</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>0.293731</td>
      <td>0.446105</td>
      <td>...</td>
      <td>0.016159</td>
      <td>0.095031</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>0.000749</td>
      <td>0.010422</td>
      <td>0.0</td>
      <td>0.000000</td>
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
      <td>0.291560</td>
      <td>0.524191</td>
      <td>...</td>
      <td>0.013664</td>
      <td>0.064003</td>
      <td>1.0</td>
      <td>0.00</td>
      <td>0.03</td>
      <td>0.97</td>
      <td>0.000855</td>
      <td>0.007773</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>6111 rows × 31 columns</p>
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
```


```python
# join main df to the weight df based on grid_point_id
join_final = df_test.merge(df_weight, on="grid_point_id", how="left")
```


```python
# Remove all columns between column index 21 to 25
join_final.drop(join_final.iloc[:, 23:27], inplace=True, axis=1)
```


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
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
      <th>...</th>
      <th>y_pred</th>
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
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
      <td>0.657219</td>
      <td>0.284772</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH015514000</td>
      <td>9233.0</td>
      <td>119.8E_16.4N</td>
      <td>689</td>
      <td>689</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000008</td>
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
      <td>0.657960</td>
      <td>0.278739</td>
      <td>...</td>
      <td>0.024520</td>
      <td>PH015508000</td>
      <td>9234.0</td>
      <td>119.8E_16.3N</td>
      <td>1844</td>
      <td>5089</td>
      <td>0.362350</td>
      <td>0.0</td>
      <td>4.755256e-07</td>
      <td>0.000019</td>
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
      <td>0.657960</td>
      <td>0.278739</td>
      <td>...</td>
      <td>0.024520</td>
      <td>PH015514000</td>
      <td>9234.0</td>
      <td>119.8E_16.3N</td>
      <td>3245</td>
      <td>5089</td>
      <td>0.637650</td>
      <td>0.0</td>
      <td>8.368116e-07</td>
      <td>0.000034</td>
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
      <td>0.646976</td>
      <td>0.266692</td>
      <td>...</td>
      <td>0.042715</td>
      <td>PH015501000</td>
      <td>9235.0</td>
      <td>119.8E_16.2N</td>
      <td>1351</td>
      <td>6106</td>
      <td>0.221258</td>
      <td>0.0</td>
      <td>5.674695e-07</td>
      <td>0.000013</td>
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
      <td>0.646976</td>
      <td>0.266692</td>
      <td>...</td>
      <td>0.042715</td>
      <td>PH015508000</td>
      <td>9235.0</td>
      <td>119.8E_16.2N</td>
      <td>4755</td>
      <td>6106</td>
      <td>0.778742</td>
      <td>0.0</td>
      <td>1.997274e-06</td>
      <td>0.000047</td>
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
      <td>0.325208</td>
      <td>0.513887</td>
      <td>...</td>
      <td>0.032423</td>
      <td>PH082606000</td>
      <td>18796.0</td>
      <td>125.5E_12.0N</td>
      <td>4070</td>
      <td>5944</td>
      <td>0.684724</td>
      <td>0.0</td>
      <td>1.909853e-06</td>
      <td>0.000059</td>
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
      <td>0.473382</td>
      <td>0.475945</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH082622000</td>
      <td>18797.0</td>
      <td>125.5E_11.9N</td>
      <td>463</td>
      <td>463</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000006</td>
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
      <td>0.293731</td>
      <td>0.446105</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>18962.0</td>
      <td>125.6E_12.1N</td>
      <td>77</td>
      <td>461</td>
      <td>0.167028</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000002</td>
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
      <td>0.293731</td>
      <td>0.446105</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH082617000</td>
      <td>18962.0</td>
      <td>125.6E_12.1N</td>
      <td>384</td>
      <td>461</td>
      <td>0.832972</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000008</td>
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
      <td>0.291560</td>
      <td>0.524191</td>
      <td>...</td>
      <td>0.000000</td>
      <td>PH082606000</td>
      <td>18963.0</td>
      <td>125.6E_12.0N</td>
      <td>731</td>
      <td>731</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000011</td>
    </tr>
  </tbody>
</table>
<p>18342 rows × 36 columns</p>
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
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
      <th>...</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>numbuildings</th>
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
      <td>0.657219</td>
      <td>0.284772</td>
      <td>...</td>
      <td>9233.0</td>
      <td>119.8E_16.4N</td>
      <td>689</td>
      <td>689</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000008</td>
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
      <td>0.657960</td>
      <td>0.278739</td>
      <td>...</td>
      <td>9234.0</td>
      <td>119.8E_16.3N</td>
      <td>1844</td>
      <td>5089</td>
      <td>0.362350</td>
      <td>0.0</td>
      <td>4.755256e-07</td>
      <td>0.000019</td>
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
      <td>0.657960</td>
      <td>0.278739</td>
      <td>...</td>
      <td>9234.0</td>
      <td>119.8E_16.3N</td>
      <td>3245</td>
      <td>5089</td>
      <td>0.637650</td>
      <td>0.0</td>
      <td>8.368116e-07</td>
      <td>0.000034</td>
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
      <td>0.646976</td>
      <td>0.266692</td>
      <td>...</td>
      <td>9235.0</td>
      <td>119.8E_16.2N</td>
      <td>1351</td>
      <td>6106</td>
      <td>0.221258</td>
      <td>0.0</td>
      <td>5.674695e-07</td>
      <td>0.000013</td>
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
      <td>0.646976</td>
      <td>0.266692</td>
      <td>...</td>
      <td>9235.0</td>
      <td>119.8E_16.2N</td>
      <td>4755</td>
      <td>6106</td>
      <td>0.778742</td>
      <td>0.0</td>
      <td>1.997274e-06</td>
      <td>0.000047</td>
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
      <td>0.325208</td>
      <td>0.513887</td>
      <td>...</td>
      <td>18796.0</td>
      <td>125.5E_12.0N</td>
      <td>4070</td>
      <td>5944</td>
      <td>0.684724</td>
      <td>0.0</td>
      <td>1.909853e-06</td>
      <td>0.000059</td>
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
      <td>0.473382</td>
      <td>0.475945</td>
      <td>...</td>
      <td>18797.0</td>
      <td>125.5E_11.9N</td>
      <td>463</td>
      <td>463</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000006</td>
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
      <td>0.293731</td>
      <td>0.446105</td>
      <td>...</td>
      <td>18962.0</td>
      <td>125.6E_12.1N</td>
      <td>77</td>
      <td>461</td>
      <td>0.167028</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000002</td>
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
      <td>0.293731</td>
      <td>0.446105</td>
      <td>...</td>
      <td>18962.0</td>
      <td>125.6E_12.1N</td>
      <td>384</td>
      <td>461</td>
      <td>0.832972</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000008</td>
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
      <td>0.291560</td>
      <td>0.524191</td>
      <td>...</td>
      <td>18963.0</td>
      <td>125.6E_12.0N</td>
      <td>731</td>
      <td>731</td>
      <td>1.000000</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000011</td>
      <td>Region VIII</td>
      <td>PH080000000</td>
    </tr>
  </tbody>
</table>
<p>18342 rows × 38 columns</p>
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
      <td>5.610564e-07</td>
      <td>1.571093</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>HAIMA</th>
      <td>0.000003</td>
      <td>1.302088e-05</td>
      <td>1.571093</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>MANGKHUT</th>
      <td>0.000008</td>
      <td>1.671826e-05</td>
      <td>1.571093</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>SAUDEL</th>
      <td>0.000000</td>
      <td>5.867593e-07</td>
      <td>1.571093</td>
      <td>0.000008</td>
    </tr>
    <tr>
      <th>PH012802000</th>
      <th>PH010000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.724799</td>
      <td>0.000135</td>
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
      <td>3.195157e-06</td>
      <td>4.000000</td>
      <td>0.000025</td>
    </tr>
    <tr>
      <th>PH175905000</th>
      <th>PH170000000</th>
      <th>KETSANA</th>
      <td>0.000000</td>
      <td>2.012662e-06</td>
      <td>1.000000</td>
      <td>0.000018</td>
    </tr>
    <tr>
      <th>PH175907000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>2.687087e-06</td>
      <td>0.340199</td>
      <td>0.000026</td>
    </tr>
    <tr>
      <th>PH175914000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>3.842557e-06</td>
      <td>2.551962</td>
      <td>0.000045</td>
    </tr>
    <tr>
      <th>PH175915000</th>
      <th>PH170000000</th>
      <th>BOPHA</th>
      <td>0.000000</td>
      <td>6.038499e-06</td>
      <td>1.692708</td>
      <td>0.000061</td>
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
```


```python
# Keep only %damg_normalized and %pred_damg_normalized columns
agg_df.drop(agg_df.columns[:4], inplace=True, axis=1)
```


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
      <td>0.000002</td>
      <td>0.000034</td>
      <td>0.025227</td>
    </tr>
    <tr>
      <th>HAIMA</th>
      <td>0.008584</td>
      <td>0.002912</td>
      <td>0.025226</td>
    </tr>
    <tr>
      <th>KETSANA</th>
      <td>0.000016</td>
      <td>0.001282</td>
      <td>0.025133</td>
    </tr>
    <tr>
      <th>MANGKHUT</th>
      <td>0.008383</td>
      <td>0.004394</td>
      <td>0.025226</td>
    </tr>
    <tr>
      <th>SAUDEL</th>
      <td>0.000000</td>
      <td>0.000824</td>
      <td>0.025226</td>
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
```


```python
# reset indexex
agg_df_2 = agg_df_1.reset_index()
```


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

agg_df_2["Percent_Difference_average_based"] = (difference / ave) * 100
```


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
      <td>-0.127543</td>
      <td>-174.982917</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH120000000</td>
      <td>BOPHA</td>
      <td>-1.253007</td>
      <td>-185.686083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH110000000</td>
      <td>BOPHA</td>
      <td>83.751252</td>
      <td>17.553016</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH100000000</td>
      <td>BOPHA</td>
      <td>-355.292351</td>
      <td>-185.758949</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH170000000</td>
      <td>BOPHA</td>
      <td>-22.150110</td>
      <td>-110.394098</td>
    </tr>
    <tr>
      <th>5</th>
      <td>PH020000000</td>
      <td>BOPHA</td>
      <td>-0.248335</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PH090000000</td>
      <td>BOPHA</td>
      <td>-69.823665</td>
      <td>-199.928579</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PH080000000</td>
      <td>BOPHA</td>
      <td>-4.429423</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PH070000000</td>
      <td>BOPHA</td>
      <td>-30.785225</td>
      <td>-151.953915</td>
    </tr>
    <tr>
      <th>9</th>
      <td>PH160000000</td>
      <td>BOPHA</td>
      <td>211.166748</td>
      <td>94.730554</td>
    </tr>
    <tr>
      <th>10</th>
      <td>PH030000000</td>
      <td>BOPHA</td>
      <td>-0.010275</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>PH060000000</td>
      <td>BOPHA</td>
      <td>-3.602340</td>
      <td>-200.000000</td>
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
      <td>-26.281640</td>
      <td>-132.842506</td>
    </tr>
    <tr>
      <th>14</th>
      <td>PH140000000</td>
      <td>BOPHA</td>
      <td>-0.088624</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PH050000000</td>
      <td>HAIMA</td>
      <td>-7.756523</td>
      <td>-199.992998</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PH140000000</td>
      <td>HAIMA</td>
      <td>46.361478</td>
      <td>98.185205</td>
    </tr>
    <tr>
      <th>17</th>
      <td>PH040000000</td>
      <td>HAIMA</td>
      <td>-4.691729</td>
      <td>-197.393125</td>
    </tr>
    <tr>
      <th>18</th>
      <td>PH010000000</td>
      <td>HAIMA</td>
      <td>22.482713</td>
      <td>98.672212</td>
    </tr>
    <tr>
      <th>19</th>
      <td>PH030000000</td>
      <td>HAIMA</td>
      <td>-0.015844</td>
      <td>-4.692472</td>
    </tr>
    <tr>
      <th>20</th>
      <td>PH020000000</td>
      <td>HAIMA</td>
      <td>211.102771</td>
      <td>127.918257</td>
    </tr>
    <tr>
      <th>21</th>
      <td>PH140000000</td>
      <td>KETSANA</td>
      <td>-0.226627</td>
      <td>-192.999069</td>
    </tr>
    <tr>
      <th>22</th>
      <td>PH130000000</td>
      <td>KETSANA</td>
      <td>39.896623</td>
      <td>107.273231</td>
    </tr>
    <tr>
      <th>23</th>
      <td>PH010000000</td>
      <td>KETSANA</td>
      <td>-5.035092</td>
      <td>-195.005553</td>
    </tr>
    <tr>
      <th>24</th>
      <td>PH040000000</td>
      <td>KETSANA</td>
      <td>22.893053</td>
      <td>84.540587</td>
    </tr>
    <tr>
      <th>25</th>
      <td>PH080000000</td>
      <td>KETSANA</td>
      <td>-5.557022</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>PH170000000</td>
      <td>KETSANA</td>
      <td>-30.417154</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>PH050000000</td>
      <td>KETSANA</td>
      <td>-10.650608</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>PH030000000</td>
      <td>KETSANA</td>
      <td>15.638458</td>
      <td>113.283910</td>
    </tr>
    <tr>
      <th>29</th>
      <td>PH020000000</td>
      <td>KETSANA</td>
      <td>-0.822697</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>PH140000000</td>
      <td>MANGKHUT</td>
      <td>10.636721</td>
      <td>22.309486</td>
    </tr>
    <tr>
      <th>31</th>
      <td>PH020000000</td>
      <td>MANGKHUT</td>
      <td>-2.555654</td>
      <td>-1.612966</td>
    </tr>
    <tr>
      <th>32</th>
      <td>PH010000000</td>
      <td>MANGKHUT</td>
      <td>15.814087</td>
      <td>62.445734</td>
    </tr>
    <tr>
      <th>33</th>
      <td>PH030000000</td>
      <td>MANGKHUT</td>
      <td>-0.278175</td>
      <td>-60.834791</td>
    </tr>
    <tr>
      <th>34</th>
      <td>PH140000000</td>
      <td>SAUDEL</td>
      <td>-1.100343</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>PH050000000</td>
      <td>SAUDEL</td>
      <td>-7.203795</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>36</th>
      <td>PH080000000</td>
      <td>SAUDEL</td>
      <td>-4.011936</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>37</th>
      <td>PH040000000</td>
      <td>SAUDEL</td>
      <td>-0.205057</td>
      <td>-193.972202</td>
    </tr>
    <tr>
      <th>38</th>
      <td>PH030000000</td>
      <td>SAUDEL</td>
      <td>-1.628723</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>39</th>
      <td>PH020000000</td>
      <td>SAUDEL</td>
      <td>-2.228053</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>40</th>
      <td>PH010000000</td>
      <td>SAUDEL</td>
      <td>-3.267107</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>41</th>
      <td>PH130000000</td>
      <td>SAUDEL</td>
      <td>-0.000004</td>
      <td>-200.000000</td>
    </tr>
    <tr>
      <th>42</th>
      <td>PH170000000</td>
      <td>SAUDEL</td>
      <td>-0.978456</td>
      <td>-200.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

# Naive baseline
#### municipality-based dataset


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

from math import sqrt
from collections import defaultdict
from sklearn import preprocessing
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
# Hist plot after data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["y_norm"], bins=bins2)
```


```python
print(samples_per_bin2)
print(binsP2)
```

    [3606 2874 1141  388   64]
    [0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]



```python
bin_index2 = np.digitize(df["y_norm"], bins=binsP2)
```


```python
y_input_strat = bin_index2
```


```python
# Split X and y from dataframe features

X = pd.Series([0] * 8073)
y = df["y_norm"]
```


```python
# Defin two lists to save RMSE and Average Error

RMSE = defaultdict(list)
AVE = defaultdict(list)
```


```python
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df["y_norm"],
        stratify=y_input_strat,
        test_size=0.2,
    )

    # create a dummy regressor
    dummy_reg = DummyRegressor(strategy="mean")

    # fit it on the training set
    dummy_reg.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = dummy_reg.predict(X_test)
    y_pred_clipped = y_pred.clip(0, 100)

    # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based
    rmse = sqrt(mean_squared_error(y_test, y_pred_clipped))
    ave = (y_pred_clipped - y_test).sum() / len(y_test)

    print(f"RMSE for mun_based model: {rmse:.2f}")
    print(f"Average Error for mun_based model: {ave:.2f}")

    RMSE["all"].append(rmse)
    AVE["all"].append(ave)

    bin_index = np.digitize(y_test, bins=binsP2)

    for bin_num in range(1, 6):

        mse_idx = mean_squared_error(
            y_test[bin_index == bin_num],
            y_pred_clipped[bin_index == bin_num],
        )
        rmse = np.sqrt(mse_idx)

        ave = (
            y_pred_clipped[bin_index == bin_num] - y_test[bin_index == bin_num]
        ).sum() / len(y_test[bin_index == bin_num])

        RMSE[bin_num].append(rmse)
        AVE[bin_num].append(ave)
```

    RMSE for mun_based model: 8.30
    Average Error for mun_based model: -0.12
    RMSE for mun_based model: 7.48
    Average Error for mun_based model: 0.14
    RMSE for mun_based model: 7.96
    Average Error for mun_based model: -0.08
    RMSE for mun_based model: 7.64
    Average Error for mun_based model: 0.01
    RMSE for mun_based model: 7.90
    Average Error for mun_based model: -0.01
    RMSE for mun_based model: 8.38
    Average Error for mun_based model: -0.05
    RMSE for mun_based model: 8.51
    Average Error for mun_based model: -0.09
    RMSE for mun_based model: 7.91
    Average Error for mun_based model: -0.08
    RMSE for mun_based model: 7.86
    Average Error for mun_based model: 0.04
    RMSE for mun_based model: 8.01
    Average Error for mun_based model: -0.03
    RMSE for mun_based model: 8.18
    Average Error for mun_based model: -0.08
    RMSE for mun_based model: 8.24
    Average Error for mun_based model: -0.05
    RMSE for mun_based model: 7.85
    Average Error for mun_based model: -0.00
    RMSE for mun_based model: 8.15
    Average Error for mun_based model: -0.12
    RMSE for mun_based model: 8.01
    Average Error for mun_based model: -0.08
    RMSE for mun_based model: 8.23
    Average Error for mun_based model: -0.11
    RMSE for mun_based model: 7.95
    Average Error for mun_based model: 0.10
    RMSE for mun_based model: 7.94
    Average Error for mun_based model: 0.01
    RMSE for mun_based model: 8.01
    Average Error for mun_based model: 0.01
    RMSE for mun_based model: 8.01
    Average Error for mun_based model: -0.01



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
    
    mean_RMSE: 8.03
    stdev_RMSE: 0.24
    mean_average_error: -0.03
    stdev_average_error: 0.07



```python
for bin_num in range(1, 6):

    print(f"\n RMSE and Average Error per bin {bin_num}\n")
    rmse_ave_mean(RMSE[bin_num], AVE[bin_num])
```

    
     RMSE and Average Error per bin 1
    
    mean_RMSE: 2.22
    stdev_RMSE: 0.01
    mean_average_error: 2.22
    stdev_average_error: 0.01
    
     RMSE and Average Error per bin 2
    
    mean_RMSE: 2.03
    stdev_RMSE: 0.02
    mean_average_error: 2.01
    stdev_average_error: 0.02
    
     RMSE and Average Error per bin 3
    
    mean_RMSE: 2.82
    stdev_RMSE: 0.16
    mean_average_error: -1.42
    stdev_average_error: 0.16
    
     RMSE and Average Error per bin 4
    
    mean_RMSE: 24.81
    stdev_RMSE: 1.15
    mean_average_error: -21.77
    stdev_average_error: 1.18
    
     RMSE and Average Error per bin 5
    
    mean_RMSE: 60.88
    stdev_RMSE: 3.39
    mean_average_error: -59.97
    stdev_average_error: 3.03



```python

```

# Typhoon split Naive baseline

#### We split based on typhoons


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
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sty import fg, rs
from itertools import chain

from utils import get_training_dataset, weight_file
```


```python
# Hide all warnings
warnings.filterwarnings("ignore")
```


```python
# Read CSV file and import to a df
df = get_training_dataset()
```


```python
for i in range(len(df)):
    df.at[i, "typhoon_year"] = str(df.loc[i]["typhoon_year"])
```


```python
df["typhoon_name"] = df["typhoon_name"] + df["typhoon_year"]
```


```python
# Replace empty cells of RWI with mean value
df["rwi"].fillna(df["rwi"].mean(), inplace=True)

# Set any values >100% to 100%,
for r in range(len(df)):
    if df.loc[r, "percent_houses_damaged"] > 100:
        df.at[r, "percent_houses_damaged"] = float(100)
```


```python
df = (df[(df[["wind_speed"]] != 0).any(axis=1)]).reset_index(drop=True)
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
      <th>0</th>
      <td>DURIAN2006</td>
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
      <th>1</th>
      <td>DURIAN2006</td>
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
      <th>2</th>
      <td>DURIAN2006</td>
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
      <th>3</th>
      <td>DURIAN2006</td>
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
      <th>4</th>
      <td>DURIAN2006</td>
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
      <th>0</th>
      <td>DURIAN2006</td>
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
      <th>1</th>
      <td>DURIAN2006</td>
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
      <th>2</th>
      <td>DURIAN2006</td>
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
      <th>3</th>
      <td>DURIAN2006</td>
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
      <th>4</th>
      <td>DURIAN2006</td>
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
# Read municipality dataset which already merged with y_norm converted ground truth
df_mun_merged = pd.read_csv("data/df_merged_2.csv")

# Remove the duplicated rows
df_mun_merged.drop_duplicates(keep="first", inplace=True)
df_mun_merged = df_mun_merged.reset_index(drop=True)

# Make the name of typhoons to uppercase
df_mun_merged["typhoon"] = df_mun_merged["typhoon"].str.upper()

# Rename y_norm column
df_mun_merged = df_mun_merged.rename(columns={"y_norm": "y_norm_mun"})

df_mun_merged
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
      <th>y_norm_mun</th>
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
      <th>7855</th>
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
      <th>7856</th>
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
      <th>7857</th>
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
      <th>7858</th>
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
      <th>7859</th>
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
<p>7860 rows × 40 columns</p>
</div>




```python
len(df_mun_merged['Mun_Code'].unique())
```




    1452




```python
# Function to transform name of typhoons to lowercase and remove begining of years
def transform_strings(strings):
    transformed_strings = []
    for string in strings:
        transformed_string = string[0].upper() + string[1:-4].lower() + string[-2:]
        transformed_strings.append(transformed_string)
    return transformed_strings
```


```python
# List of typhoons
typhoons = [
    "DURIAN2006",
    "FENGSHEN2008",
    "KETSANA2009",
    "CONSON2010",
    "NESAT2011",
    "BOPHA2012",
    "NARI2013",
    "KROSA2013",
    "HAIYAN2013",
    "USAGI2013",
    "UTOR2013",
    "JANGMI2014",
    "KALMAEGI2014",
    "RAMMASUN2014",
    "HAGUPIT2014",
    "FUNG-WONG2014",
    "LINGLING2014",
    "MUJIGAE2015",
    "MELOR2015",
    "NOUL2015",
    "GONI2015",
    "LINFA2015",
    "KOPPU2015",
    "MEKKHALA2015",
    "HAIMA2016",
    "TOKAGE2016",
    "MERANTI2016",
    "NOCK-TEN2016",
    "SARIKA2016",
    "MANGKHUT2018",
    "YUTU2018",
    "KAMMURI2019",
    "NAKRI2019",
    "PHANFONE2019",
    "SAUDEL2020",
    "GONI2020",
    "VAMCO2020",
    "VONGFONG2020",
    "MOLAVE2020",
]
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




```python
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df_data["percent_houses_damaged"], bins=bins2)
```


```python
# Define range of for loops
num_exp_main = 20

num_exp = 12
typhoons_for_test = typhoons[-num_exp:]

# All typhoons
# num_exp = len(typhoons)

# Define number of bins
num_bins = len(bins2)
```


```python
# Define empty list to save RMSE in each iteration
main_RMSE_lst = []
main_RMSE_bin = defaultdict(list)

main_AVE_lst = []
main_AVE_bin = defaultdict(list)
```


```python
for run_exm in range(num_exp_main):

    RMSE = defaultdict(list)
    AVE = defaultdict(list)

    # for run_ix in range(27, num_exp):
    for run_ix in range(num_exp):

        # WITHOUT removing old typhoons from training set
        typhoons_train_lst = typhoons[0 : run_ix + 27]

        # WITH removing old typhoons from training set
        # typhoons_train_lst = typhoons[run_ix : run_ix + 27]

        # In each run Keep one typhoon for the test list while the rest of the typhoons in the training set
        # typhoons_for_test = typhoons[run_ix]
        # typhoons_train_lst = typhoons[:run_ix] + typhoons[run_ix + 1 :]

        # print(typhoons_train_lst)

        bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
        y_input_strat = bin_index2

        # Split X and y from dataframe features
        X = pd.Series([0] * 49754)
        y = df_data["percent_houses_damaged"]

        # Split df to train and test (one typhoon for test and the rest of typhoons for train)
        # For when we train over all typhoon this df_test is required
        # df_test = df_data[df_data["typhoon_name"] == typhoons_for_test]

        df_test = df[df["typhoon_name"] == typhoons_for_test[run_ix]]

        df_train = pd.DataFrame()
        for run_ix_train in range(len(typhoons_train_lst)):
            df_train = df_train.append(
                df_data[df_data["typhoon_name"] == typhoons_train_lst[run_ix_train]]
            )

        # Split X and y from dataframe features
        # X_test = df_test["percent_houses_damaged"]
        # X_train = df_train["percent_houses_damaged"]

        X_test = pd.Series([0] * len(df_test))
        X_train = pd.Series([0] * len(df_train))

        y_train = df_train["percent_houses_damaged"]
        y_test = df_test["percent_houses_damaged"]

        print(df_test["typhoon_name"].unique())

        # create a dummy regressor
        dummy_reg = DummyRegressor(strategy="mean")

        # fit it on the training set
        dummy_reg.fit(X_train, y_train)

        # make predictions on the test set
        y_pred = dummy_reg.predict(X_test)

        # Define an empty data frame to insert real and prediction values
        pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])

        # Normal prediction of test data (in all the ranges values of target)
        pred_df["y_all"] = y_test
        pred_df["y_pred_all"] = y_pred

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
        agg_df = join_df.groupby(["ADM3_PCODE", "typhoon_name", "typhoon_year"]).agg(
            "sum"
        )

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
        final_df = final_df.reset_index()
        print(len(final_df))

        # Intersection of two datasets grid and municipality
        # Rename a column
        final_df = final_df.rename(
            columns={"ADM3_PCODE": "Mun_Code", "typhoon_name": "typhoon"}
        )

        # Merge DataFrames based on 'typhoon_name' and 'Mun_Code'
        merged_df = pd.merge(
            final_df, df_mun_merged, on=["Mun_Code", "typhoon"], how="inner"
        )
        print(len(merged_df))

        # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based

        if (len(merged_df["y_norm"])) != 0:
            rmse = sqrt(
                mean_squared_error(merged_df["y_norm"], merged_df["y_pred_norm"])
            )
            ave = (merged_df["y_pred_norm"] - merged_df["y_norm"]).sum() / len(
                merged_df["y_norm"]
            )

            print(f"RMSE for grid_based model: {rmse:.2f}")
            print(f"Average Error for grid_based model: {ave:.2f}")

            RMSE["all"].append(rmse)
            AVE["all"].append(ave)

        bin_index = np.digitize(merged_df["y_norm"], bins=binsP2)

        for bin_num in range(1, 6):
            if len(merged_df["y_norm"][bin_index == bin_num]) > 0:

                mse_idx = mean_squared_error(
                    merged_df["y_norm"][bin_index == bin_num],
                    merged_df["y_pred_norm"][bin_index == bin_num],
                )
                rmse = np.sqrt(mse_idx)

                ave = (
                    merged_df["y_pred_norm"][bin_index == bin_num]
                    - merged_df["y_norm"][bin_index == bin_num]
                ).sum() / len(merged_df["y_norm"][bin_index == bin_num])

                RMSE[bin_num].append(rmse)
                AVE[bin_num].append(ave)

        # Save total RMSE in each iteration

    main_RMSE_lst.append(RMSE["all"])
    main_AVE_lst.append(AVE["all"])

    for bin_num in range(1, 6):

        # Save RMSE per bin in each iteration
        main_RMSE_bin[bin_num].append(RMSE[bin_num])
        main_AVE_bin[bin_num].append(AVE[bin_num])
```

    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42
    ['NOCK-TEN2016']
    801
    432
    RMSE for grid_based model: 5.89
    Average Error for grid_based model: -0.84
    ['SARIKA2016']
    830
    66
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: 0.91
    ['MANGKHUT2018']
    345
    280
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.21
    ['YUTU2018']
    562
    173
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.55
    ['KAMMURI2019']
    821
    328
    RMSE for grid_based model: 4.43
    Average Error for grid_based model: -1.53
    ['NAKRI2019']
    6
    2
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: 0.93
    ['PHANFONE2019']
    922
    218
    RMSE for grid_based model: 5.00
    Average Error for grid_based model: -1.81
    ['SAUDEL2020']
    711
    2
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.92
    ['GONI2020']
    826
    229
    RMSE for grid_based model: 5.57
    Average Error for grid_based model: -1.50
    ['VAMCO2020']
    751
    296
    RMSE for grid_based model: 1.52
    Average Error for grid_based model: 0.25
    ['VONGFONG2020']
    1172
    313
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 0.15
    ['MOLAVE2020']
    737
    125
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: 0.42



```python
# Estimate total RMSE

rmse = statistics.mean(list(chain.from_iterable(main_RMSE_lst)))
print(f"RMSE: {rmse:.2f}")

sd_rmse = statistics.stdev(list(chain.from_iterable(main_RMSE_lst)))
print(f"stdev: {sd_rmse:.2f}")

ave = statistics.mean(list(chain.from_iterable(main_AVE_lst)))
print(f"Average Error: {ave:.2f}")

sd_ave = statistics.stdev(list(chain.from_iterable(main_AVE_lst)))
print(f"Stdev of Average Error: {sd_ave:.2f}")
```

    RMSE: 2.85
    stdev: 2.00
    Average Error: -0.15
    Stdev of Average Error: 0.98



```python
# Estimate RMSE per bin

for bin_num in range(1, 6):

    rmse_bin = statistics.mean(list(chain.from_iterable(main_RMSE_bin[bin_num])))
    sd_rmse_bin = statistics.stdev(list(chain.from_iterable(main_RMSE_bin[bin_num])))
    ave_bin = statistics.mean(list(chain.from_iterable(main_AVE_bin[bin_num])))
    sd_ave_bin = statistics.stdev(list(chain.from_iterable(main_AVE_bin[bin_num])))

    print(f"\nRMSE & STDEV & Average Error per bin {bin_num}")

    print(f"RMSE: {rmse_bin:.2f}")
    print(f"STDEV: {sd_rmse_bin:.2f}")
    print(f"Average_Error: {ave_bin:.2f}")
    print(f"Stdev of Average_Error: {sd_ave_bin:.2f}")
```

    
    RMSE & STDEV & Average Error per bin 1
    RMSE: 0.91
    STDEV: 0.03
    Average_Error: 0.91
    Stdev of Average_Error: 0.03
    
    RMSE & STDEV & Average Error per bin 2
    RMSE: 0.76
    STDEV: 0.08
    Average_Error: 0.71
    Stdev of Average_Error: 0.11
    
    RMSE & STDEV & Average Error per bin 3
    RMSE: 3.41
    STDEV: 1.10
    Average_Error: -2.68
    Stdev of Average_Error: 0.97
    
    RMSE & STDEV & Average Error per bin 4
    RMSE: 16.33
    STDEV: 3.94
    Average_Error: -15.27
    Stdev of Average_Error: 3.18
    
    RMSE & STDEV & Average Error per bin 5
    RMSE: 56.95
    STDEV: 6.24
    Average_Error: -56.95
    Stdev of Average_Error: 6.24



```python

```
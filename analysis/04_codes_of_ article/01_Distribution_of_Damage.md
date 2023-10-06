### Intersection of two datasets grid and municipality


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
import statsmodels.api as sm
import statistics
import warnings

from matplotlib.ticker import StrMethodFormatter
from sty import fg, rs
from matplotlib import cm
```


```python
df_mun_merged = pd.read_csv("data/df_merged_2.csv")

# Remove the duplicated rows and reset the indices
df_mun_merged.drop_duplicates(keep="first", inplace=True)
df_mun_merged = df_mun_merged.reset_index(drop=True)
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
# The converted grid to mun dataset
final_df_new = pd.read_csv("final_df_new.csv")
final_df_new
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
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>weight*houses</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>y_pred_norm</th>
      <th>y_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012801000</td>
      <td>BOPHA</td>
      <td>2012</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.000000</td>
      <td>0.029315</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012801000</td>
      <td>CONSON</td>
      <td>2010</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.067228</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012801000</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.067228</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012801000</td>
      <td>FENGSHEN</td>
      <td>2008</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.132570</td>
      <td>0.000000</td>
      <td>0.027743</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012801000</td>
      <td>FUNG-WONG</td>
      <td>2014</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.309934</td>
      <td>0.029315</td>
      <td>0.064859</td>
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
    </tr>
    <tr>
      <th>64228</th>
      <td>PH175917000</td>
      <td>USAGI</td>
      <td>2013</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.266012</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64229</th>
      <td>PH175917000</td>
      <td>UTOR</td>
      <td>2013</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.266012</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64230</th>
      <td>PH175917000</td>
      <td>VAMCO</td>
      <td>2020</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.526049</td>
      <td>0.000000</td>
      <td>0.027821</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64231</th>
      <td>PH175917000</td>
      <td>VONGFONG</td>
      <td>2020</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.266012</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64232</th>
      <td>PH175917000</td>
      <td>YUTU</td>
      <td>2018</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.266012</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>64233 rows × 9 columns</p>
</div>




```python
# Combine typhoon name and year columns

for i in range(len(final_df_new)):
    final_df_new.at[i, "typhoon_name"] = final_df_new.loc[i, "typhoon_name"] + str(
        final_df_new.loc[i, "typhoon_year"]
    )

final_df_new.drop(["typhoon_year"], axis=1, inplace=True)
final_df_new
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
      <th>typhoon_name</th>
      <th>grid_point_id</th>
      <th>weight*houses</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>y_pred_norm</th>
      <th>y_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012801000</td>
      <td>BOPHA2012</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.000000</td>
      <td>0.029315</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012801000</td>
      <td>CONSON2010</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.067228</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012801000</td>
      <td>DURIAN2006</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.067228</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012801000</td>
      <td>FENGSHEN2008</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.132570</td>
      <td>0.000000</td>
      <td>0.027743</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012801000</td>
      <td>FUNG-WONG2014</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.309934</td>
      <td>0.029315</td>
      <td>0.064859</td>
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
    </tr>
    <tr>
      <th>64228</th>
      <td>PH175917000</td>
      <td>USAGI2013</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.266012</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64229</th>
      <td>PH175917000</td>
      <td>UTOR2013</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.266012</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64230</th>
      <td>PH175917000</td>
      <td>VAMCO2020</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.526049</td>
      <td>0.000000</td>
      <td>0.027821</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64231</th>
      <td>PH175917000</td>
      <td>VONGFONG2020</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.266012</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>64232</th>
      <td>PH175917000</td>
      <td>YUTU2018</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.266012</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>64233 rows × 8 columns</p>
</div>




```python
# Rename the columns
final_df_new = final_df_new.rename(
    columns={"ADM3_PCODE": "Mun_Code", "typhoon_name": "typhoon"}
)
final_df_new.head()
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
      <th>grid_point_id</th>
      <th>weight*houses</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>y_pred_norm</th>
      <th>y_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012801000</td>
      <td>BOPHA2012</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.000000</td>
      <td>0.029315</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012801000</td>
      <td>CONSON2010</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.067228</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012801000</td>
      <td>DURIAN2006</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.067228</td>
      <td>0.000000</td>
      <td>0.014069</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012801000</td>
      <td>FENGSHEN2008</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.132570</td>
      <td>0.000000</td>
      <td>0.027743</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012801000</td>
      <td>FUNG-WONG2014</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.309934</td>
      <td>0.029315</td>
      <td>0.064859</td>
    </tr>
  </tbody>
</table>
</div>



### Joined two datasets together (grid and municipality)


```python
# Merge DataFrames based on 'typhoon_name' and 'Mun_Code'
merged_df = pd.merge(
    final_df_new, df_mun_merged, on=["Mun_Code", "typhoon"], how="inner"
)
merged_df
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
      <th>grid_point_id</th>
      <th>weight*houses</th>
      <th>weight*y_pred*houses</th>
      <th>weight*y*houses</th>
      <th>y_pred_norm</th>
      <th>y_norm_x</th>
      <th>HAZ_rainfall_Total</th>
      <th>HAZ_rainfall_max_6h</th>
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
      <th>y_norm_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012801000</td>
      <td>BOPHA2012</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.000000</td>
      <td>0.029315</td>
      <td>0.000000</td>
      <td>13.433333</td>
      <td>1.358333</td>
      <td>...</td>
      <td>2.433090</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2.455357</td>
      <td>41.119221</td>
      <td>0.000000</td>
      <td>5496.144192</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012801000</td>
      <td>FENGSHEN2008</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.132570</td>
      <td>0.000000</td>
      <td>0.027743</td>
      <td>0.000000</td>
      <td>8.291667</td>
      <td>0.691667</td>
      <td>...</td>
      <td>2.433090</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2.455357</td>
      <td>41.119221</td>
      <td>0.000000</td>
      <td>1062.394083</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012801000</td>
      <td>FUNG-WONG2014</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.309934</td>
      <td>0.029315</td>
      <td>0.064859</td>
      <td>275.266667</td>
      <td>7.452778</td>
      <td>...</td>
      <td>2.433090</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2.455357</td>
      <td>41.119221</td>
      <td>0.000000</td>
      <td>4619.063214</td>
      <td>0.064859</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012801000</td>
      <td>GONI2015</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>0.140085</td>
      <td>0.193709</td>
      <td>0.029315</td>
      <td>0.040537</td>
      <td>463.866667</td>
      <td>10.637500</td>
      <td>...</td>
      <td>2.433090</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2.455357</td>
      <td>41.119221</td>
      <td>0.000000</td>
      <td>3303.342891</td>
      <td>0.040537</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012801000</td>
      <td>HAIMA2016</td>
      <td>11049</td>
      <td>477.856074</td>
      <td>15.099461</td>
      <td>1.467305</td>
      <td>3.159834</td>
      <td>0.307060</td>
      <td>138.525000</td>
      <td>9.805556</td>
      <td>...</td>
      <td>2.433090</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000000</td>
      <td>2.455357</td>
      <td>41.119221</td>
      <td>0.243309</td>
      <td>187612.253359</td>
      <td>0.307060</td>
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
      <td>PH175917000</td>
      <td>MEKKHALA2015</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>0.524565</td>
      <td>0.000000</td>
      <td>0.027743</td>
      <td>0.000000</td>
      <td>43.950000</td>
      <td>2.927273</td>
      <td>...</td>
      <td>34.946809</td>
      <td>1.010638</td>
      <td>0.478723</td>
      <td>0.37234</td>
      <td>1.010638</td>
      <td>3.208745</td>
      <td>27.287234</td>
      <td>0.000000</td>
      <td>1152.192272</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7856</th>
      <td>PH175917000</td>
      <td>MELOR2015</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>12.004983</td>
      <td>0.456090</td>
      <td>0.634909</td>
      <td>0.024121</td>
      <td>269.150000</td>
      <td>19.089583</td>
      <td>...</td>
      <td>34.946809</td>
      <td>1.010638</td>
      <td>0.478723</td>
      <td>0.37234</td>
      <td>1.010638</td>
      <td>3.208745</td>
      <td>27.287234</td>
      <td>0.000000</td>
      <td>41719.573828</td>
      <td>0.024121</td>
    </tr>
    <tr>
      <th>7857</th>
      <td>PH175917000</td>
      <td>NOCK-TEN2016</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>2.430880</td>
      <td>0.000000</td>
      <td>0.128562</td>
      <td>0.000000</td>
      <td>19.300000</td>
      <td>1.888636</td>
      <td>...</td>
      <td>34.946809</td>
      <td>1.010638</td>
      <td>0.478723</td>
      <td>0.37234</td>
      <td>1.010638</td>
      <td>3.208745</td>
      <td>27.287234</td>
      <td>0.000000</td>
      <td>11621.854065</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7858</th>
      <td>PH175917000</td>
      <td>PHANFONE2019</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>10.485046</td>
      <td>7.184720</td>
      <td>0.554524</td>
      <td>0.379979</td>
      <td>145.525000</td>
      <td>12.252083</td>
      <td>...</td>
      <td>34.946809</td>
      <td>1.010638</td>
      <td>0.478723</td>
      <td>0.37234</td>
      <td>1.010638</td>
      <td>3.208745</td>
      <td>27.287234</td>
      <td>0.425532</td>
      <td>63356.052808</td>
      <td>0.379979</td>
    </tr>
    <tr>
      <th>7859</th>
      <td>PH175917000</td>
      <td>RAMMASUN2014</td>
      <td>12947</td>
      <td>1890.820029</td>
      <td>1.249109</td>
      <td>0.958985</td>
      <td>0.066062</td>
      <td>0.050718</td>
      <td>209.400000</td>
      <td>10.208333</td>
      <td>...</td>
      <td>34.946809</td>
      <td>1.010638</td>
      <td>0.478723</td>
      <td>0.37234</td>
      <td>1.010638</td>
      <td>3.208745</td>
      <td>27.287234</td>
      <td>0.053191</td>
      <td>5415.199043</td>
      <td>0.050718</td>
    </tr>
  </tbody>
</table>
<p>7860 rows × 46 columns</p>
</div>




```python
# makes the data
y1 = merged_df["y_norm_x"]
y2 = final_df_new["y_norm"]
colors = ["b", "r"]

# plots the histogram
fig, ax1 = plt.subplots()
ax1.hist(
    [y1, y2],
    color=colors,
    log=True,
    bins=20,
    label=["Original ground truth", "Grid to Municipality transformation"],
    width=2.3,
)
# ax1.set_xlim(-2, 100)
ax1.set_ylabel("Frequency", labelpad=20, size=12)
ax1.set_xlabel("Damage Percentage", labelpad=20, size=12)
plt.tight_layout()

plt.legend(loc="upper right")

# plt.savefig("figures/y_norm_histplot.pdf")
# Displaying the plot
plt.show()
```


    
![png](output_9_0.png)
    



```python
# Define the bins
bins = [0, 0.00009, 1, 10, 50, 100]

# Plotting the bar plots side by side
check_df1 = pd.cut(merged_df["y_norm_x"], bins=bins, include_lowest=True)
df1_counts = check_df1.value_counts(sort=False)

check_df2 = pd.cut(final_df_new["y_norm"], bins=bins, include_lowest=True)
df2_counts = check_df2.value_counts(sort=False)


# Calculate the x-axis positions for the bars
x = np.arange(len(df1_counts))
width = 0.35

# Plotting df1 bars
fig, ax = plt.subplots()
rects1 = ax.bar(
    x - width / 2,
    df1_counts,
    width,
    color="b",
    # alpha=0.5,
    log=True,
    label="Original ground truth",
)

# Plotting df2 bars
rects2 = ax.bar(
    x + width / 2,
    df2_counts,
    width,
    color="r",
    # alpha=0.5,
    log=True,
    label="Grid to Municipality transformation",
)

# Adding counts for df1
for rect in rects1:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        height + 0.5,
        height,
        ha="center",
        va="bottom",
    )

# Adding counts for df2
for rect in rects2:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        height + 0.5,
        height,
        ha="center",
        va="bottom",
    )

# Adjusting y-axis limits
ax.set_ylim(0, max(max(df1_counts), max(df2_counts)) * 2)


# Adding labels and title to the plot
ax.set_xlabel("Damage Percentage", labelpad=20, size=12)
ax.set_ylabel("Frequency", labelpad=20, size=12)
ax.set_title("")
ax.set_xticks(x)
ax.set_xticklabels(df1_counts.index)

plt.tight_layout()
ax.legend()

# Displaying the plot
# fig.savefig("figures/y_norm_barplots.pdf")
plt.show()
```

    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_10363/3796434428.py:62: UserWarning: Attempted to set non-positive bottom ylim on a log-scaled axis.
    Invalid limit will be ignored.
      ax.set_ylim(0, max(max(df1_counts), max(df2_counts)) * 2)



    
![png](output_10_1.png)
    



```python

```

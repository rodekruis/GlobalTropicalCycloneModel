```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
# This file is the municipality dataset merged with ground truth as column y_norm (grid to mun)
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
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df['DAM_perc_dmg'], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:],samples_per_bin2)
```




    [<matplotlib.lines.Line2D at 0x7ffc83320790>]




    
![png](output_2_1.png)
    



```python
print(samples_per_bin2)
print(binsP2)
```

    [4086 2431 1061  405   90]
    [0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]



```python
df["DAM_perc_dmg"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    4086
    (9e-05, 1.0]       2431
    (1.0, 10.0]        1061
    (10.0, 50.0]        405
    (50.0, 101.0]        90
    Name: DAM_perc_dmg, dtype: int64




```python
#Count numbers in each bin and create a bar plot

check = pd.cut(df['y_norm'], bins=[0, 0.00009, 1, 10, 50, 100], include_lowest=True)
ax = check.value_counts(sort=False).plot.bar(rot=0, color="b", figsize=(6,4))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
#ax.spines['left'].set_visible(False)
    

# Set x-axis label
ax.set_xlabel("Damage percentage", labelpad=20, size=12)

# Set y-axis label
ax.set_ylabel("Frequency", labelpad=20, size=12)


fig = ax.get_figure()
fig.savefig('figures/bin_of_damage_mun(y_norm).pdf', bbox_inches='tight')
#plt.show()
```


    
![png](output_5_0.png)
    



```python
from matplotlib.ticker import StrMethodFormatter
ax = df.hist(column='y_norm', bins=20, grid=False, figsize=(6,4), color='b', zorder=2, rwidth=0.9,
            log=True
            )

ax = ax[0]
for x in ax:

    # Despine
    x.spines['right'].set_visible(False)
    x.spines['top'].set_visible(False)
    x.spines['left'].set_visible(False)
    

    # Set title
    x.set_title("", size=10, color='b')

    # Set x-axis label
    x.set_xlabel("Damage percentage", labelpad=20, size=12)

    # Set y-axis label
    x.set_ylabel("Frequency", labelpad=20, size=12)
    
fig = ax[0].get_figure()
fig.savefig('figures/hist_of_damage_mun(y_norm).pdf', bbox_inches='tight')
```


    
![png](output_6_0.png)
    



```python
plt.rcParams["figure.figsize"] = [5, 4]
plt.rcParams["figure.autolayout"] = True

samples_per_bin2, binsP2 = np.histogram(df["y_norm"], bins=100)
pdf = samples_per_bin2 / sum(samples_per_bin2)
cdf = np.insert(np.cumsum(pdf), 0, 0)
plt.plot(binsP2, cdf, label="CDF", color='b')
#plt.legend()

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)


plt.xlabel("Damage values")
plt.ylabel("Probability of Frequency")
plt.savefig("figures/cdf_of_damage_mun(y_norm).pdf", bbox_inches='tight')
```


    
![png](output_7_0.png)
    



```python

```

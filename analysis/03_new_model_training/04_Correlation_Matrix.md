# Correlation Matrix

This code is utilized to compute the correlation among various features present in the input dataset.



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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from utils import get_training_dataset
```


```python
# Read csv file and import to df
df = pd.read_csv("data/updated_corr.csv")
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
      <td>-0.213039</td>
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



#### You can check the correlation among features before and after removing windspeed == 0


```python
# Remove wind_speed where values==0
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
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
      <th>138</th>
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
      <th>139</th>
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
      <th>148</th>
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
      <th>149</th>
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
      <th>150</th>
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
# Estimate correlation matrix

fig, ax = plt.subplots()

corrMatrix = df.corr()

plt.rcParams["figure.figsize"] = (10, 10)

sn.set(font_scale=1)
heatmap = sn.heatmap(
    corrMatrix,
    annot=True,
    cbar_kws={"shrink": 0.5},
    annot_kws={"size": 7},
    cmap="vlag_r",
    center=0,
)

# sn.diverging_palette(220, 20, as_cmap=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va="top", color="red")

heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 10}, pad=22)

plt.show()
```


    
![png](output_6_0.png)
    



```python
# The absolute value of correlation

fig, ax = plt.subplots()

corrMatrix_abs = df.corr().abs()
# print (corrMatrix)

plt.rcParams["figure.figsize"] = (9, 6)

sn.set(font_scale=1)
heatmap = sn.heatmap(
    corrMatrix_abs,
    annot=True,
    cbar_kws={"shrink": 0.5},
    annot_kws={"size": 8},
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va="top", color="red")

heatmap.set_title("Correlation Heatmap (abs)", fontdict={"fontsize": 12}, pad=22)

plt.show()
```


    
![png](output_7_0.png)
    



```python
# Print out correlated pairs of features
pair = (
    corrMatrix_abs.where(np.triu(np.ones(corrMatrix_abs.shape), k=1).astype(np.bool))
    .stack()
    .sort_values(ascending=True)
)
pairs = pair[pair.gt(0.8)]

if len(pairs) > 0:
    print(pairs)
else:
    print("No correlated features found!")
```

    mean_slope       std_tri             0.819225
    mean_tri         std_tri             0.819606
    rural            water               0.829211
    std_slope        mean_tri            0.852646
    mean_slope       std_slope           0.854719
    rainfall_max_6h  rainfall_max_24h    0.922418
    total_houses     total_pop           0.961985
    std_slope        std_tri             0.989076
    mean_slope       mean_tri            0.998442
    dtype: float64


    /var/folders/sx/c10hm4fj3glf7mw1_mzwcl700000gn/T/ipykernel_27270/2844072590.py:3: DeprecationWarning: `np.bool` is a deprecated alias for the builtin `bool`. To silence this warning, use `bool` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.bool_` here.
    Deprecated in NumPy 1.20; for more details and guidance: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
      corrMatrix_abs.where(np.triu(np.ones(corrMatrix_abs.shape), k=1).astype(np.bool))


#### Simple Scatterplot to show relation between ‘total_buildings’ and ‘relative_wealth_index’, coloured by damage


```python
# For All damaged values
x = df["rwi"]
y = df["total_houses"]
plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 100})
plt.scatter(
    x,
    y,
    c=df["percent_houses_damaged"],
    cmap="viridis_r",
    s=5,
    vmin=0,
    vmax=100,
)

plt.colorbar().set_label("low To high damage", size=12, color="green")
plt.title("RWI vs total_houses")
plt.xlabel("RWI")
plt.ylabel("total_houses")
plt.show()
```


    
![png](output_10_0.png)
    



```python
# For damaged values > 10
x = df[df.percent_houses_damaged > 10]["rwi"]
y = df[df.percent_houses_damaged > 10]["total_houses"]
plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 100})
plt.scatter(
    x,
    y,
    c=df[df.percent_houses_damaged > 10]["percent_houses_damaged"],
    cmap="viridis_r",
    s=5,
    vmin=0,
    vmax=100,
)
# plt.scatter(x, y, c='red', cmap=df['percent_buildings_damaged'], s=20)
plt.colorbar().set_label("low To high damage", size=12, color="green")
plt.title("RWI vs total_houses")
plt.xlabel("RWI")
plt.ylabel("total_houses")
plt.show()
```


    
![png](output_11_0.png)
    


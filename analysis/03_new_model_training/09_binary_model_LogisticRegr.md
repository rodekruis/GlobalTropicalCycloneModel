# Binary Logistic Regression Model

Based on the target values and using the test_trial method, we have set a threshold of 10.0 to convert the continuous target into a binary target. Subsequently, we employ the Logistic Regression algorithm on the input dataset with the binary target. We evaluate the model performance using the Confusion Matrix and Classification Report.


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
import imblearn
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from utils import get_training_dataset
```


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
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)

# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```


```python
# Define a threshold to separate target into damaged and not_damaged
thres = 10.0

for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] >= thres:
        df.at[i, "binary_damage"] = 1
    else:
        df.at[i, "binary_damage"] = 0

df["binary_damage"] = df["binary_damage"].astype("int")

# Remove previous target 'percent_buildings_damaged' from the dataframe
df = df.drop(["percent_houses_damaged"], axis=1)

# Remove one of the rainfall features
df = df.drop(["rainfall_max_24h"], axis=1)
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
      <td>31.000000</td>
      <td>-0.213039</td>
      <td>1.018526</td>
      <td>0.481382</td>
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
      <td>3.301020</td>
      <td>-0.527000</td>
      <td>1.579400</td>
      <td>1.060468</td>
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
      <td>12.103741</td>
      <td>-0.283000</td>
      <td>0.551764</td>
      <td>0.298116</td>
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
      <td>645.899660</td>
      <td>-0.358889</td>
      <td>2.107949</td>
      <td>2.638290</td>
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
      <td>1071.731293</td>
      <td>-0.462800</td>
      <td>3.538881</td>
      <td>3.981129</td>
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
      <td>4449.357133</td>
      <td>0.508167</td>
      <td>3.790141</td>
      <td>4.198243</td>
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
      <td>1521.435795</td>
      <td>-0.174100</td>
      <td>3.532580</td>
      <td>3.041204</td>
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
      <td>930.647069</td>
      <td>-0.244286</td>
      <td>4.444498</td>
      <td>2.646619</td>
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
      <td>1800.666044</td>
      <td>0.038000</td>
      <td>5.816195</td>
      <td>4.268518</td>
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
      <td>373.146778</td>
      <td>-0.175000</td>
      <td>6.730992</td>
      <td>4.143207</td>
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
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>141258 rows × 21 columns</p>
</div>




```python
# Show bar plot
plt.figure(figsize=(4, 3))
sns.countplot(x="binary_damage", data=df, palette="hls")
plt.title("bar_plot (counts of observations)")
plt.show()
```



![png](output_6_0.png)




```python
# Remove zeros from wind_speed
df = (df[(df[["wind_speed"]] != 0).any(axis=1)]).reset_index(drop=True)
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
      <th>percent_houses_damaged_5years</th>
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
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN</td>
      <td>11.428974</td>
      <td>297.027578</td>
      <td>0.929167</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN</td>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>0.716667</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN</td>
      <td>12.511864</td>
      <td>273.639330</td>
      <td>0.568750</td>
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
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN</td>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>0.589583</td>
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
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Show histogram of damage
df.hist(column="binary_damage", figsize=(4, 3))
```




    array([[<AxesSubplot:title={'center':'binary_damage'}>]], dtype=object)





![png](output_8_1.png)




```python
# Define bins and data stratification
bins2 = [0, 0.1, 1]
samples_per_bin2, binsP2 = np.histogram(df["binary_damage"], bins=bins2)
```


```python
# Check the bins' intervalls (first bin means all zeros, second bin means 0 < values <= 1)
df["binary_damage"].value_counts(bins=binsP2)
```




    (-0.001, 0.1]    48685
    (0.1, 1.0]        1069
    Name: binary_damage, dtype: int64




```python
print(samples_per_bin2)
print(binsP2)
```

    [48685  1069]
    [0.  0.1 1. ]



```python
bin_index2 = np.digitize(df["binary_damage"], bins=binsP2)
```


```python
y_input_strat = bin_index2
```


```python
# For this model we need to remove highly correlated features obtained from correlation_matrix code
features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    # "rainfall_max_24h",
    "rwi",
    "mean_slope",
    # "std_slope",
    # "mean_tri",
    # "std_tri",
    # "mean_elev",
    "coast_length",
    # "with_coast",
    "urban",
    "rural",
    # "water",
    # "total_pop",
    "percent_houses_damaged_5years",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["binary_damage"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```


    Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
           'rwi', 'mean_slope', 'coast_length', 'urban', 'rural',
           'percent_houses_damaged_5years'],
          dtype='object')



```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["binary_damage"], stratify=y_input_strat, test_size=0.2
)
```


```python
# Check train data before resampling
print(Counter(y_train))
```

    Counter({0: 38948, 1: 855})



```python
# Ask the user whether to perform oversampling or undersampling
sampling_type = int(
    input(
        "Enter 1 for oversampling, 2 for undersampling or 3 for combination of both: "
    )
)
```

    Enter 1 for oversampling, 2 for undersampling or 3 for combination of both: 2



```python
if sampling_type == 1:
    # Define oversampling strategy
    over = RandomOverSampler(sampling_strategy=0.1)
    # Fit and apply the transform
    X_train, y_train = over.fit_resample(X_train, y_train)

elif sampling_type == 2:
    under = RandomUnderSampler(sampling_strategy=0.7)
    X_train, y_train = under.fit_resample(X_train, y_train)

elif sampling_type == 3:
    over = RandomOverSampler(sampling_strategy=0.1)
    X_train, y_train = over.fit_resample(X_train, y_train)

    under = RandomUnderSampler(sampling_strategy=0.7)
    X_train, y_train = under.fit_resample(X_train, y_train)


else:
    print("Invalid input. Please enter 1, 2 or 3.")
```


```python
# Check train data After resampling
print(Counter(y_train))
```

    Counter({0: 1221, 1: 855})



```python
# Define binary logistic regression model

model = LogisticRegression(solver="liblinear", random_state=0)
model.fit(X_train, y_train)

LogisticRegression(
    C=1.0,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    l1_ratio=None,
    # max_iter=100,
    multi_class="warn",
    n_jobs=None,
    penalty="l2",
    random_state=0,
    solver="liblinear",
    tol=0.0001,
    verbose=0,
)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LogisticRegression(multi_class=&#x27;warn&#x27;, random_state=0, solver=&#x27;liblinear&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">LogisticRegression</label><div class="sk-toggleable__content"><pre>LogisticRegression(multi_class=&#x27;warn&#x27;, random_state=0, solver=&#x27;liblinear&#x27;)</pre></div></div></div></div></div>




```python
y_pred = model.predict(X_test)
```


```python
# Check the coefficient values of features
model.coef_
```




    array([[ 1.75954572, -0.35680141, -0.70304726,  0.48033979, -0.26778132,
            -0.2103076 ,  0.14023605, -0.21488527, -0.27081205,  0.18078408]])




```python
model.score(X_train, y_train)
```




    0.9190751445086706




```python
# Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
```




    array([[8932,  805],
           [  13,  201]])




```python
# Classification Report
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support

               0       1.00      0.92      0.96      9737
               1       0.20      0.94      0.33       214

        accuracy                           0.92      9951
       macro avg       0.60      0.93      0.64      9951
    weighted avg       0.98      0.92      0.94      9951




```python
# Plot Confusion Matrix (Binary absolute and relative with colorbar)
fig, ax = plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    show_absolute=True,
    show_normed=True,
    colorbar=True,
)

ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted 0s", "Predicted 1s"))
ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual 0s", "Actual 1s"))
plt.title("Confusion_matrix")
plt.show()
```



![png](output_26_0.png)



## Feature Importance


```python
# insert coefficient values in a list
coef_lst = model.coef_[0].tolist()

# insert coefficient values in a df
df_coef = pd.DataFrame(columns=["feature", "coef_value"])
for i in range(len(features)):
    df_coef.at[i, "feature"] = features[i]
    df_coef.at[i, "coef_value"] = coef_lst[i]

df_coef
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
      <th>feature</th>
      <th>coef_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wind_speed</td>
      <td>1.759546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>track_distance</td>
      <td>-0.356801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>total_houses</td>
      <td>-0.703047</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rainfall_max_6h</td>
      <td>0.48034</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rwi</td>
      <td>-0.267781</td>
    </tr>
    <tr>
      <th>5</th>
      <td>mean_slope</td>
      <td>-0.210308</td>
    </tr>
    <tr>
      <th>6</th>
      <td>coast_length</td>
      <td>0.140236</td>
    </tr>
    <tr>
      <th>7</th>
      <td>urban</td>
      <td>-0.214885</td>
    </tr>
    <tr>
      <th>8</th>
      <td>rural</td>
      <td>-0.270812</td>
    </tr>
    <tr>
      <th>9</th>
      <td>percent_houses_damaged_5years</td>
      <td>0.180784</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Sorting the dataframe of coefficient values in a descending order

final_sorted_df = df_coef.sort_values(by=["coef_value"], ascending=False)
final_sorted_df = final_sorted_df.reset_index(drop=True)
final_sorted_df
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
      <th>feature</th>
      <th>coef_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>wind_speed</td>
      <td>1.759546</td>
    </tr>
    <tr>
      <th>1</th>
      <td>rainfall_max_6h</td>
      <td>0.48034</td>
    </tr>
    <tr>
      <th>2</th>
      <td>percent_houses_damaged_5years</td>
      <td>0.180784</td>
    </tr>
    <tr>
      <th>3</th>
      <td>coast_length</td>
      <td>0.140236</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mean_slope</td>
      <td>-0.210308</td>
    </tr>
    <tr>
      <th>5</th>
      <td>urban</td>
      <td>-0.214885</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rwi</td>
      <td>-0.267781</td>
    </tr>
    <tr>
      <th>7</th>
      <td>rural</td>
      <td>-0.270812</td>
    </tr>
    <tr>
      <th>8</th>
      <td>track_distance</td>
      <td>-0.356801</td>
    </tr>
    <tr>
      <th>9</th>
      <td>total_houses</td>
      <td>-0.703047</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0, 0, 1, 1])
# ax.bar(features,values)
ax.bar(final_sorted_df["feature"], final_sorted_df["coef_value"])
np.rot90(plt.xticks(rotation=90, fontsize=10))
ax.set_title("Coefficients_Values of Binary Logistic Regression Model", fontsize=12)
plt.show()
```



![png](output_30_0.png)

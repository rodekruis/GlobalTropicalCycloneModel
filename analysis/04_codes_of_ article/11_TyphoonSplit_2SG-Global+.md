# Typhoon's time split

#### We split based on typhoons' time, the training list includes the oldest 70% of typhoons.


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
from collections import Counter
from sty import fg, rs
from itertools import chain

from sklearn.metrics import confusion_matrix
from matplotlib import cm
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from utils import get_training_dataset, weight_file
```

    pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.



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
# Specify features
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
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df_data["percent_houses_damaged"], bins=bins2)
```


```python
# Define range of for loops
num_exp_main = 20

# Latest typhoons in terms of time (walk-forward)
num_exp = 12
typhoons_for_test = typhoons[-num_exp:]

# LOOCV
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

    # Define empty list to save RMSE in combined model
    test_RMSE_lst = []
    test_RMSE_bin = defaultdict(list)

    # Define empty list to save RMSE in model1
    test_RMSE_lst_M1 = []
    test_RMSE_bin_M1 = defaultdict(list)

    # Defin two lists to save RMSE and Average Error
    RMSE = defaultdict(list)
    AVE = defaultdict(list)

    for run_ix in range(num_exp):

        # WITHOUT removing old typhoons from training set
        typhoons_train_lst = typhoons[0 : run_ix + 27]

        # for run_ix in range(27, num_exp):

        # typhoons_for_test = typhoons[run_ix]
        # typhoons_train_lst = typhoons[:run_ix] + typhoons[run_ix + 1 :]

        bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
        y_input_strat = bin_index2

        # Split X and y from dataframe features
        X = df_data[features]
        y = df_data["percent_houses_damaged"]

        # LOOCV
        # For when we train over all typhoon this df_test is required
        # df_test = df_data[df_data["typhoon_name"] == typhoons_for_test]

        # Walk-forward
        df_test = df[df["typhoon_name"] == typhoons_for_test[run_ix]]

        df_train = pd.DataFrame()
        for run_ix_train in range(len(typhoons_train_lst)):
            df_train = df_train.append(
                df_data[df_data["typhoon_name"] == typhoons_train_lst[run_ix_train]]
            )

        # Split X and y from dataframe features
        X_test = df_test[features]
        X_train = df_train[features]

        y_train = df_train["percent_houses_damaged"]
        y_test = df_test["percent_houses_damaged"]

        print(df_test["typhoon_name"].unique())

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
        test_RMSE_lst_M1.insert(run_ix, rmseM1)
        # print("RMSE in total and per bin (M1 model)")
        # print(f"total: {rmseM1:.2f}")

        # Calculate RMSE per bins
        bin_index_test = np.digitize(y_test, bins=binsP2)
        RSME_test_model1 = np.zeros(num_bins - 1)

        for bin_num in range(1, num_bins):

            # Estimation of RMSE for test data
            if (
                len(y_test[bin_index_test == bin_num]) != 0
                and len(y_pred[bin_index_test == bin_num]) != 0
            ):
                mse_idx = mean_squared_error(
                    y_test[bin_index_test == bin_num], y_pred[bin_index_test == bin_num]
                )
                RSME_test_model1[bin_num - 1] = np.sqrt(mse_idx)

                # Add RMSE of Model1 to the list of each bin
                test_RMSE_bin_M1[bin_num].append(RSME_test_model1[bin_num - 1])
                # print(f"bin{[bin_num]}:{RSME_test_model1[bin_num-1]}")
        # else:
        #    test_RMSE_bin_M1[bin_num].insert(run_ix, "No exist")

        # Define a threshold to separate target into damaged and not_damaged
        thres = 10.0
        y_test_bool = y_test >= thres
        y_train_bool = y_train >= thres
        y_test_bin = (y_test_bool) * 1
        y_train_bin = (y_train_bool) * 1

        sum(y_train_bin)

        ## Define undersampling strategy
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
        cm_test = confusion_matrix(y_test_bin, y_pred_test)
        # print(cm_test)

        # Make prediction on train data and print Confusion Matrix
        y_pred_train = xgb_model.predict(X_train)
        cm_train = confusion_matrix(y_train_bin, y_pred_train)
        # print(cm_train)

        y_pred_train_us = xgb_model.predict(X_train_us)
        cm_train_us = confusion_matrix(y_train_us, y_pred_train_us)
        # print(cm_train_us)

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

        ## Last step is to add model combination (model M1 with model MR)
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
        y1 = fliterd_test_df1["percent_houses_damaged"]

        # For the output equal to 0 apply M1 to evaluate the performance
        y0_pred = xgb.predict(X0)
        y0 = fliterd_test_df0["percent_houses_damaged"]

        fliterd_test_df0["predicted_percent_damage"] = y0_pred
        if len(y1_pred) > 0:
            fliterd_test_df1["predicted_percent_damage"] = y1_pred

        # Join two dataframes together
        join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])

        y_join = join_test_dfs["percent_houses_damaged"]
        y_pred_join = join_test_dfs["predicted_percent_damage"]

        pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])

        pred_df["y_all"] = y_join
        pred_df["y_pred_all"] = y_pred_join

        # Filter damages greater than 10 to estimate RMSE for these values
        # pred_df["y_all"] = y_join[y_join > 10]
        # pred_df["y_pred_all"] = y_pred_join[y_join > 10]

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

        # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based
        if len(final_df["y_norm"]) > 0:
            rmse = sqrt(mean_squared_error(final_df["y_norm"], final_df["y_pred_norm"]))
            ave = (final_df["y_pred_norm"] - final_df["y_norm"]).sum() / len(
                final_df["y_norm"]
            )

            print(f"RMSE for grid_based model: {rmse:.2f}")
            print(f"Average Error for grid_based model: {ave:.2f}")

            RMSE["all"].append(rmse)
            AVE["all"].append(ave)

        bin_index = np.digitize(final_df["y_norm"], bins=binsP2)

        for bin_num in range(1, 6):
            if len(final_df["y_norm"][bin_index == bin_num]) > 0:

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
    print(RMSE["all"])
    # Save total RMSE in each iteration

    main_RMSE_lst.append(RMSE["all"])
    main_AVE_lst.append(AVE["all"])

    for bin_num in range(1, 6):

        # Save RMSE per bin in each iteration
        main_RMSE_bin[bin_num].append(RMSE[bin_num])
        main_AVE_bin[bin_num].append(AVE[bin_num])
```

    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.90
    Average Error for grid_based model: 1.10
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.25
    Average Error for grid_based model: 1.13
    ['YUTU2018']
    RMSE for grid_based model: 0.54
    Average Error for grid_based model: 0.01
    ['KAMMURI2019']
    RMSE for grid_based model: 3.33
    Average Error for grid_based model: 0.38
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.29
    Average Error for grid_based model: -0.47
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.09
    Average Error for grid_based model: -0.15
    ['VAMCO2020']
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: -0.12
    ['VONGFONG2020']
    RMSE for grid_based model: 1.71
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.89798699120394, 0.3873628753117008, 4.24658115128932, 0.5359081660343328, 3.332726223342172, 0.10032573368610705, 2.287726696867257, 0.20350617391876227, 2.0927506775920652, 0.908721576196806, 1.706438898938881, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.64
    Average Error for grid_based model: 0.99
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.38
    Average Error for grid_based model: 1.19
    ['YUTU2018']
    RMSE for grid_based model: 0.98
    Average Error for grid_based model: 0.05
    ['KAMMURI2019']
    RMSE for grid_based model: 3.17
    Average Error for grid_based model: 0.35
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.32
    Average Error for grid_based model: -0.45
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.30
    Average Error for grid_based model: -0.13
    ['VAMCO2020']
    RMSE for grid_based model: 0.99
    Average Error for grid_based model: -0.09
    ['VONGFONG2020']
    RMSE for grid_based model: 2.12
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.72
    Average Error for grid_based model: 0.15
    [4.643926906790069, 0.38724916405381743, 4.375342193921481, 0.9780362158040369, 3.1660162760198967, 0.10032573368610705, 2.3210961956636944, 0.20350617391876227, 2.302544982188544, 0.9904081800157728, 2.1157440448541753, 0.7210847327012103]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 5.14
    Average Error for grid_based model: 1.16
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.45
    Average Error for grid_based model: 1.30
    ['YUTU2018']
    RMSE for grid_based model: 1.04
    Average Error for grid_based model: 0.06
    ['KAMMURI2019']
    RMSE for grid_based model: 3.53
    Average Error for grid_based model: 0.47
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.32
    Average Error for grid_based model: -0.44
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.23
    Average Error for grid_based model: -0.09
    ['VAMCO2020']
    RMSE for grid_based model: 0.96
    Average Error for grid_based model: -0.11
    ['VONGFONG2020']
    RMSE for grid_based model: 2.02
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [5.143640790452074, 0.38724916405381743, 4.4492676285941295, 1.0421415621052081, 3.533052034458467, 0.10032573368610705, 2.3212648989765485, 0.20350617391876227, 2.230969720264949, 0.9649510437752827, 2.0162429658101426, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.87
    Average Error for grid_based model: 1.07
    ['SARIKA2016']
    RMSE for grid_based model: 0.40
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.19
    Average Error for grid_based model: 1.13
    ['YUTU2018']
    RMSE for grid_based model: 0.89
    Average Error for grid_based model: 0.03
    ['KAMMURI2019']
    RMSE for grid_based model: 3.41
    Average Error for grid_based model: 0.40
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.29
    Average Error for grid_based model: -0.47
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.11
    Average Error for grid_based model: -0.10
    ['VAMCO2020']
    RMSE for grid_based model: 0.96
    Average Error for grid_based model: -0.11
    ['VONGFONG2020']
    RMSE for grid_based model: 2.06
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.868625096746711, 0.39742034091076656, 4.191917265960617, 0.8860574066138054, 3.406700212139519, 0.10032573368610705, 2.287726696867257, 0.20350617391876227, 2.1084027641303598, 0.9593247164031854, 2.059920459103043, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.73
    Average Error for grid_based model: 1.01
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.53
    Average Error for grid_based model: 1.24
    ['YUTU2018']
    RMSE for grid_based model: 1.22
    Average Error for grid_based model: 0.09
    ['KAMMURI2019']
    RMSE for grid_based model: 3.24
    Average Error for grid_based model: 0.38
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.21
    Average Error for grid_based model: -0.43
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.22
    Average Error for grid_based model: -0.09
    ['VAMCO2020']
    RMSE for grid_based model: 1.00
    Average Error for grid_based model: -0.09
    ['VONGFONG2020']
    RMSE for grid_based model: 2.06
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.732549460441822, 0.38724916405381743, 4.528689816509462, 1.2188166463511148, 3.235300255969021, 0.10032573368610705, 2.2134221830010303, 0.20350617391876227, 2.2243463781936836, 0.9987380331000212, 2.063793356697999, 0.5883826769844145]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 5.01
    Average Error for grid_based model: 1.06
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 3.81
    Average Error for grid_based model: 1.07
    ['YUTU2018']
    RMSE for grid_based model: 1.00
    Average Error for grid_based model: 0.04
    ['KAMMURI2019']
    RMSE for grid_based model: 3.44
    Average Error for grid_based model: 0.39
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.28
    Average Error for grid_based model: -0.46
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.23
    Average Error for grid_based model: -0.09
    ['VAMCO2020']
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: -0.10
    ['VONGFONG2020']
    RMSE for grid_based model: 1.92
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [5.014177915819655, 0.38724916405381743, 3.812056150324613, 0.9963477899215575, 3.443075667162197, 0.10032573368610705, 2.278475361348838, 0.20350617391876227, 2.2310086276465064, 0.9348916499114593, 1.9173405707026907, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 5.11
    Average Error for grid_based model: 1.12
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.55
    Average Error for grid_based model: 1.29
    ['YUTU2018']
    RMSE for grid_based model: 0.47
    Average Error for grid_based model: -0.00
    ['KAMMURI2019']
    RMSE for grid_based model: 3.55
    Average Error for grid_based model: 0.47
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.33
    Average Error for grid_based model: -0.45
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 1.99
    Average Error for grid_based model: -0.10
    ['VAMCO2020']
    RMSE for grid_based model: 0.89
    Average Error for grid_based model: -0.13
    ['VONGFONG2020']
    RMSE for grid_based model: 1.44
    Average Error for grid_based model: 0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [5.106424213369466, 0.3872442971563539, 4.553492549811862, 0.47476210568345534, 3.546425147481553, 0.10032573368610705, 2.326828373688305, 0.20350617391876227, 1.9917318626486527, 0.8857008789152593, 1.442550709757815, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.93
    Average Error for grid_based model: 1.13
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.15
    Average Error for grid_based model: 1.15
    ['YUTU2018']
    RMSE for grid_based model: 1.24
    Average Error for grid_based model: 0.11
    ['KAMMURI2019']
    RMSE for grid_based model: 3.53
    Average Error for grid_based model: 0.43
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.29
    Average Error for grid_based model: -0.47
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.12
    Average Error for grid_based model: -0.10
    ['VAMCO2020']
    RMSE for grid_based model: 0.98
    Average Error for grid_based model: -0.10
    ['VONGFONG2020']
    RMSE for grid_based model: 2.05
    Average Error for grid_based model: -0.03
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.929243650942223, 0.38724916405381743, 4.152063085927219, 1.2394550350584275, 3.5322492800363037, 0.10032573368610705, 2.287726696867257, 0.20350617391876227, 2.1200469782112625, 0.9827205347531022, 2.048017968342111, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.97
    Average Error for grid_based model: 1.12
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.02
    Average Error for grid_based model: 1.13
    ['YUTU2018']
    RMSE for grid_based model: 1.02
    Average Error for grid_based model: 0.07
    ['KAMMURI2019']
    RMSE for grid_based model: 3.87
    Average Error for grid_based model: 0.58
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.29
    Average Error for grid_based model: -0.47
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.37
    Average Error for grid_based model: -0.12
    ['VAMCO2020']
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: -0.09
    ['VONGFONG2020']
    RMSE for grid_based model: 1.96
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.973178819193389, 0.3873138604049723, 4.019583331095755, 1.020361896786988, 3.865395138417035, 0.10032573368610705, 2.287726696867257, 0.20350617391876227, 2.3723694823831356, 1.0121753462276213, 1.96229744314873, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 5.30
    Average Error for grid_based model: 1.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.18
    Average Error for grid_based model: 1.17
    ['YUTU2018']
    RMSE for grid_based model: 0.56
    Average Error for grid_based model: 0.01
    ['KAMMURI2019']
    RMSE for grid_based model: 3.47
    Average Error for grid_based model: 0.41
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.34
    Average Error for grid_based model: -0.45
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.05
    Average Error for grid_based model: -0.09
    ['VAMCO2020']
    RMSE for grid_based model: 0.99
    Average Error for grid_based model: -0.09
    ['VONGFONG2020']
    RMSE for grid_based model: 2.09
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [5.296971308122341, 0.38728715348291715, 4.17701582670148, 0.5596328008957518, 3.465772018283456, 0.10032573368610705, 2.3423737648330367, 0.20350617391876227, 2.0455273293774474, 0.989380556447285, 2.086973897720817, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.98
    Average Error for grid_based model: 1.11
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.07
    Average Error for grid_based model: 1.13
    ['YUTU2018']
    RMSE for grid_based model: 0.48
    Average Error for grid_based model: 0.01
    ['KAMMURI2019']
    RMSE for grid_based model: 3.42
    Average Error for grid_based model: 0.38
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.31
    Average Error for grid_based model: -0.45
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.11
    ['VAMCO2020']
    RMSE for grid_based model: 1.22
    Average Error for grid_based model: -0.05
    ['VONGFONG2020']
    RMSE for grid_based model: 2.00
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.9777454057137085, 0.3873104181233576, 4.070209581062064, 0.4797905084273595, 3.423304529252159, 0.10032573368610705, 2.3133262891678705, 0.20350617391876227, 2.042360841472641, 1.2192382010087743, 1.9980885808098954, 0.5883797533621494]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.83
    Average Error for grid_based model: 1.05
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.22
    Average Error for grid_based model: 1.22
    ['YUTU2018']
    RMSE for grid_based model: 1.00
    Average Error for grid_based model: 0.08
    ['KAMMURI2019']
    RMSE for grid_based model: 3.15
    Average Error for grid_based model: 0.29
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.33
    Average Error for grid_based model: -0.45
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.06
    ['VAMCO2020']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: -0.08
    ['VONGFONG2020']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.83190262606735, 0.38724916405381743, 4.224958742098013, 1.0008666477206758, 3.148863822502166, 0.10032573368610705, 2.3276177236524584, 0.20350617391876227, 2.0388939334134446, 1.0943084445195743, 2.0438574339891256, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.85
    Average Error for grid_based model: 1.04
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.14
    Average Error for grid_based model: 1.20
    ['YUTU2018']
    RMSE for grid_based model: 1.03
    Average Error for grid_based model: 0.06
    ['KAMMURI2019']
    RMSE for grid_based model: 3.07
    Average Error for grid_based model: 0.25
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.31
    Average Error for grid_based model: -0.44
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.15
    Average Error for grid_based model: -0.08
    ['VAMCO2020']
    RMSE for grid_based model: 0.98
    Average Error for grid_based model: -0.11
    ['VONGFONG2020']
    RMSE for grid_based model: 2.08
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.854433089038929, 0.38998413160696255, 4.138002110690171, 1.0319740837294527, 3.067380395224317, 0.10032573368610705, 2.312264823063394, 0.20350617391876227, 2.148576851549262, 0.9796592522965012, 2.076082053522725, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.82
    Average Error for grid_based model: 1.02
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 3.57
    Average Error for grid_based model: 0.98
    ['YUTU2018']
    RMSE for grid_based model: 0.44
    Average Error for grid_based model: -0.01
    ['KAMMURI2019']
    RMSE for grid_based model: 3.25
    Average Error for grid_based model: 0.32
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.30
    Average Error for grid_based model: -0.45
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.07
    Average Error for grid_based model: -0.10
    ['VAMCO2020']
    RMSE for grid_based model: 0.94
    Average Error for grid_based model: -0.11
    ['VONGFONG2020']
    RMSE for grid_based model: 2.10
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.816002773684076, 0.38724916405381743, 3.5727813681580765, 0.43697227193048965, 3.2478848731407477, 0.10032573368610705, 2.30160036637117, 0.20350617391876227, 2.067759181582148, 0.9421079238233235, 2.1024933595053477, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 5.05
    Average Error for grid_based model: 1.11
    ['SARIKA2016']
    RMSE for grid_based model: 0.40
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: 1.08
    ['YUTU2018']
    RMSE for grid_based model: 0.92
    Average Error for grid_based model: 0.05
    ['KAMMURI2019']
    RMSE for grid_based model: 3.32
    Average Error for grid_based model: 0.46
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.30
    Average Error for grid_based model: -0.46
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.23
    Average Error for grid_based model: -0.10
    ['VAMCO2020']
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: -0.11
    ['VONGFONG2020']
    RMSE for grid_based model: 2.02
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [5.05397279186827, 0.39731921472896875, 3.9599996289858823, 0.9215965806802032, 3.315050047925868, 0.10032573368610705, 2.3014329803834155, 0.20350617391876227, 2.2255491785518204, 0.9292147003264294, 2.0167096827978357, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.96
    Average Error for grid_based model: 1.08
    ['SARIKA2016']
    RMSE for grid_based model: 0.40
    Average Error for grid_based model: 0.06
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.08
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 0.62
    Average Error for grid_based model: 0.01
    ['KAMMURI2019']
    RMSE for grid_based model: 3.49
    Average Error for grid_based model: 0.40
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.24
    Average Error for grid_based model: -0.46
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.14
    Average Error for grid_based model: -0.12
    ['VAMCO2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: -0.12
    ['VONGFONG2020']
    RMSE for grid_based model: 1.96
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.963966890192227, 0.40176488252989934, 4.083711137885532, 0.6166691523372914, 3.4873729884718094, 0.10032573368610705, 2.2375380350065344, 0.20350617391876227, 2.1403793502210684, 0.8847502923986874, 1.9573446226300233, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.86
    Average Error for grid_based model: 1.03
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.16
    Average Error for grid_based model: 1.14
    ['YUTU2018']
    RMSE for grid_based model: 1.11
    Average Error for grid_based model: 0.07
    ['KAMMURI2019']
    RMSE for grid_based model: 3.30
    Average Error for grid_based model: 0.40
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.30
    Average Error for grid_based model: -0.44
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.39
    Average Error for grid_based model: -0.06
    ['VAMCO2020']
    RMSE for grid_based model: 0.89
    Average Error for grid_based model: -0.12
    ['VONGFONG2020']
    RMSE for grid_based model: 2.07
    Average Error for grid_based model: -0.03
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.8570616097936155, 0.38724916405381743, 4.163282812854993, 1.1080752167028771, 3.3017356105000824, 0.10032573368610705, 2.2996489147832366, 0.20350617391876227, 2.3888923438563676, 0.8850307190953508, 2.0689866230102703, 0.5896785488203226]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.95
    Average Error for grid_based model: 1.12
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.18
    Average Error for grid_based model: 1.17
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.08
    ['KAMMURI2019']
    RMSE for grid_based model: 3.52
    Average Error for grid_based model: 0.49
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.32
    Average Error for grid_based model: -0.45
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.33
    Average Error for grid_based model: -0.16
    ['VAMCO2020']
    RMSE for grid_based model: 1.01
    Average Error for grid_based model: -0.09
    ['VONGFONG2020']
    RMSE for grid_based model: 1.99
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.9516809809505995, 0.38724916405381743, 4.175708742439066, 1.0907823169217505, 3.5178609251613238, 0.10032573368610705, 2.324781771107227, 0.20350617391876227, 2.3329881133668446, 1.0122596467277387, 1.9930489123357673, 0.5883715122664314]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 5.19
    Average Error for grid_based model: 1.12
    ['SARIKA2016']
    RMSE for grid_based model: 0.39
    Average Error for grid_based model: 0.05
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.14
    ['YUTU2018']
    RMSE for grid_based model: 0.47
    Average Error for grid_based model: -0.00
    ['KAMMURI2019']
    RMSE for grid_based model: 3.43
    Average Error for grid_based model: 0.46
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.24
    Average Error for grid_based model: -0.46
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.02
    Average Error for grid_based model: -0.09
    ['VAMCO2020']
    RMSE for grid_based model: 0.93
    Average Error for grid_based model: -0.12
    ['VONGFONG2020']
    RMSE for grid_based model: 1.62
    Average Error for grid_based model: -0.00
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [5.192198172104794, 0.38749284000333517, 4.212860907248365, 0.473931438138058, 3.4254259357477785, 0.10032573368610705, 2.2401689551816557, 0.20350617391876227, 2.0179739596745745, 0.9286890082493965, 1.6205665058952532, 0.5883725951331853]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.49
    Average Error for grid_based model: 1.00
    ['SARIKA2016']
    RMSE for grid_based model: 0.48
    Average Error for grid_based model: 0.06
    ['MANGKHUT2018']
    RMSE for grid_based model: 3.95
    Average Error for grid_based model: 1.14
    ['YUTU2018']
    RMSE for grid_based model: 0.66
    Average Error for grid_based model: 0.02
    ['KAMMURI2019']
    RMSE for grid_based model: 3.44
    Average Error for grid_based model: 0.46
    ['NAKRI2019']
    RMSE for grid_based model: 0.10
    Average Error for grid_based model: 0.06
    ['PHANFONE2019']
    RMSE for grid_based model: 2.29
    Average Error for grid_based model: -0.47
    ['SAUDEL2020']
    RMSE for grid_based model: 0.20
    Average Error for grid_based model: -0.00
    ['GONI2020']
    RMSE for grid_based model: 2.16
    Average Error for grid_based model: -0.06
    ['VAMCO2020']
    RMSE for grid_based model: 0.91
    Average Error for grid_based model: -0.12
    ['VONGFONG2020']
    RMSE for grid_based model: 2.03
    Average Error for grid_based model: -0.01
    ['MOLAVE2020']
    RMSE for grid_based model: 0.59
    Average Error for grid_based model: 0.13
    [4.493230986966791, 0.4793319905326768, 3.9491222795475136, 0.6553166284963444, 3.436030992885679, 0.10032573368610705, 2.290263620614003, 0.20350617391876227, 2.1620125795392844, 0.9132946379642469, 2.03061126246027, 0.5883715122664314]



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

    RMSE: 1.83
    stdev: 1.56
    Average Error: 0.19
    Stdev of Average Error: 0.46



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
    RMSE: 0.37
    STDEV: 0.31
    Average_Error: 0.05
    Stdev of Average_Error: 0.03
    
    RMSE & STDEV & Average Error per bin 2
    RMSE: 1.19
    STDEV: 1.24
    Average_Error: 0.35
    Stdev of Average_Error: 0.61
    
    RMSE & STDEV & Average Error per bin 3
    RMSE: 5.36
    STDEV: 3.96
    Average_Error: 1.16
    Stdev of Average_Error: 3.68
    
    RMSE & STDEV & Average Error per bin 4
    RMSE: 12.93
    STDEV: 2.81
    Average_Error: -2.92
    Stdev of Average_Error: 9.87
    
    RMSE & STDEV & Average Error per bin 5
    RMSE: 31.95
    STDEV: 22.56
    Average_Error: -31.95
    Stdev of Average_Error: 22.56


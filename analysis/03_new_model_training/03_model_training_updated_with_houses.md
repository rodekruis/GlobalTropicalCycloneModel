# Model training

The input data is prepared by joining the calculated windfield with damaged values.
Subsampling is done by dropping those rows where the windspeed is 0, the the data stratification is done on damaged values.
The XGBoost Reduced Over fitting model, was trained on this prepared input data with gridcells.
The RMSE calculated in total and per each bin.


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
from collections import defaultdict
import statistics

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
from sklearn.dummy import DummyRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import colorama
from colorama import Fore

from utils import get_training_dataset
```

    /Users/mersedehkooshki/opt/anaconda3/envs/global-storm/lib/python3.8/site-packages/xgboost/compat.py:36: FutureWarning: pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.
      from pandas import MultiIndex, Int64Index



```python
# Read csv file (which also damage5years added to it) and import to df
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
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



#### Check rows including NaN values regarding ['rwi']


```python
# Keep only rows including NaN values
df1 = df[df.isnull().any(axis=1)]

# Estimate number of rows with unique 'grid_point_id' including Nan values
len(df1["grid_point_id"].unique())
```




    224




```python
df["rwi"].isnull().sum()
```




    8736




```python
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)
```


```python
# Number of damaged values greater than 100
(df["percent_houses_damaged"] > 100).sum()
```




    3




```python
# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```


```python
# Show histogram of damage
df.hist(column="percent_houses_damaged", figsize=(4, 3))
```




    array([[<AxesSubplot:title={'center':'percent_houses_damaged'}>]],
          dtype=object)




    
![png](output_10_1.png)
    



```python
# Hist plot after data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```




    [<matplotlib.lines.Line2D at 0x7fc970775400>]




    
![png](output_11_1.png)
    



```python
# Check the bins' intervalls (first bin means all zeros, second bin means 0 < values <= 1)
df["percent_houses_damaged"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    129600
    (9e-05, 1.0]         7938
    (1.0, 10.0]          2634
    (10.0, 50.0]          939
    (50.0, 101.0]         147
    Name: percent_houses_damaged, dtype: int64




```python
# Remove zeros from wind_speed
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
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Hist plot after removing rows where windspeed is 0
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```




    [<matplotlib.lines.Line2D at 0x7fc970851c70>]




    
![png](output_14_1.png)
    



```python
print(samples_per_bin2)
print(binsP2)
```

    [38901  7232  2552   925   144]
    [0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]



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
# Defin two lists to save total RMSE of test and train data

test_RMSE = defaultdict(list)
train_RMSE = defaultdict(list)
```


```python
features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "rwi",
    "percent_houses_damaged_5years",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["percent_houses_damaged"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```


    Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
           'rainfall_max_24h', 'rwi', 'percent_houses_damaged_5years'],
          dtype='object')



```python
# Run XGBoost Reduced Overfitting in for loop to estimate RMSE per bins

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df["percent_houses_damaged"], stratify=y_input_strat, test_size=0.2
    )

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
        verbosity=1,
        eval_metric=["rmse", "logloss"],
        random_state=0,
    )

    eval_set = [(X_test, y_test)]
    xgb_model = xgb.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
        # sample_weight=pow(y_train, 2),
    )

    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    print(est2.summary())

    X2_test = sm.add_constant(X_test)

    y_pred_train_LREG = est2.predict(X2)
    mse_train_idx_LREG = mean_squared_error(y_train, y_pred_train_LREG)
    rmse_train_LREG = np.sqrt(mse_train_idx_LREG)

    ypred_LREG = est2.predict(X2_test)
    mse_idx_LREG = mean_squared_error(y_test, ypred_LREG)
    rmse_LREG = np.sqrt(mse_idx_LREG)

    print("----- Training ------")
    print(f"LREG Root mean squared error: {rmse_train_LREG:.2f}")
    print("----- Test ------")
    print(f"LREG Root mean squared error: {rmse_LREG:.2f}")

    # Calculate RMSE in total

    y_pred_train = xgb.predict(X_train)
    mse_train_idx = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train_idx)

    y_pred = xgb.predict(X_test)
    mse_idx = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse_idx)

    print("----- Training ------")
    print(f"Root mean squared error: {rmse_train:.2f}")

    print("----- Test ------")
    print(f"Root mean squared error: {rmse:.2f}")

    test_RMSE["all"].append(rmse)
    train_RMSE["all"].append(rmse_train)

    # Calculate RMSE per bins

    bin_index_test = np.digitize(y_test, bins=binsP2)
    bin_index_train = np.digitize(y_train, bins=binsP2)

    # Estimation of RMSE for train data
    y_pred_train = xgb.predict(X_train)
    for bin_num in range(1, 6):

        mse_train_idx = mean_squared_error(
            y_train[bin_index_train == bin_num],
            y_pred_train[bin_index_train == bin_num],
        )
        rmse_train = np.sqrt(mse_train_idx)

        # Estimation of RMSE for test data
        y_pred = xgb.predict(X_test)

        mse_idx = mean_squared_error(
            y_test[bin_index_test == bin_num],
            y_pred[bin_index_test == bin_num],
        )
        rmse = np.sqrt(mse_idx)

        train_RMSE[bin_num].append(rmse_train)
        test_RMSE[bin_num].append(rmse)
```

    [17:47:24] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.204
    Model:                                OLS   Adj. R-squared:                  0.204
    Method:                     Least Squares   F-statistic:                     1455.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:25   Log-Likelihood:            -1.1658e+05
    No. Observations:                   39803   AIC:                         2.332e+05
    Df Residuals:                       39795   BIC:                         2.332e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8312      0.023     36.632      0.000       0.787       0.876
    x1             2.8843      0.034     84.159      0.000       2.817       2.951
    x2             1.0020      0.034     29.063      0.000       0.934       1.070
    x3            -0.0541      0.026     -2.064      0.039      -0.106      -0.003
    x4             0.5699      0.059      9.689      0.000       0.455       0.685
    x5            -0.4995      0.059     -8.467      0.000      -0.615      -0.384
    x6            -0.0221      0.026     -0.848      0.396      -0.073       0.029
    x7             0.0895      0.022      4.095      0.000       0.047       0.132
    ==============================================================================
    Omnibus:                    57548.312   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21707490.098
    Skew:                           8.755   Prob(JB):                         0.00
    Kurtosis:                     116.059   Cond. No.                         5.88
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.53
    ----- Test ------
    LREG Root mean squared error: 4.46
    ----- Training ------
    Root mean squared error: 2.75
    ----- Test ------
    Root mean squared error: 3.14
    [17:47:25] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.200
    Model:                                OLS   Adj. R-squared:                  0.199
    Method:                     Least Squares   F-statistic:                     1418.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:27   Log-Likelihood:            -1.1657e+05
    No. Observations:                   39803   AIC:                         2.332e+05
    Df Residuals:                       39795   BIC:                         2.332e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8312      0.023     36.643      0.000       0.787       0.876
    x1             2.8227      0.034     82.810      0.000       2.756       2.890
    x2             0.9639      0.034     28.102      0.000       0.897       1.031
    x3            -0.0498      0.027     -1.877      0.060      -0.102       0.002
    x4             0.5619      0.059      9.498      0.000       0.446       0.678
    x5            -0.4788      0.059     -8.100      0.000      -0.595      -0.363
    x6            -0.0380      0.026     -1.449      0.147      -0.089       0.013
    x7             0.0969      0.022      4.399      0.000       0.054       0.140
    ==============================================================================
    Omnibus:                    57779.885   Durbin-Watson:                   2.019
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22269397.500
    Skew:                           8.817   Prob(JB):                         0.00
    Kurtosis:                     117.529   Cond. No.                         5.89
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.53
    ----- Test ------
    LREG Root mean squared error: 4.46
    ----- Training ------
    Root mean squared error: 2.76
    ----- Test ------
    Root mean squared error: 3.07
    [17:47:27] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.203
    Model:                                OLS   Adj. R-squared:                  0.203
    Method:                     Least Squares   F-statistic:                     1447.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:28   Log-Likelihood:            -1.1674e+05
    No. Observations:                   39803   AIC:                         2.335e+05
    Df Residuals:                       39795   BIC:                         2.336e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8365      0.023     36.712      0.000       0.792       0.881
    x1             2.8845      0.034     84.042      0.000       2.817       2.952
    x2             0.9869      0.035     28.533      0.000       0.919       1.055
    x3            -0.0604      0.026     -2.315      0.021      -0.112      -0.009
    x4             0.5274      0.060      8.815      0.000       0.410       0.645
    x5            -0.4702      0.060     -7.874      0.000      -0.587      -0.353
    x6            -0.0133      0.026     -0.506      0.613      -0.065       0.038
    x7             0.0971      0.022      4.439      0.000       0.054       0.140
    ==============================================================================
    Omnibus:                    57846.917   Durbin-Watson:                   2.004
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22628297.256
    Skew:                           8.831   Prob(JB):                         0.00
    Kurtosis:                     118.465   Cond. No.                         5.92
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.55
    ----- Test ------
    LREG Root mean squared error: 4.38
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 3.14
    [17:47:28] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.200
    Model:                                OLS   Adj. R-squared:                  0.200
    Method:                     Least Squares   F-statistic:                     1421.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:29   Log-Likelihood:            -1.1648e+05
    No. Observations:                   39803   AIC:                         2.330e+05
    Df Residuals:                       39795   BIC:                         2.330e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8157      0.023     36.041      0.000       0.771       0.860
    x1             2.8397      0.034     83.098      0.000       2.773       2.907
    x2             0.9772      0.034     28.439      0.000       0.910       1.045
    x3            -0.0488      0.026     -1.883      0.060      -0.100       0.002
    x4             0.5662      0.059      9.573      0.000       0.450       0.682
    x5            -0.5066      0.059     -8.570      0.000      -0.623      -0.391
    x6            -0.0439      0.026     -1.677      0.094      -0.095       0.007
    x7             0.1022      0.022      4.694      0.000       0.060       0.145
    ==============================================================================
    Omnibus:                    58475.647   Durbin-Watson:                   1.984
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         24207485.938
    Skew:                           9.003   Prob(JB):                         0.00
    Kurtosis:                     122.466   Cond. No.                         5.91
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.51
    ----- Test ------
    LREG Root mean squared error: 4.50
    ----- Training ------
    Root mean squared error: 2.68
    ----- Test ------
    Root mean squared error: 3.37
    [17:47:30] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.204
    Model:                                OLS   Adj. R-squared:                  0.204
    Method:                     Least Squares   F-statistic:                     1458.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:31   Log-Likelihood:            -1.1669e+05
    No. Observations:                   39803   AIC:                         2.334e+05
    Df Residuals:                       39795   BIC:                         2.335e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8368      0.023     36.772      0.000       0.792       0.881
    x1             2.8932      0.034     84.425      0.000       2.826       2.960
    x2             1.0058      0.035     29.138      0.000       0.938       1.074
    x3            -0.0492      0.026     -1.889      0.059      -0.100       0.002
    x4             0.5679      0.059      9.560      0.000       0.451       0.684
    x5            -0.4958      0.059     -8.357      0.000      -0.612      -0.380
    x6            -0.0182      0.026     -0.692      0.489      -0.070       0.033
    x7             0.0987      0.022      4.433      0.000       0.055       0.142
    ==============================================================================
    Omnibus:                    57661.378   Durbin-Watson:                   2.001
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22029811.687
    Skew:                           8.784   Prob(JB):                         0.00
    Kurtosis:                     116.906   Cond. No.                         5.89
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.54
    ----- Test ------
    LREG Root mean squared error: 4.40
    ----- Training ------
    Root mean squared error: 2.65
    ----- Test ------
    Root mean squared error: 3.40
    [17:47:31] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     1442.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:32   Log-Likelihood:            -1.1671e+05
    No. Observations:                   39803   AIC:                         2.334e+05
    Df Residuals:                       39795   BIC:                         2.335e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8384      0.023     36.825      0.000       0.794       0.883
    x1             2.8659      0.034     83.740      0.000       2.799       2.933
    x2             0.9705      0.034     28.206      0.000       0.903       1.038
    x3            -0.0513      0.026     -1.971      0.049      -0.102      -0.000
    x4             0.5402      0.060      9.078      0.000       0.424       0.657
    x5            -0.4853      0.059     -8.191      0.000      -0.601      -0.369
    x6            -0.0245      0.026     -0.934      0.350      -0.076       0.027
    x7             0.1022      0.022      4.570      0.000       0.058       0.146
    ==============================================================================
    Omnibus:                    57300.618   Durbin-Watson:                   2.001
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21062609.565
    Skew:                           8.690   Prob(JB):                         0.00
    Kurtosis:                     114.346   Cond. No.                         5.90
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.54
    ----- Test ------
    LREG Root mean squared error: 4.39
    ----- Training ------
    Root mean squared error: 2.79
    ----- Test ------
    Root mean squared error: 3.05
    [17:47:32] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.203
    Model:                                OLS   Adj. R-squared:                  0.203
    Method:                     Least Squares   F-statistic:                     1450.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:34   Log-Likelihood:            -1.1626e+05
    No. Observations:                   39803   AIC:                         2.325e+05
    Df Residuals:                       39795   BIC:                         2.326e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8292      0.023     36.836      0.000       0.785       0.873
    x1             2.8509      0.034     84.025      0.000       2.784       2.917
    x2             0.9842      0.034     28.861      0.000       0.917       1.051
    x3            -0.0522      0.026     -1.997      0.046      -0.103      -0.001
    x4             0.5230      0.059      8.920      0.000       0.408       0.638
    x5            -0.4549      0.059     -7.771      0.000      -0.570      -0.340
    x6            -0.0225      0.026     -0.866      0.387      -0.073       0.028
    x7             0.0912      0.022      4.210      0.000       0.049       0.134
    ==============================================================================
    Omnibus:                    58079.685   Durbin-Watson:                   1.988
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23079376.477
    Skew:                           8.897   Prob(JB):                         0.00
    Kurtosis:                     119.617   Cond. No.                         5.89
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.49
    ----- Test ------
    LREG Root mean squared error: 4.60
    ----- Training ------
    Root mean squared error: 2.69
    ----- Test ------
    Root mean squared error: 3.41
    [17:47:34] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.203
    Model:                                OLS   Adj. R-squared:                  0.203
    Method:                     Least Squares   F-statistic:                     1448.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:35   Log-Likelihood:            -1.1652e+05
    No. Observations:                   39803   AIC:                         2.331e+05
    Df Residuals:                       39795   BIC:                         2.331e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8293      0.023     36.601      0.000       0.785       0.874
    x1             2.8799      0.034     83.974      0.000       2.813       2.947
    x2             0.9983      0.034     29.043      0.000       0.931       1.066
    x3            -0.0562      0.027     -2.093      0.036      -0.109      -0.004
    x4             0.5228      0.059      8.840      0.000       0.407       0.639
    x5            -0.4579      0.059     -7.735      0.000      -0.574      -0.342
    x6            -0.0320      0.026     -1.215      0.224      -0.084       0.020
    x7             0.0994      0.022      4.507      0.000       0.056       0.143
    ==============================================================================
    Omnibus:                    57629.631   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21668132.394
    Skew:                           8.782   Prob(JB):                         0.00
    Kurtosis:                     115.946   Cond. No.                         5.90
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.52
    ----- Test ------
    LREG Root mean squared error: 4.48
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 3.26
    [17:47:35] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     1442.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:36   Log-Likelihood:            -1.1616e+05
    No. Observations:                   39803   AIC:                         2.323e+05
    Df Residuals:                       39795   BIC:                         2.324e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8245      0.022     36.717      0.000       0.780       0.869
    x1             2.8395      0.034     83.956      0.000       2.773       2.906
    x2             0.9783      0.034     28.774      0.000       0.912       1.045
    x3            -0.0539      0.026     -2.054      0.040      -0.105      -0.002
    x4             0.4499      0.058      7.702      0.000       0.335       0.564
    x5            -0.3846      0.058     -6.596      0.000      -0.499      -0.270
    x6            -0.0255      0.026     -0.977      0.329      -0.077       0.026
    x7             0.0904      0.022      4.101      0.000       0.047       0.134
    ==============================================================================
    Omnibus:                    57627.892   Durbin-Watson:                   1.994
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21885111.297
    Skew:                           8.777   Prob(JB):                         0.00
    Kurtosis:                     116.525   Cond. No.                         5.88
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.48
    ----- Test ------
    LREG Root mean squared error: 4.64
    ----- Training ------
    Root mean squared error: 2.65
    ----- Test ------
    Root mean squared error: 3.48
    [17:47:37] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.204
    Model:                                OLS   Adj. R-squared:                  0.204
    Method:                     Least Squares   F-statistic:                     1456.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:38   Log-Likelihood:            -1.1619e+05
    No. Observations:                   39803   AIC:                         2.324e+05
    Df Residuals:                       39795   BIC:                         2.325e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8310      0.022     36.984      0.000       0.787       0.875
    x1             2.8629      0.034     84.404      0.000       2.796       2.929
    x2             0.9864      0.034     28.976      0.000       0.920       1.053
    x3            -0.0521      0.026     -1.985      0.047      -0.104      -0.001
    x4             0.5038      0.059      8.580      0.000       0.389       0.619
    x5            -0.4417      0.059     -7.529      0.000      -0.557      -0.327
    x6            -0.0239      0.026     -0.921      0.357      -0.075       0.027
    x7             0.1062      0.023      4.685      0.000       0.062       0.151
    ==============================================================================
    Omnibus:                    57959.277   Durbin-Watson:                   2.006
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22899752.813
    Skew:                           8.862   Prob(JB):                         0.00
    Kurtosis:                     119.163   Cond. No.                         5.90
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.48
    ----- Test ------
    LREG Root mean squared error: 4.63
    ----- Training ------
    Root mean squared error: 2.65
    ----- Test ------
    Root mean squared error: 3.54
    [17:47:38] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     1437.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:39   Log-Likelihood:            -1.1659e+05
    No. Observations:                   39803   AIC:                         2.332e+05
    Df Residuals:                       39795   BIC:                         2.333e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8292      0.023     36.532      0.000       0.785       0.874
    x1             2.8604      0.034     83.313      0.000       2.793       2.928
    x2             0.9841      0.034     28.542      0.000       0.916       1.052
    x3            -0.0510      0.026     -1.950      0.051      -0.102       0.000
    x4             0.5623      0.059      9.500      0.000       0.446       0.678
    x5            -0.4886      0.059     -8.277      0.000      -0.604      -0.373
    x6            -0.0314      0.026     -1.203      0.229      -0.083       0.020
    x7             0.0979      0.022      4.365      0.000       0.054       0.142
    ==============================================================================
    Omnibus:                    58044.541   Durbin-Watson:                   2.001
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23154555.699
    Skew:                           8.884   Prob(JB):                         0.00
    Kurtosis:                     119.815   Cond. No.                         5.88
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.53
    ----- Test ------
    LREG Root mean squared error: 4.45
    ----- Training ------
    Root mean squared error: 2.74
    ----- Test ------
    Root mean squared error: 3.17
    [17:47:39] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.206
    Model:                                OLS   Adj. R-squared:                  0.206
    Method:                     Least Squares   F-statistic:                     1472.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:41   Log-Likelihood:            -1.1611e+05
    No. Observations:                   39803   AIC:                         2.322e+05
    Df Residuals:                       39795   BIC:                         2.323e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8210      0.022     36.610      0.000       0.777       0.865
    x1             2.8691      0.034     85.076      0.000       2.803       2.935
    x2             0.9887      0.034     29.149      0.000       0.922       1.055
    x3            -0.0582      0.026     -2.229      0.026      -0.109      -0.007
    x4             0.4977      0.059      8.500      0.000       0.383       0.612
    x5            -0.4531      0.059     -7.738      0.000      -0.568      -0.338
    x6            -0.0195      0.026     -0.754      0.451      -0.070       0.031
    x7             0.1019      0.022      4.684      0.000       0.059       0.145
    ==============================================================================
    Omnibus:                    57428.740   Durbin-Watson:                   2.004
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21410952.406
    Skew:                           8.723   Prob(JB):                         0.00
    Kurtosis:                     115.275   Cond. No.                         5.90
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.47
    ----- Test ------
    LREG Root mean squared error: 4.66
    ----- Training ------
    Root mean squared error: 2.63
    ----- Test ------
    Root mean squared error: 3.57
    [17:47:41] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     1436.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:42   Log-Likelihood:            -1.1667e+05
    No. Observations:                   39803   AIC:                         2.333e+05
    Df Residuals:                       39795   BIC:                         2.334e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8324      0.023     36.603      0.000       0.788       0.877
    x1             2.8632      0.034     83.484      0.000       2.796       2.930
    x2             0.9854      0.035     28.537      0.000       0.918       1.053
    x3            -0.0519      0.026     -1.977      0.048      -0.103      -0.000
    x4             0.5442      0.059      9.226      0.000       0.429       0.660
    x5            -0.4706      0.059     -7.954      0.000      -0.587      -0.355
    x6            -0.0204      0.026     -0.778      0.436      -0.072       0.031
    x7             0.0978      0.022      4.405      0.000       0.054       0.141
    ==============================================================================
    Omnibus:                    57479.780   Durbin-Watson:                   2.005
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21437145.339
    Skew:                           8.739   Prob(JB):                         0.00
    Kurtosis:                     115.341   Cond. No.                         5.88
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.54
    ----- Test ------
    LREG Root mean squared error: 4.41
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 3.21
    [17:47:42] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.204
    Model:                                OLS   Adj. R-squared:                  0.204
    Method:                     Least Squares   F-statistic:                     1458.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:43   Log-Likelihood:            -1.1647e+05
    No. Observations:                   39803   AIC:                         2.329e+05
    Df Residuals:                       39795   BIC:                         2.330e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8272      0.023     36.556      0.000       0.783       0.872
    x1             2.8741      0.034     84.097      0.000       2.807       2.941
    x2             1.0046      0.034     29.229      0.000       0.937       1.072
    x3            -0.0433      0.026     -1.665      0.096      -0.094       0.008
    x4             0.6017      0.059     10.180      0.000       0.486       0.718
    x5            -0.5188      0.059     -8.792      0.000      -0.634      -0.403
    x6            -0.0383      0.026     -1.464      0.143      -0.090       0.013
    x7             0.1034      0.023      4.588      0.000       0.059       0.148
    ==============================================================================
    Omnibus:                    57420.709   Durbin-Watson:                   2.003
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21338312.670
    Skew:                           8.722   Prob(JB):                         0.00
    Kurtosis:                     115.080   Cond. No.                         5.91
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.51
    ----- Test ------
    LREG Root mean squared error: 4.51
    ----- Training ------
    Root mean squared error: 2.65
    ----- Test ------
    Root mean squared error: 3.39
    [17:47:44] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.204
    Model:                                OLS   Adj. R-squared:                  0.204
    Method:                     Least Squares   F-statistic:                     1457.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:45   Log-Likelihood:            -1.1647e+05
    No. Observations:                   39803   AIC:                         2.330e+05
    Df Residuals:                       39795   BIC:                         2.330e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8355      0.023     36.917      0.000       0.791       0.880
    x1             2.8792      0.034     84.317      0.000       2.812       2.946
    x2             0.9852      0.034     28.707      0.000       0.918       1.052
    x3            -0.0546      0.026     -2.091      0.037      -0.106      -0.003
    x4             0.4772      0.059      8.107      0.000       0.362       0.593
    x5            -0.4219      0.059     -7.145      0.000      -0.538      -0.306
    x6            -0.0360      0.026     -1.378      0.168      -0.087       0.015
    x7             0.0905      0.022      4.150      0.000       0.048       0.133
    ==============================================================================
    Omnibus:                    57916.247   Durbin-Watson:                   2.008
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22670246.020
    Skew:                           8.853   Prob(JB):                         0.00
    Kurtosis:                     118.568   Cond. No.                         5.88
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.51
    ----- Test ------
    LREG Root mean squared error: 4.50
    ----- Training ------
    Root mean squared error: 2.65
    ----- Test ------
    Root mean squared error: 3.34
    [17:47:45] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.206
    Model:                                OLS   Adj. R-squared:                  0.205
    Method:                     Least Squares   F-statistic:                     1472.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:46   Log-Likelihood:            -1.1614e+05
    No. Observations:                   39803   AIC:                         2.323e+05
    Df Residuals:                       39795   BIC:                         2.324e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8358      0.022     37.245      0.000       0.792       0.880
    x1             2.8570      0.034     84.439      0.000       2.791       2.923
    x2             0.9898      0.034     29.168      0.000       0.923       1.056
    x3            -0.0683      0.027     -2.562      0.010      -0.120      -0.016
    x4             0.6161      0.059     10.487      0.000       0.501       0.731
    x5            -0.5329      0.058     -9.117      0.000      -0.647      -0.418
    x6            -0.0092      0.026     -0.355      0.723      -0.060       0.042
    x7             0.0900      0.022      4.137      0.000       0.047       0.133
    ==============================================================================
    Omnibus:                    56890.026   Durbin-Watson:                   2.001
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         20280640.565
    Skew:                           8.577   Prob(JB):                         0.00
    Kurtosis:                     112.244   Cond. No.                         5.91
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.48
    ----- Test ------
    LREG Root mean squared error: 4.65
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 3.22
    [17:47:46] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     1437.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:48   Log-Likelihood:            -1.1676e+05
    No. Observations:                   39803   AIC:                         2.335e+05
    Df Residuals:                       39795   BIC:                         2.336e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8339      0.023     36.584      0.000       0.789       0.879
    x1             2.8859      0.034     83.743      0.000       2.818       2.953
    x2             0.9998      0.035     28.844      0.000       0.932       1.068
    x3            -0.0450      0.028     -1.629      0.103      -0.099       0.009
    x4             0.5398      0.060      9.048      0.000       0.423       0.657
    x5            -0.4835      0.060     -8.125      0.000      -0.600      -0.367
    x6            -0.0401      0.026     -1.523      0.128      -0.092       0.012
    x7             0.1403      0.026      5.397      0.000       0.089       0.191
    ==============================================================================
    Omnibus:                    58190.167   Durbin-Watson:                   2.013
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23469276.984
    Skew:                           8.925   Prob(JB):                         0.00
    Kurtosis:                     120.612   Cond. No.                         5.91
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.55
    ----- Test ------
    LREG Root mean squared error: 4.37
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 3.12
    [17:47:48] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     1440.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:49   Log-Likelihood:            -1.1659e+05
    No. Observations:                   39803   AIC:                         2.332e+05
    Df Residuals:                       39795   BIC:                         2.333e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8352      0.023     36.799      0.000       0.791       0.880
    x1             2.8573      0.034     83.424      0.000       2.790       2.924
    x2             0.9845      0.034     28.599      0.000       0.917       1.052
    x3            -0.0580      0.026     -2.214      0.027      -0.109      -0.007
    x4             0.5667      0.059      9.571      0.000       0.451       0.683
    x5            -0.4814      0.059     -8.129      0.000      -0.598      -0.365
    x6            -0.0264      0.026     -1.005      0.315      -0.078       0.025
    x7             0.1009      0.022      4.535      0.000       0.057       0.145
    ==============================================================================
    Omnibus:                    57271.856   Durbin-Watson:                   1.988
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         20909111.663
    Skew:                           8.685   Prob(JB):                         0.00
    Kurtosis:                     113.932   Cond. No.                         5.89
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.53
    ----- Test ------
    LREG Root mean squared error: 4.45
    ----- Training ------
    Root mean squared error: 2.73
    ----- Test ------
    Root mean squared error: 3.20
    [17:47:49] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.202
    Model:                                OLS   Adj. R-squared:                  0.202
    Method:                     Least Squares   F-statistic:                     1436.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:50   Log-Likelihood:            -1.1641e+05
    No. Observations:                   39803   AIC:                         2.328e+05
    Df Residuals:                       39795   BIC:                         2.329e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8313      0.023     36.798      0.000       0.787       0.876
    x1             2.8584      0.034     83.706      0.000       2.791       2.925
    x2             0.9867      0.034     28.773      0.000       0.919       1.054
    x3            -0.0525      0.026     -2.019      0.043      -0.103      -0.002
    x4             0.5118      0.059      8.701      0.000       0.396       0.627
    x5            -0.4464      0.059     -7.605      0.000      -0.561      -0.331
    x6            -0.0295      0.026     -1.134      0.257      -0.081       0.022
    x7             0.1449      0.025      5.690      0.000       0.095       0.195
    ==============================================================================
    Omnibus:                    57954.841   Durbin-Watson:                   2.008
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22890744.384
    Skew:                           8.860   Prob(JB):                         0.00
    Kurtosis:                     119.140   Cond. No.                         5.88
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.51
    ----- Test ------
    LREG Root mean squared error: 4.53
    ----- Training ------
    Root mean squared error: 2.74
    ----- Test ------
    Root mean squared error: 3.28
    [17:47:51] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                  OLS Regression Results                              
    ==================================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                       0.201
    Model:                                OLS   Adj. R-squared:                  0.201
    Method:                     Least Squares   F-statistic:                     1433.
    Date:                    Thu, 16 Feb 2023   Prob (F-statistic):               0.00
    Time:                            17:47:52   Log-Likelihood:            -1.1655e+05
    No. Observations:                   39803   AIC:                         2.331e+05
    Df Residuals:                       39795   BIC:                         2.332e+05
    Df Model:                               7                                         
    Covariance Type:                nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8333      0.023     36.750      0.000       0.789       0.878
    x1             2.8461      0.034     83.186      0.000       2.779       2.913
    x2             0.9772      0.034     28.396      0.000       0.910       1.045
    x3            -0.0463      0.026     -1.799      0.072      -0.097       0.004
    x4             0.5497      0.059      9.269      0.000       0.433       0.666
    x5            -0.4734      0.059     -8.001      0.000      -0.589      -0.357
    x6            -0.0411      0.026     -1.569      0.117      -0.092       0.010
    x7             0.1065      0.022      4.774      0.000       0.063       0.150
    ==============================================================================
    Omnibus:                    57671.674   Durbin-Watson:                   2.018
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22076461.560
    Skew:                           8.787   Prob(JB):                         0.00
    Kurtosis:                     117.029   Cond. No.                         5.90
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 4.52
    ----- Test ------
    LREG Root mean squared error: 4.47
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 3.23



```python
# Define a function to plot RMSEs
def rmse_bin_plot(te_rmse, tr_rmse, min_rg, max_rg, step):

    m_test_rmse = statistics.mean(te_rmse)
    plt.figure(figsize=(4, 3))
    plt.axvline(m_test_rmse, color="red", linestyle="dashed")
    plt.hist(
        te_rmse,
        bins=np.arange(min_rg, max_rg, step),
        edgecolor="k",
        histtype="bar",
        density=True,
    )
    sd_test_rmse = statistics.stdev(te_rmse)

    m_train_rmse = statistics.mean(tr_rmse)
    plt.axvline(m_train_rmse, color="b", linestyle="dashed")
    plt.hist(
        tr_rmse,
        bins=np.arange(min_rg, max_rg, step),
        color="orange",
        edgecolor="k",
        histtype="bar",
        density=True,
        alpha=0.7,
    )
    sd_train_rmse = statistics.stdev(tr_rmse)

    print(Fore.RED)
    print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
    print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

    print(f"mean_RMSE_test: {m_test_rmse:.2f}")
    print(f"mean_RMSE_train: {m_train_rmse:.2f}")

    # create legend
    labels = ["Mean_test", "Mean_train", "test", "train"]
    plt.legend(labels)

    plt.xlabel("The RMSE error")
    plt.ylabel("Frequency")
    plt.title("histogram of the RMSE distribution")
    plt.show()
```

## Plot RMSE in total


```python
print("RMSE in total", "\n")
rmse_bin_plot(test_RMSE["all"], train_RMSE["all"], 2.5, 4.0, 0.1)
```

    RMSE in total 
    
    [31m
    stdev_RMSE_test: 0.15
    stdev_RMSE_train: 0.04
    mean_RMSE_test: 3.28
    mean_RMSE_train: 2.70



    
![png](output_24_1.png)
    


## Plot RMSE per bin


```python
bin_params = {
    1: (0.0, 1.0, 0.07),
    2: (1.0, 2.0, 0.07),
    3: (4.0, 5.0, 0.07),
    4: (13.0, 15.0, 0.13),
    5: (28.0, 38.0, 0.7),
}


for bin_num in range(1, 6):

    print(f"RMSE per bin {bin_num}\n")
    rmse_bin_plot(test_RMSE[bin_num], train_RMSE[bin_num], *bin_params[bin_num])
```

    RMSE per bin 1
    
    [31m
    stdev_RMSE_test: 0.18
    stdev_RMSE_train: 0.02
    mean_RMSE_test: 0.96
    mean_RMSE_train: 0.79



    
![png](output_26_1.png)
    


    RMSE per bin 2
    
    [31m
    stdev_RMSE_test: 0.23
    stdev_RMSE_train: 0.04
    mean_RMSE_test: 1.96
    mean_RMSE_train: 1.67



    
![png](output_26_3.png)
    


    RMSE per bin 3
    
    [31m
    stdev_RMSE_test: 0.47
    stdev_RMSE_train: 0.08
    mean_RMSE_test: 4.83
    mean_RMSE_train: 4.00



    
![png](output_26_5.png)
    


    RMSE per bin 4
    
    [31m
    stdev_RMSE_test: 0.95
    stdev_RMSE_train: 0.19
    mean_RMSE_test: 14.74
    mean_RMSE_train: 13.16



    
![png](output_26_7.png)
    


    RMSE per bin 5
    
    [31m
    stdev_RMSE_test: 3.47
    stdev_RMSE_train: 0.98
    mean_RMSE_test: 37.88
    mean_RMSE_train: 28.68



    
![png](output_26_9.png)
    



```python

```

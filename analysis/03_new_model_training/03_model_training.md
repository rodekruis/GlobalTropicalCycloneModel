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
# Read csv file and import to df
df = get_training_dataset()
# df
```


```python
# Show histogram of damage
df.hist(column="percent_buildings_damaged", figsize=(4, 3))
```




    array([[<AxesSubplot:title={'center':'percent_buildings_damaged'}>]],
          dtype=object)




    
![png](output_4_1.png)
    



```python
# Hist plot after data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_buildings_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```




    [<matplotlib.lines.Line2D at 0x7fabf0606af0>]




    
![png](output_5_1.png)
    



```python
# Check the bins' intervalls (first bin means all zeros, second bin means 0 < values <= 1)
df["percent_buildings_damaged"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    88370
    (9e-05, 1.0]        4741
    (1.0, 10.0]         2616
    (10.0, 50.0]        1410
    (50.0, 101.0]       1377
    Name: percent_buildings_damaged, dtype: int64




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
      <th>total_buildings</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>percent_buildings_damaged</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>48</th>
      <td>DURIAN</td>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>35.0</td>
      <td>0.716667</td>
      <td>0.424479</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>DURIAN</td>
      <td>12.511864</td>
      <td>273.639330</td>
      <td>179.0</td>
      <td>0.568750</td>
      <td>0.336979</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50</th>
      <td>DURIAN</td>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>44.0</td>
      <td>0.589583</td>
      <td>0.290625</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>51</th>
      <td>DURIAN</td>
      <td>11.471921</td>
      <td>295.721263</td>
      <td>14.0</td>
      <td>0.620833</td>
      <td>0.301042</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>DURIAN</td>
      <td>14.394863</td>
      <td>239.279840</td>
      <td>5.0</td>
      <td>2.464583</td>
      <td>1.222917</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Hist plot after removing rows where windspeed is 0
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_buildings_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```




    [<matplotlib.lines.Line2D at 0x7fabf076ee20>]




    
![png](output_8_1.png)
    



```python
print(samples_per_bin2)
print(binsP2)
```

    [31812  4232  2501  1351  1348]
    [0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]



```python
# Check the bins' intervalls
df["percent_buildings_damaged"].value_counts(bins=binsP2)
```




    (-0.001, 9e-05]    31812
    (9e-05, 1.0]        4232
    (1.0, 10.0]         2501
    (10.0, 50.0]        1351
    (50.0, 101.0]       1348
    Name: percent_buildings_damaged, dtype: int64




```python
bin_index2 = np.digitize(df["percent_buildings_damaged"], bins=binsP2)
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
    "total_buildings",
    "rainfall_max_6h",
    "rainfall_max_24h",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["percent_buildings_damaged"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```


    Index(['wind_speed', 'track_distance', 'total_buildings', 'rainfall_max_6h',
           'rainfall_max_24h'],
          dtype='object')



```python
# Run XGBoost Reduced Overfitting in for loop to estimate RMSE per bins

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df["percent_buildings_damaged"], stratify=y_input_strat, test_size=0.2
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

    [16:58:00] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.228
    Model:                                   OLS   Adj. R-squared:                  0.228
    Method:                        Least Squares   F-statistic:                     1945.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:00   Log-Likelihood:            -1.3575e+05
    No. Observations:                      32995   AIC:                         2.715e+05
    Df Residuals:                          32989   BIC:                         2.716e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0352      0.082     49.493      0.000       3.875       4.195
    x1             8.9871      0.123     73.009      0.000       8.746       9.228
    x2             2.4066      0.124     19.441      0.000       2.164       2.649
    x3            -0.5253      0.082     -6.385      0.000      -0.687      -0.364
    x4             2.4151      0.211     11.428      0.000       2.001       2.829
    x5            -1.0932      0.211     -5.184      0.000      -1.506      -0.680
    ==============================================================================
    Omnibus:                    26232.251   Durbin-Watson:                   1.993
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           547524.870
    Skew:                           3.808   Prob(JB):                         0.00
    Kurtosis:                      21.446   Cond. No.                         5.85
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.81
    ----- Test ------
    LREG Root mean squared error: 14.85
    ----- Training ------
    Root mean squared error: 12.35
    ----- Test ------
    Root mean squared error: 13.27
    [16:58:01] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.232
    Model:                                   OLS   Adj. R-squared:                  0.232
    Method:                        Least Squares   F-statistic:                     1994.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:01   Log-Likelihood:            -1.3577e+05
    No. Observations:                      32995   AIC:                         2.715e+05
    Df Residuals:                          32989   BIC:                         2.716e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0821      0.082     50.034      0.000       3.922       4.242
    x1             9.2169      0.123     74.860      0.000       8.976       9.458
    x2             2.5219      0.124     20.362      0.000       2.279       2.765
    x3            -0.5424      0.082     -6.593      0.000      -0.704      -0.381
    x4             2.4286      0.211     11.486      0.000       2.014       2.843
    x5            -1.1907      0.212     -5.625      0.000      -1.606      -0.776
    ==============================================================================
    Omnibus:                    26017.316   Durbin-Watson:                   1.989
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           532219.546
    Skew:                           3.768   Prob(JB):                         0.00
    Kurtosis:                      21.175   Cond. No.                         5.84
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.82
    ----- Test ------
    LREG Root mean squared error: 14.81
    ----- Training ------
    Root mean squared error: 12.38
    ----- Test ------
    Root mean squared error: 13.23
    [16:58:02] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.229
    Model:                                   OLS   Adj. R-squared:                  0.228
    Method:                        Least Squares   F-statistic:                     1955.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:02   Log-Likelihood:            -1.3582e+05
    No. Observations:                      32995   AIC:                         2.717e+05
    Df Residuals:                          32989   BIC:                         2.717e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0412      0.082     49.452      0.000       3.881       4.201
    x1             9.1312      0.123     73.997      0.000       8.889       9.373
    x2             2.4662      0.124     19.923      0.000       2.224       2.709
    x3            -0.5307      0.083     -6.388      0.000      -0.694      -0.368
    x4             2.3509      0.211     11.153      0.000       1.938       2.764
    x5            -1.1475      0.211     -5.443      0.000      -1.561      -0.734
    ==============================================================================
    Omnibus:                    26152.885   Durbin-Watson:                   1.999
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           540464.639
    Skew:                           3.794   Prob(JB):                         0.00
    Kurtosis:                      21.318   Cond. No.                         5.83
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.84
    ----- Test ------
    LREG Root mean squared error: 14.71
    ----- Training ------
    Root mean squared error: 12.39
    ----- Test ------
    Root mean squared error: 13.33
    [16:58:03] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.230
    Model:                                   OLS   Adj. R-squared:                  0.230
    Method:                        Least Squares   F-statistic:                     1976.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:04   Log-Likelihood:            -1.3571e+05
    No. Observations:                      32995   AIC:                         2.714e+05
    Df Residuals:                          32989   BIC:                         2.715e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0262      0.081     49.432      0.000       3.867       4.186
    x1             9.1555      0.123     74.728      0.000       8.915       9.396
    x2             2.4806      0.124     20.058      0.000       2.238       2.723
    x3            -0.4891      0.081     -6.036      0.000      -0.648      -0.330
    x4             2.4185      0.210     11.527      0.000       2.007       2.830
    x5            -1.2824      0.210     -6.105      0.000      -1.694      -0.871
    ==============================================================================
    Omnibus:                    26038.158   Durbin-Watson:                   1.991
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           533223.468
    Skew:                           3.773   Prob(JB):                         0.00
    Kurtosis:                      21.191   Cond. No.                         5.82
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.79
    ----- Test ------
    LREG Root mean squared error: 14.91
    ----- Training ------
    Root mean squared error: 12.37
    ----- Test ------
    Root mean squared error: 13.36
    [16:58:04] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.225
    Model:                                   OLS   Adj. R-squared:                  0.225
    Method:                        Least Squares   F-statistic:                     1913.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:05   Log-Likelihood:            -1.3583e+05
    No. Observations:                      32995   AIC:                         2.717e+05
    Df Residuals:                          32989   BIC:                         2.717e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0683      0.082     49.776      0.000       3.908       4.228
    x1             8.9505      0.124     72.414      0.000       8.708       9.193
    x2             2.4025      0.124     19.416      0.000       2.160       2.645
    x3            -0.5035      0.082     -6.166      0.000      -0.663      -0.343
    x4             2.5301      0.212     11.915      0.000       2.114       2.946
    x5            -1.1873      0.212     -5.595      0.000      -1.603      -0.771
    ==============================================================================
    Omnibus:                    26182.033   Durbin-Watson:                   1.996
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           541034.897
    Skew:                           3.801   Prob(JB):                         0.00
    Kurtosis:                      21.324   Cond. No.                         5.85
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.84
    ----- Test ------
    LREG Root mean squared error: 14.71
    ----- Training ------
    Root mean squared error: 12.41
    ----- Test ------
    Root mean squared error: 13.08
    [16:58:05] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.227
    Model:                                   OLS   Adj. R-squared:                  0.227
    Method:                        Least Squares   F-statistic:                     1943.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:06   Log-Likelihood:            -1.3572e+05
    No. Observations:                      32995   AIC:                         2.714e+05
    Df Residuals:                          32989   BIC:                         2.715e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0188      0.081     49.336      0.000       3.859       4.178
    x1             9.0335      0.123     73.568      0.000       8.793       9.274
    x2             2.4177      0.124     19.566      0.000       2.176       2.660
    x3            -0.5063      0.082     -6.164      0.000      -0.667      -0.345
    x4             2.5930      0.212     12.218      0.000       2.177       3.009
    x5            -1.3962      0.212     -6.601      0.000      -1.811      -0.982
    ==============================================================================
    Omnibus:                    26259.332   Durbin-Watson:                   1.986
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           550381.644
    Skew:                           3.812   Prob(JB):                         0.00
    Kurtosis:                      21.499   Cond. No.                         5.87
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.80
    ----- Test ------
    LREG Root mean squared error: 14.91
    ----- Training ------
    Root mean squared error: 12.36
    ----- Test ------
    Root mean squared error: 13.42
    [16:58:06] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.227
    Model:                                   OLS   Adj. R-squared:                  0.227
    Method:                        Least Squares   F-statistic:                     1939.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:07   Log-Likelihood:            -1.3572e+05
    No. Observations:                      32995   AIC:                         2.714e+05
    Df Residuals:                          32989   BIC:                         2.715e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0306      0.081     49.476      0.000       3.871       4.190
    x1             9.0360      0.123     73.651      0.000       8.796       9.276
    x2             2.3680      0.123     19.196      0.000       2.126       2.610
    x3            -0.5334      0.082     -6.516      0.000      -0.694      -0.373
    x4             2.4510      0.210     11.645      0.000       2.038       2.864
    x5            -1.3062      0.210     -6.218      0.000      -1.718      -0.894
    ==============================================================================
    Omnibus:                    26256.045   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           550343.438
    Skew:                           3.811   Prob(JB):                         0.00
    Kurtosis:                      21.499   Cond. No.                         5.82
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.80
    ----- Test ------
    LREG Root mean squared error: 14.90
    ----- Training ------
    Root mean squared error: 12.36
    ----- Test ------
    Root mean squared error: 13.34
    [16:58:07] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.230
    Model:                                   OLS   Adj. R-squared:                  0.230
    Method:                        Least Squares   F-statistic:                     1975.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:08   Log-Likelihood:            -1.3566e+05
    No. Observations:                      32995   AIC:                         2.713e+05
    Df Residuals:                          32989   BIC:                         2.714e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0467      0.081     49.763      0.000       3.887       4.206
    x1             9.0748      0.122     74.145      0.000       8.835       9.315
    x2             2.4007      0.123     19.508      0.000       2.159       2.642
    x3            -0.5137      0.081     -6.306      0.000      -0.673      -0.354
    x4             2.5377      0.210     12.082      0.000       2.126       2.949
    x5            -1.3256      0.210     -6.322      0.000      -1.737      -0.915
    ==============================================================================
    Omnibus:                    26129.861   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           541594.605
    Skew:                           3.788   Prob(JB):                         0.00
    Kurtosis:                      21.346   Cond. No.                         5.81
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.77
    ----- Test ------
    LREG Root mean squared error: 15.00
    ----- Training ------
    Root mean squared error: 12.34
    ----- Test ------
    Root mean squared error: 13.53
    [16:58:08] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.231
    Model:                                   OLS   Adj. R-squared:                  0.231
    Method:                        Least Squares   F-statistic:                     1986.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:09   Log-Likelihood:            -1.3566e+05
    No. Observations:                      32995   AIC:                         2.713e+05
    Df Residuals:                          32989   BIC:                         2.714e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0441      0.081     49.729      0.000       3.885       4.204
    x1             9.1844      0.123     74.766      0.000       8.944       9.425
    x2             2.4498      0.124     19.830      0.000       2.208       2.692
    x3            -0.5288      0.079     -6.673      0.000      -0.684      -0.374
    x4             2.2941      0.210     10.946      0.000       1.883       2.705
    x5            -1.1823      0.209     -5.644      0.000      -1.593      -0.772
    ==============================================================================
    Omnibus:                    25985.172   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           529895.166
    Skew:                           3.763   Prob(JB):                         0.00
    Kurtosis:                      21.133   Cond. No.                         5.82
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.77
    ----- Test ------
    LREG Root mean squared error: 15.00
    ----- Training ------
    Root mean squared error: 12.28
    ----- Test ------
    Root mean squared error: 13.55
    [16:58:09] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.231
    Model:                                   OLS   Adj. R-squared:                  0.231
    Method:                        Least Squares   F-statistic:                     1980.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:10   Log-Likelihood:            -1.3572e+05
    No. Observations:                      32995   AIC:                         2.715e+05
    Df Residuals:                          32989   BIC:                         2.715e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0319      0.081     49.492      0.000       3.872       4.192
    x1             9.1504      0.123     74.329      0.000       8.909       9.392
    x2             2.4688      0.124     19.971      0.000       2.227       2.711
    x3            -0.5086      0.080     -6.358      0.000      -0.665      -0.352
    x4             2.5247      0.210     12.005      0.000       2.112       2.937
    x5            -1.3248      0.210     -6.308      0.000      -1.737      -0.913
    ==============================================================================
    Omnibus:                    26072.749   Durbin-Watson:                   2.011
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           537466.543
    Skew:                           3.777   Prob(JB):                         0.00
    Kurtosis:                      21.272   Cond. No.                         5.83
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.80
    ----- Test ------
    LREG Root mean squared error: 14.90
    ----- Training ------
    Root mean squared error: 12.38
    ----- Test ------
    Root mean squared error: 13.43
    [16:58:10] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.237
    Model:                                   OLS   Adj. R-squared:                  0.236
    Method:                        Least Squares   F-statistic:                     2044.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:11   Log-Likelihood:            -1.3560e+05
    No. Observations:                      32995   AIC:                         2.712e+05
    Df Residuals:                          32989   BIC:                         2.713e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0563      0.081     49.972      0.000       3.897       4.215
    x1             9.3250      0.122     76.157      0.000       9.085       9.565
    x2             2.5322      0.123     20.535      0.000       2.291       2.774
    x3            -0.5338      0.080     -6.643      0.000      -0.691      -0.376
    x4             2.5949      0.210     12.361      0.000       2.183       3.006
    x5            -1.4749      0.209     -7.053      0.000      -1.885      -1.065
    ==============================================================================
    Omnibus:                    25978.719   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           534208.200
    Skew:                           3.757   Prob(JB):                         0.00
    Kurtosis:                      21.224   Cond. No.                         5.83
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.74
    ----- Test ------
    LREG Root mean squared error: 15.11
    ----- Training ------
    Root mean squared error: 12.31
    ----- Test ------
    Root mean squared error: 13.80
    [16:58:11] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.230
    Model:                                   OLS   Adj. R-squared:                  0.230
    Method:                        Least Squares   F-statistic:                     1970.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:12   Log-Likelihood:            -1.3567e+05
    No. Observations:                      32995   AIC:                         2.714e+05
    Df Residuals:                          32989   BIC:                         2.714e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0268      0.081     49.499      0.000       3.867       4.186
    x1             9.0643      0.122     74.130      0.000       8.825       9.304
    x2             2.4492      0.123     19.894      0.000       2.208       2.691
    x3            -0.5300      0.081     -6.515      0.000      -0.689      -0.371
    x4             2.5994      0.211     12.332      0.000       2.186       3.013
    x5            -1.3564      0.211     -6.440      0.000      -1.769      -0.944
    ==============================================================================
    Omnibus:                    26134.837   Durbin-Watson:                   1.995
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           541663.451
    Skew:                           3.789   Prob(JB):                         0.00
    Kurtosis:                      21.346   Cond. No.                         5.84
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.78
    ----- Test ------
    LREG Root mean squared error: 14.98
    ----- Training ------
    Root mean squared error: 12.36
    ----- Test ------
    Root mean squared error: 13.37
    [16:58:12] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.227
    Model:                                   OLS   Adj. R-squared:                  0.227
    Method:                        Least Squares   F-statistic:                     1935.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:13   Log-Likelihood:            -1.3585e+05
    No. Observations:                      32995   AIC:                         2.717e+05
    Df Residuals:                          32989   BIC:                         2.718e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0619      0.082     49.659      0.000       3.902       4.222
    x1             9.0747      0.124     73.402      0.000       8.832       9.317
    x2             2.4018      0.124     19.331      0.000       2.158       2.645
    x3            -0.5523      0.082     -6.770      0.000      -0.712      -0.392
    x4             2.5258      0.211     11.979      0.000       2.113       2.939
    x5            -1.3745      0.211     -6.529      0.000      -1.787      -0.962
    ==============================================================================
    Omnibus:                    26120.578   Durbin-Watson:                   2.004
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           537816.662
    Skew:                           3.789   Prob(JB):                         0.00
    Kurtosis:                      21.270   Cond. No.                         5.80
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.86
    ----- Test ------
    LREG Root mean squared error: 14.66
    ----- Training ------
    Root mean squared error: 12.41
    ----- Test ------
    Root mean squared error: 13.09
    [16:58:13] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.230
    Model:                                   OLS   Adj. R-squared:                  0.230
    Method:                        Least Squares   F-statistic:                     1969.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:14   Log-Likelihood:            -1.3566e+05
    No. Observations:                      32995   AIC:                         2.713e+05
    Df Residuals:                          32989   BIC:                         2.714e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0598      0.081     49.926      0.000       3.900       4.219
    x1             9.1233      0.122     74.697      0.000       8.884       9.363
    x2             2.3590      0.123     19.164      0.000       2.118       2.600
    x3            -0.5262      0.080     -6.548      0.000      -0.684      -0.369
    x4             2.2706      0.210     10.808      0.000       1.859       2.682
    x5            -1.2184      0.209     -5.827      0.000      -1.628      -0.809
    ==============================================================================
    Omnibus:                    26188.620   Durbin-Watson:                   2.014
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           545782.370
    Skew:                           3.798   Prob(JB):                         0.00
    Kurtosis:                      21.420   Cond. No.                         5.80
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.77
    ----- Test ------
    LREG Root mean squared error: 15.01
    ----- Training ------
    Root mean squared error: 12.37
    ----- Test ------
    Root mean squared error: 13.39
    [16:58:14] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.229
    Model:                                   OLS   Adj. R-squared:                  0.228
    Method:                        Least Squares   F-statistic:                     1955.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:15   Log-Likelihood:            -1.3566e+05
    No. Observations:                      32995   AIC:                         2.713e+05
    Df Residuals:                          32989   BIC:                         2.714e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0399      0.081     49.680      0.000       3.881       4.199
    x1             9.1267      0.123     74.287      0.000       8.886       9.367
    x2             2.4280      0.123     19.673      0.000       2.186       2.670
    x3            -0.5135      0.082     -6.269      0.000      -0.674      -0.353
    x4             2.3704      0.212     11.183      0.000       1.955       2.786
    x5            -1.2546      0.212     -5.927      0.000      -1.669      -0.840
    ==============================================================================
    Omnibus:                    26234.566   Durbin-Watson:                   2.003
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           549187.499
    Skew:                           3.807   Prob(JB):                         0.00
    Kurtosis:                      21.480   Cond. No.                         5.85
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.77
    ----- Test ------
    LREG Root mean squared error: 15.00
    ----- Training ------
    Root mean squared error: 12.36
    ----- Test ------
    Root mean squared error: 13.34
    [16:58:15] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.228
    Model:                                   OLS   Adj. R-squared:                  0.228
    Method:                        Least Squares   F-statistic:                     1953.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:16   Log-Likelihood:            -1.3567e+05
    No. Observations:                      32995   AIC:                         2.714e+05
    Df Residuals:                          32989   BIC:                         2.714e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0356      0.081     49.609      0.000       3.876       4.195
    x1             9.0639      0.123     73.921      0.000       8.824       9.304
    x2             2.4427      0.124     19.770      0.000       2.200       2.685
    x3            -0.5102      0.080     -6.362      0.000      -0.667      -0.353
    x4             2.4922      0.210     11.872      0.000       2.081       2.904
    x5            -1.2945      0.210     -6.163      0.000      -1.706      -0.883
    ==============================================================================
    Omnibus:                    26191.454   Durbin-Watson:                   2.008
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           545649.067
    Skew:                           3.799   Prob(JB):                         0.00
    Kurtosis:                      21.416   Cond. No.                         5.81
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.77
    ----- Test ------
    LREG Root mean squared error: 14.98
    ----- Training ------
    Root mean squared error: 12.30
    ----- Test ------
    Root mean squared error: 13.31
    [16:58:16] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.230
    Model:                                   OLS   Adj. R-squared:                  0.230
    Method:                        Least Squares   F-statistic:                     1972.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:17   Log-Likelihood:            -1.3585e+05
    No. Observations:                      32995   AIC:                         2.717e+05
    Df Residuals:                          32989   BIC:                         2.718e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0662      0.082     49.717      0.000       3.906       4.227
    x1             9.1622      0.124     74.176      0.000       8.920       9.404
    x2             2.4078      0.124     19.417      0.000       2.165       2.651
    x3            -0.5623      0.082     -6.819      0.000      -0.724      -0.401
    x4             2.4481      0.211     11.581      0.000       2.034       2.862
    x5            -1.3335      0.211     -6.315      0.000      -1.747      -0.920
    ==============================================================================
    Omnibus:                    26096.690   Durbin-Watson:                   2.005
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           537405.267
    Skew:                           3.783   Prob(JB):                         0.00
    Kurtosis:                      21.266   Cond. No.                         5.83
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.85
    ----- Test ------
    LREG Root mean squared error: 14.67
    ----- Training ------
    Root mean squared error: 12.47
    ----- Test ------
    Root mean squared error: 13.10
    [16:58:17] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.230
    Model:                                   OLS   Adj. R-squared:                  0.230
    Method:                        Least Squares   F-statistic:                     1974.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:18   Log-Likelihood:            -1.3564e+05
    No. Observations:                      32995   AIC:                         2.713e+05
    Df Residuals:                          32989   BIC:                         2.713e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0522      0.081     49.864      0.000       3.893       4.212
    x1             9.1493      0.123     74.619      0.000       8.909       9.390
    x2             2.4002      0.123     19.494      0.000       2.159       2.642
    x3            -0.5309      0.081     -6.546      0.000      -0.690      -0.372
    x4             2.3894      0.211     11.345      0.000       1.977       2.802
    x5            -1.3073      0.211     -6.205      0.000      -1.720      -0.894
    ==============================================================================
    Omnibus:                    26103.897   Durbin-Watson:                   1.991
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           539072.747
    Skew:                           3.783   Prob(JB):                         0.00
    Kurtosis:                      21.299   Cond. No.                         5.82
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.76
    ----- Test ------
    LREG Root mean squared error: 15.04
    ----- Training ------
    Root mean squared error: 12.35
    ----- Test ------
    Root mean squared error: 13.39
    [16:58:18] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.230
    Model:                                   OLS   Adj. R-squared:                  0.230
    Method:                        Least Squares   F-statistic:                     1976.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:19   Log-Likelihood:            -1.3566e+05
    No. Observations:                      32995   AIC:                         2.713e+05
    Df Residuals:                          32989   BIC:                         2.714e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0357      0.081     49.628      0.000       3.876       4.195
    x1             9.1239      0.122     74.605      0.000       8.884       9.364
    x2             2.4123      0.123     19.608      0.000       2.171       2.653
    x3            -0.5312      0.082     -6.446      0.000      -0.693      -0.370
    x4             2.3809      0.209     11.388      0.000       1.971       2.791
    x5            -1.2242      0.210     -5.842      0.000      -1.635      -0.813
    ==============================================================================
    Omnibus:                    26149.429   Durbin-Watson:                   1.992
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           544660.922
    Skew:                           3.790   Prob(JB):                         0.00
    Kurtosis:                      21.405   Cond. No.                         5.80
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.77
    ----- Test ------
    LREG Root mean squared error: 15.00
    ----- Training ------
    Root mean squared error: 12.33
    ----- Test ------
    Root mean squared error: 13.45
    [16:58:19] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work/src/learner.cc:576: 
    Parameters: { "early_stopping_rounds" } might not be used.
    
      This could be a false alarm, with some parameters getting used by language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter actually being used
      but getting flagged wrongly here. Please open an issue if you find any such cases.
    
    
                                    OLS Regression Results                               
    =====================================================================================
    Dep. Variable:     percent_buildings_damaged   R-squared:                       0.229
    Model:                                   OLS   Adj. R-squared:                  0.229
    Method:                        Least Squares   F-statistic:                     1956.
    Date:                       Thu, 19 Jan 2023   Prob (F-statistic):               0.00
    Time:                               16:58:20   Log-Likelihood:            -1.3582e+05
    No. Observations:                      32995   AIC:                         2.716e+05
    Df Residuals:                          32989   BIC:                         2.717e+05
    Df Model:                                  5                                         
    Covariance Type:                   nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          4.0617      0.082     49.711      0.000       3.902       4.222
    x1             9.1174      0.123     74.000      0.000       8.876       9.359
    x2             2.4093      0.124     19.449      0.000       2.166       2.652
    x3            -0.5401      0.081     -6.631      0.000      -0.700      -0.380
    x4             2.3576      0.210     11.218      0.000       1.946       2.770
    x5            -1.1991      0.211     -5.690      0.000      -1.612      -0.786
    ==============================================================================
    Omnibus:                    26115.989   Durbin-Watson:                   1.980
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           537879.496
    Skew:                           3.787   Prob(JB):                         0.00
    Kurtosis:                      21.272   Cond. No.                         5.82
    ==============================================================================
    
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    ----- Training ------
    LREG Root mean squared error: 14.84
    ----- Test ------
    LREG Root mean squared error: 14.72
    ----- Training ------
    Root mean squared error: 12.35
    ----- Test ------
    Root mean squared error: 13.21



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
rmse_bin_plot(test_RMSE["all"], train_RMSE["all"], 12.0, 13.5, 0.09)
```

    RMSE in total 
    
    [31m
    stdev_RMSE_test: 0.17
    stdev_RMSE_train: 0.04
    mean_RMSE_test: 13.35
    mean_RMSE_train: 12.36



    
![png](output_18_1.png)
    


## Plot RMSE per bin


```python
bin_params = {
    1: (3.5, 4.5, 0.06),
    2: (8.0, 9.0, 0.06),
    3: (12.0, 14.0, 0.15),
    4: (18.0, 21.0, 0.2),
    5: (59.0, 64.0, 0.35),
}


for bin_num in range(1, 6):

    print(f"RMSE per bin {bin_num}\n")
    rmse_bin_plot(test_RMSE[bin_num], train_RMSE[bin_num], *bin_params[bin_num])
```

    RMSE per bin 1
    
    [31m
    stdev_RMSE_test: 0.19
    stdev_RMSE_train: 0.05
    mean_RMSE_test: 4.19
    mean_RMSE_train: 3.88



    
![png](output_20_1.png)
    


    RMSE per bin 2
    
    [31m
    stdev_RMSE_test: 0.59
    stdev_RMSE_train: 0.15
    mean_RMSE_test: 8.86
    mean_RMSE_train: 8.13



    
![png](output_20_3.png)
    


    RMSE per bin 3
    
    [31m
    stdev_RMSE_test: 0.75
    stdev_RMSE_train: 0.17
    mean_RMSE_test: 13.52
    mean_RMSE_train: 12.21



    
![png](output_20_5.png)
    


    RMSE per bin 4
    
    [31m
    stdev_RMSE_test: 0.87
    stdev_RMSE_train: 0.23
    mean_RMSE_test: 19.75
    mean_RMSE_train: 18.21



    
![png](output_20_7.png)
    


    RMSE per bin 5
    
    [31m
    stdev_RMSE_test: 1.07
    stdev_RMSE_train: 0.26
    mean_RMSE_test: 63.65
    mean_RMSE_train: 59.19



    
![png](output_20_9.png)
    



```python

```

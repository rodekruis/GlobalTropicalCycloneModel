# Run XGBoost Regression Model for Municipality_based dataset

```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
import pandas as pd
import numpy as np
import statistics
import os

from sklearn.metrics import f1_score, mean_squared_error, mean_absolute_error, max_error
from collections import defaultdict
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
```

```python
# Import the CSV file to a dataframe
df = pd.read_csv("data/df_merged_2.csv")

# Remove the duplicated rows
df.drop_duplicates(keep="first", inplace=True)
df = df.reset_index(drop=True)

df
```

```python
# display(combined_input_data)
df.hist(column="y_norm", figsize=(4, 3))
```

array([[<AxesSubplot:title={'center':'y_norm'}>]], dtype=object)

![png](output_4_1.png)

```python
# The Old and New set of bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
plt.figure(figsize=(4, 3))
samples_per_bin2, binsP2 = np.histogram(df["y_norm"], bins=bins2)
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

[<matplotlib.lines.Line2D at 0x7f9762088a00>]

![png](output_5_1.png)

```python
print(samples_per_bin2)
print(binsP2)
```

[3605 2721 1086  384   64]
[0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]

```python
df["y_norm"].value_counts(bins=binsP2)
```

(-0.001, 9e-05]    3605
(9e-05, 1.0]       2721
(1.0, 10.0]        1086
(10.0, 50.0]        384
(50.0, 101.0]        64
Name: y_norm, dtype: int64

```python
bin_index2 = np.digitize(df["y_norm"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
all_features = [
    "HAZ_rainfall_Total",
    "HAZ_rainfall_max_6h",
    "HAZ_rainfall_max_24h",
    "HAZ_v_max",
    "HAZ_v_max_3",
    "HAZ_dis_track_min",
    "GEN_landslide_per",
    "GEN_stormsurge_per",
    #'GEN_Bu_p_inSSA',
    #'GEN_Bu_p_LS',
    "GEN_Red_per_LSbldg",
    "GEN_Or_per_LSblg",
    "GEN_Yel_per_LSSAb",
    "GEN_RED_per_SSAbldg",
    #'GEN_OR_per_SSAbldg',
    "GEN_Yellow_per_LSbl",
    "TOP_mean_slope",
    "TOP_mean_elevation_m",
    "TOP_ruggedness_stdev",
    #'TOP_mean_ruggedness',
    #'TOP_slope_stdev',
    "VUL_poverty_perc",
    "GEN_with_coast",
    "GEN_coast_length",
    "VUL_Housing_Units",
    "VUL_StrongRoof_StrongWall",
    "VUL_StrongRoof_LightWall",
    "VUL_StrongRoof_SalvageWall",
    "VUL_LightRoof_StrongWall",
    "VUL_LightRoof_LightWall",
    "VUL_LightRoof_SalvageWall",
    "VUL_SalvagedRoof_StrongWall",
    "VUL_SalvagedRoof_LightWall",
    "VUL_SalvagedRoof_SalvageWall",
    "VUL_vulnerable_groups",
    "VUL_pantawid_pamilya_beneficiary",
]


glob_features = [  #'HAZ_rainfall_Total',
    "HAZ_rainfall_max_6h",
    "HAZ_rainfall_max_24h",
    "HAZ_v_max",
    #'HAZ_v_max_3',
    "HAZ_dis_track_min",
    #'GEN_landslide_per',
    #'GEN_stormsurge_per',
    #'GEN_Bu_p_inSSA',
    #'GEN_Bu_p_LS',
    #'GEN_Red_per_LSbldg',
    #'GEN_Or_per_LSblg',
    #'GEN_Yel_per_LSSAb',
    #'GEN_RED_per_SSAbldg',
    #'GEN_OR_per_SSAbldg',
    #'GEN_Yellow_per_LSbl',
    "TOP_mean_slope",
    "TOP_mean_elevation_m",
    "TOP_ruggedness_stdev",
    "TOP_mean_ruggedness",
    "TOP_slope_stdev",
    #'VUL_poverty_perc',
    "GEN_with_coast",
    "GEN_coast_length",
    "VUL_Housing_Units",
    #'VUL_StrongRoof_StrongWall',
    #'VUL_StrongRoof_LightWall',
    #'VUL_StrongRoof_SalvageWall',
    #'VUL_LightRoof_StrongWall',
    #'VUL_LightRoof_LightWall',
    #'VUL_LightRoof_SalvageWall',
    #'VUL_SalvagedRoof_StrongWall',
    #'VUL_SalvagedRoof_LightWall',
    #'VUL_SalvagedRoof_SalvageWall',
    #'VUL_vulnerable_groups',
    #'VUL_pantawid_pamilya_beneficiary',
]
```

```python
# Defin two lists to save total RMSE of test and train data

test_RMSE = defaultdict(list)
train_RMSE = defaultdict(list)

test_AVE = defaultdict(list)
```

```python
# Ask the user whether to use all features set or global features set
feature_set = int(input("Enter 1 for all features, 2 for global features: "))
```

Enter 1 for all features, 2 for global features: 1

```python
if feature_set == 1:
    features = all_features
    print(len(features))

elif feature_set == 2:
    features = glob_features
    print(len(features))

else:
    print("Invalid input. Please enter 1 or 2")
```

31

```python
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["y_norm"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```

Index(['HAZ_rainfall_Total', 'HAZ_rainfall_max_6h', 'HAZ_rainfall_max_24h',
        'HAZ_v_max', 'HAZ_v_max_3', 'HAZ_dis_track_min', 'GEN_landslide_per',
        'GEN_stormsurge_per', 'GEN_Red_per_LSbldg', 'GEN_Or_per_LSblg',
        'GEN_Yel_per_LSSAb', 'GEN_RED_per_SSAbldg', 'GEN_Yellow_per_LSbl',
        'TOP_mean_slope', 'TOP_mean_elevation_m', 'TOP_ruggedness_stdev',
        'VUL_poverty_perc', 'GEN_with_coast', 'GEN_coast_length',
        'VUL_Housing_Units', 'VUL_StrongRoof_StrongWall',
        'VUL_StrongRoof_LightWall', 'VUL_StrongRoof_SalvageWall',
        'VUL_LightRoof_StrongWall', 'VUL_LightRoof_LightWall',
        'VUL_LightRoof_SalvageWall', 'VUL_SalvagedRoof_StrongWall',
        'VUL_SalvagedRoof_LightWall', 'VUL_SalvagedRoof_SalvageWall',
        'VUL_vulnerable_groups', 'VUL_pantawid_pamilya_beneficiary'],
      dtype='object')

```python
# Run XGBoost Reduced Overfitting in for loop to estimate RMSE per bins

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df["y_norm"], stratify=y_input_strat, test_size=0.2
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

    eval_set = [(X_train, y_train)]
    xgb_model = xgb.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
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

    # Clip the predicted values to be within the range of zero to 100
    y_pred_train_clipped = y_pred_train.clip(0, 100)
    mse_train_idx = mean_squared_error(y_train, y_pred_train_clipped)
    rmse_train = np.sqrt(mse_train_idx)

    y_pred = xgb.predict(X_test)

    # Clip the predicted values to be within the range of zero to 100
    y_pred_clipped = y_pred.clip(0, 100)
    mse_idx = mean_squared_error(y_test, y_pred_clipped)
    rmse = np.sqrt(mse_idx)

    print("----- Training ------")
    print(f"Root mean squared error: {rmse_train:.2f}")

    print("----- Test ------")
    print(f"Root mean squared error: {rmse:.2f}")

    test_RMSE["all"].append(rmse)
    train_RMSE["all"].append(rmse_train)

    # Calculate Average Error in total

    ave = (y_pred_clipped - y_test).sum() / len(y_test)

    print("----- Test ------")
    print(f"Average Error: {ave:.2f}")

    test_AVE["all"].append(ave)

    # Calculate RMSE per bins

    bin_index_test = np.digitize(y_test, bins=binsP2)
    bin_index_train = np.digitize(y_train, bins=binsP2)

    for bin_num in range(1, 6):

        mse_train_idx = mean_squared_error(
            y_train[bin_index_train == bin_num],
            y_pred_train_clipped[bin_index_train == bin_num],
        )
        rmse_train = np.sqrt(mse_train_idx)

        mse_idx = mean_squared_error(
            y_test[bin_index_test == bin_num],
            y_pred_clipped[bin_index_test == bin_num],
        )
        rmse = np.sqrt(mse_idx)

        train_RMSE[bin_num].append(rmse_train)
        test_RMSE[bin_num].append(rmse)

        # Calculate Average Error per bins

        ave = (
            y_pred_clipped[bin_index_test == bin_num]
            - y_test[bin_index_test == bin_num]
        ).sum() / len(y_test[bin_index_test == bin_num])
        test_AVE[bin_num].append(ave)
```

[12:59:00] WARNING: /Users/runner/miniforge3/conda-bld/
xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.551
    Model:                            OLS   Adj. R-squared:                  0.549
    Method:                 Least Squares   F-statistic:                     247.7
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:01   Log-Likelihood:                -19512.
    No. Observations:                6288   AIC:                         3.909e+04
    Df Residuals:                    6256   BIC:                         3.930e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2568      0.068     33.120      0.000       2.123       2.390
    x1            -1.2525      0.299     -4.182      0.000      -1.840      -0.665
    x2             1.3420      0.229      5.854      0.000       0.893       1.791
    x3             0.7696      0.419      1.835      0.067      -0.053       1.592
    x4            -4.8998      0.259    -18.896      0.000      -5.408      -4.391
    x5             9.1276      0.179     50.992      0.000       8.777       9.479
    x6            -0.5336      0.157     -3.396      0.001      -0.842      -0.226
    x7           -76.3363     49.463     -1.543      0.123    -173.300      20.628
    x8             1.0345      2.492      0.415      0.678      -3.851       5.920
    x9            44.5231     28.759      1.548      0.122     -11.854     100.900
    x10            0.4926      0.376      1.312      0.190      -0.244       1.229
    x11           61.8739     40.192      1.539      0.124     -16.916     140.663
    x12           -0.9219      2.492     -0.370      0.711      -5.807       3.963
    x13           -0.1129      0.077     -1.470      0.142      -0.264       0.038
    x14            0.0866      0.215      0.403      0.687      -0.335       0.508
    x15           -0.2949      0.142     -2.080      0.038      -0.573      -0.017
    x16           -0.0621      0.148     -0.420      0.674      -0.352       0.228
    x17            0.6358      0.225      2.823      0.005       0.194       1.077
    x18            0.1231      0.096      1.289      0.197      -0.064       0.310
    x19           -0.1375      0.087     -1.583      0.114      -0.308       0.033
    x20           -0.1688      0.071     -2.365      0.018      -0.309      -0.029
    x21           -1.5634      2.293     -0.682      0.495      -6.059       2.932
    x22           -1.0893      1.689     -0.645      0.519      -4.401       2.222
    x23            0.3527      0.114      3.086      0.002       0.129       0.577
    x24           -0.0817      0.333     -0.246      0.806      -0.734       0.570
    x25           -0.9954      1.554     -0.641      0.522      -4.042       2.051
    x26            0.2839      0.104      2.731      0.006       0.080       0.488
    x27            0.1207      0.086      1.403      0.161      -0.048       0.289
    x28            0.0472      0.091      0.519      0.604      -0.131       0.226
    x29           -0.0963      0.098     -0.984      0.325      -0.288       0.095
    x30            0.5357      0.111      4.828      0.000       0.318       0.753
    x31           -0.5677      0.216     -2.623      0.009      -0.992      -0.143
    ==============================================================================
    Omnibus:                     5613.255   Durbin-Watson:                   1.953
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           393142.455
    Skew:                           3.982   Prob(JB):                         0.00
    Kurtosis:                      40.909   Cond. No.                     2.32e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.32e+03. This might indicate that
    there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.39
    ----- Test ------
    LREG Root mean squared error: 5.37
    ----- Training ------
    Root mean squared error: 2.68
    ----- Test ------
    Root mean squared error: 4.55
    ----- Test ------
    Average Error: 0.07
    [12:59:01] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.566
    Model:                            OLS   Adj. R-squared:                  0.564
    Method:                 Least Squares   F-statistic:                     263.1
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:02   Log-Likelihood:                -19451.
    No. Observations:                6288   AIC:                         3.897e+04
    Df Residuals:                    6256   BIC:                         3.918e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2617      0.068     33.507      0.000       2.129       2.394
    x1            -1.3886      0.303     -4.581      0.000      -1.983      -0.794
    x2             1.3501      0.224      6.033      0.000       0.911       1.789
    x3             0.9139      0.419      2.182      0.029       0.093       1.735
    x4            -5.0294      0.255    -19.759      0.000      -5.528      -4.530
    x5             9.2453      0.174     53.150      0.000       8.904       9.586
    x6            -0.5378      0.155     -3.477      0.001      -0.841      -0.235
    x7           -41.3176     47.777     -0.865      0.387    -134.977      52.342
    x8             0.9063      2.229      0.407      0.684      -3.464       5.276
    x9            24.1544     27.778      0.870      0.385     -30.301      78.610
    x10            0.2561      0.363      0.706      0.480      -0.455       0.967
    x11           33.4494     38.823      0.862      0.389     -42.657     109.556
    x12           -0.8264      2.229     -0.371      0.711      -5.197       3.544
    x13           -0.1213      0.078     -1.559      0.119      -0.274       0.031
    x14           -0.1098      0.212     -0.518      0.604      -0.525       0.305
    x15           -0.2161      0.140     -1.545      0.122      -0.490       0.058
    x16            0.1478      0.145      1.016      0.310      -0.137       0.433
    x17            0.6051      0.219      2.767      0.006       0.176       1.034
    x18            0.1032      0.095      1.087      0.277      -0.083       0.289
    x19           -0.2246      0.083     -2.690      0.007      -0.388      -0.061
    x20           -0.1737      0.073     -2.375      0.018      -0.317      -0.030
    x21           -1.0370      2.220     -0.467      0.640      -5.388       3.314
    x22           -0.8007      1.634     -0.490      0.624      -4.005       2.403
    x23            0.2911      0.113      2.569      0.010       0.069       0.513
    x24            0.0240      0.322      0.075      0.941      -0.606       0.654
    x25           -0.6072      1.504     -0.404      0.686      -3.556       2.341
    x26            0.2108      0.090      2.345      0.019       0.035       0.387
    x27           -0.0363      0.075     -0.484      0.628      -0.184       0.111
    x28            0.1543      0.079      1.942      0.052      -0.001       0.310
    x29            0.1012      0.096      1.057      0.290      -0.086       0.289
    x30            0.6575      0.112      5.873      0.000       0.438       0.877
    x31           -0.5577      0.207     -2.690      0.007      -0.964      -0.151
    ==============================================================================
    Omnibus:                     5606.591   Durbin-Watson:                   2.004
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           400986.676
    Skew:                           3.966   Prob(JB):                         0.00
    Kurtosis:                      41.309   Cond. No.                     2.26e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.26e+03. This might indicate that
    there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.34
    ----- Test ------
    LREG Root mean squared error: 5.57
    ----- Training ------
    Root mean squared error: 2.69
    ----- Test ------
    Root mean squared error: 4.52
    ----- Test ------
    Average Error: 0.06
    [12:59:02] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find any
      such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.560
    Model:                            OLS   Adj. R-squared:                  0.558
    Method:                 Least Squares   F-statistic:                     257.0
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:02   Log-Likelihood:                -19447.
    No. Observations:                6288   AIC:                         3.896e+04
    Df Residuals:                    6256   BIC:                         3.917e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2450      0.067     33.283      0.000       2.113       2.377
    x1            -1.2371      0.309     -4.010      0.000      -1.842      -0.632
    x2             1.7624      0.229      7.706      0.000       1.314       2.211
    x3             0.3381      0.427      0.791      0.429      -0.499       1.176
    x4            -4.8538      0.257    -18.859      0.000      -5.358      -4.349
    x5             9.0556      0.176     51.423      0.000       8.710       9.401
    x6            -0.5572      0.156     -3.578      0.000      -0.862      -0.252
    x7             2.0995     48.846      0.043      0.966     -93.655      97.854
    x8             0.0906      2.294      0.040      0.968      -4.407       4.588
    x9            -1.0388     28.400     -0.037      0.971     -56.713      54.635
    x10           -0.0914      0.370     -0.247      0.805      -0.816       0.633
    x11           -1.8595     39.691     -0.047      0.963     -79.667      75.948
    x12            0.0348      2.293      0.015      0.988      -4.460       4.530
    x13           -0.1331      0.076     -1.762      0.078      -0.281       0.015
    x14           -0.1258      0.209     -0.602      0.547      -0.535       0.284
    x15           -0.2530      0.138     -1.832      0.067      -0.524       0.018
    x16            0.0797      0.144      0.552      0.581      -0.203       0.363
    x17            0.6572      0.208      3.161      0.002       0.250       1.065
    x18            0.0415      0.095      0.439      0.661      -0.144       0.227
    x19           -0.1368      0.084     -1.631      0.103      -0.301       0.028
    x20           -0.1927      0.082     -2.351      0.019      -0.353      -0.032
    x21           -1.7559      2.355     -0.746      0.456      -6.372       2.860
    x22           -1.2052      1.733     -0.695      0.487      -4.603       2.193
    x23            0.2382      0.113      2.112      0.035       0.017       0.459
    x24           -0.0912      0.340     -0.269      0.788      -0.757       0.575
    x25           -1.0646      1.596     -0.667      0.505      -4.193       2.064
    x26            0.2034      0.091      2.226      0.026       0.024       0.383
    x27            0.0220      0.077      0.284      0.776      -0.130       0.174
    x28            0.1853      0.081      2.275      0.023       0.026       0.345
    x29            0.0485      0.098      0.497      0.619      -0.143       0.240
    x30            0.5367      0.111      4.854      0.000       0.320       0.753
    x31           -0.6127      0.193     -3.178      0.001      -0.991      -0.235
    ==============================================================================
    Omnibus:                     5290.653   Durbin-Watson:                   2.015
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           315192.975
    Skew:                           3.666   Prob(JB):                         0.00
    Kurtosis:                      36.901   Cond. No.                     2.32e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.32e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.33
    ----- Test ------
    LREG Root mean squared error: 5.58
    ----- Training ------
    Root mean squared error: 2.59
    ----- Test ------
    Root mean squared error: 4.74
    ----- Test ------
    Average Error: -0.03
    [12:59:03] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.557
    Model:                            OLS   Adj. R-squared:                  0.554
    Method:                 Least Squares   F-statistic:                     253.3
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:03   Log-Likelihood:                -19527.
    No. Observations:                6288   AIC:                         3.912e+04
    Df Residuals:                    6256   BIC:                         3.933e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2830      0.068     33.415      0.000       2.149       2.417
    x1            -1.1895      0.314     -3.786      0.000      -1.805      -0.574
    x2             1.9830      0.234      8.466      0.000       1.524       2.442
    x3             0.1765      0.439      0.402      0.688      -0.684       1.037
    x4            -4.7561      0.258    -18.460      0.000      -5.261      -4.251
    x5             9.0265      0.177     51.032      0.000       8.680       9.373
    x6            -0.4399      0.157     -2.802      0.005      -0.748      -0.132
    x7           -27.1405     48.514     -0.559      0.576    -122.245      67.964
    x8             0.7568      2.369      0.320      0.749      -3.886       5.400
    x9            15.9655     28.206      0.566      0.571     -39.327      71.258
    x10            0.1416      0.368      0.385      0.700      -0.580       0.863
    x11           21.9343     39.422      0.556      0.578     -55.346      99.214
    x12           -0.6508      2.368     -0.275      0.783      -5.293       3.992
    x13           -0.1084      0.078     -1.388      0.165      -0.261       0.045
    x14           -0.1721      0.215     -0.799      0.424      -0.594       0.250
    x15           -0.1950      0.143     -1.365      0.172      -0.475       0.085
    x16            0.0848      0.148      0.573      0.567      -0.205       0.375
    x17            0.5429      0.202      2.686      0.007       0.147       0.939
    x18            0.0657      0.095      0.688      0.491      -0.121       0.253
    x19           -0.1041      0.084     -1.243      0.214      -0.268       0.060
    x20           -0.1463      0.074     -1.982      0.047      -0.291      -0.002
    x21           -2.1443      2.334     -0.919      0.358      -6.720       2.431
    x22           -1.5058      1.719     -0.876      0.381      -4.876       1.864
    x23            0.3517      0.111      3.175      0.002       0.135       0.569
    x24           -0.0646      0.337     -0.191      0.848      -0.726       0.597
    x25           -1.2916      1.583     -0.816      0.415      -4.395       1.812
    x26            0.1162      0.091      1.284      0.199      -0.061       0.294
    x27           -0.0530      0.079     -0.668      0.504      -0.208       0.102
    x28            0.2060      0.089      2.311      0.021       0.031       0.381
    x29            0.0471      0.098      0.481      0.630      -0.145       0.239
    x30            0.6725      0.112      5.984      0.000       0.452       0.893
    x31           -0.5646      0.186     -3.039      0.002      -0.929      -0.200
    ==============================================================================
    Omnibus:                     5720.223   Durbin-Watson:                   2.026
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           431329.811
    Skew:                           4.082   Prob(JB):                         0.00
    Kurtosis:                      42.745   Cond. No.                     2.26e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.26e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.40
    ----- Test ------
    LREG Root mean squared error: 5.32
    ----- Training ------
    Root mean squared error: 2.76
    ----- Test ------
    Root mean squared error: 4.20
    ----- Test ------
    Average Error: 0.12
    [12:59:03] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.545
    Model:                            OLS   Adj. R-squared:                  0.542
    Method:                 Least Squares   F-statistic:                     241.4
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:04   Log-Likelihood:                -19624.
    No. Observations:                6288   AIC:                         3.931e+04
    Df Residuals:                    6256   BIC:                         3.953e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2614      0.069     32.598      0.000       2.125       2.397
    x1            -1.2826      0.315     -4.065      0.000      -1.901      -0.664
    x2             1.6529      0.233      7.084      0.000       1.196       2.110
    x3             0.6021      0.437      1.377      0.169      -0.255       1.459
    x4            -4.9214      0.265    -18.576      0.000      -5.441      -4.402
    x5             9.1253      0.182     50.165      0.000       8.769       9.482
    x6            -0.4826      0.161     -2.997      0.003      -0.798      -0.167
    x7           -48.7301     49.428     -0.986      0.324    -145.627      48.166
    x8             0.8343      2.485      0.336      0.737      -4.037       5.706
    x9            28.4553     28.738      0.990      0.322     -27.881      84.791
    x10            0.2806      0.376      0.746      0.456      -0.457       1.018
    x11           39.4511     40.164      0.982      0.326     -39.284     118.186
    x12           -0.7268      2.485     -0.293      0.770      -5.598       4.144
    x13           -0.1270      0.080     -1.594      0.111      -0.283       0.029
    x14            0.0091      0.217      0.042      0.967      -0.417       0.435
    x15           -0.2310      0.143     -1.616      0.106      -0.511       0.049
    x16           -0.0372      0.150     -0.249      0.804      -0.331       0.256
    x17            0.8379      0.246      3.405      0.001       0.355       1.320
    x18            0.1441      0.097      1.482      0.139      -0.047       0.335
    x19           -0.1391      0.087     -1.606      0.108      -0.309       0.031
    x20           -0.1685      0.075     -2.238      0.025      -0.316      -0.021
    x21           -1.8838      2.357     -0.799      0.424      -6.504       2.736
    x22           -1.3891      1.736     -0.800      0.424      -4.792       2.014
    x23            0.2706      0.118      2.287      0.022       0.039       0.503
    x24           -0.0873      0.340     -0.256      0.798      -0.755       0.580
    x25           -1.1966      1.597     -0.749      0.454      -4.327       1.934
    x26            0.1366      0.106      1.286      0.198      -0.072       0.345
    x27            0.0856      0.090      0.947      0.344      -0.092       0.263
    x28            0.1030      0.084      1.234      0.217      -0.061       0.267
    x29            0.0495      0.101      0.492      0.623      -0.148       0.246
    x30            0.6133      0.117      5.259      0.000       0.385       0.842
    x31           -0.7074      0.232     -3.043      0.002      -1.163      -0.252
    ==============================================================================
    Omnibus:                     5754.803   Durbin-Watson:                   1.982
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           430961.446
    Skew:                           4.127   Prob(JB):                         0.00
    Kurtosis:                      42.708   Cond. No.                     2.26e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.26e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.48
    ----- Test ------
    LREG Root mean squared error: 4.95
    ----- Training ------
    Root mean squared error: 2.70
    ----- Test ------
    Root mean squared error: 4.22
    ----- Test ------
    Average Error: 0.02
    [12:59:04] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.564
    Model:                            OLS   Adj. R-squared:                  0.562
    Method:                 Least Squares   F-statistic:                     260.9
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:05   Log-Likelihood:                -19447.
    No. Observations:                6288   AIC:                         3.896e+04
    Df Residuals:                    6256   BIC:                         3.917e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2749      0.067     33.725      0.000       2.143       2.407
    x1            -1.4250      0.303     -4.705      0.000      -2.019      -0.831
    x2             1.1788      0.228      5.169      0.000       0.732       1.626
    x3             1.0717      0.421      2.548      0.011       0.247       1.896
    x4            -4.9353      0.256    -19.263      0.000      -5.438      -4.433
    x5             9.1514      0.174     52.690      0.000       8.811       9.492
    x6            -0.5777      0.157     -3.689      0.000      -0.885      -0.271
    x7           -57.0618     49.305     -1.157      0.247    -153.717      39.594
    x8             0.3443      2.220      0.155      0.877      -4.007       4.696
    x9            33.2973     28.666      1.162      0.245     -22.898      89.493
    x10            0.3806      0.374      1.018      0.309      -0.352       1.113
    x11           46.2396     40.065      1.154      0.248     -32.301     124.780
    x12           -0.2275      2.221     -0.102      0.918      -4.582       4.127
    x13           -0.1218      0.076     -1.598      0.110      -0.271       0.028
    x14            0.0828      0.210      0.394      0.693      -0.329       0.495
    x15           -0.2316      0.140     -1.655      0.098      -0.506       0.043
    x16           -0.1101      0.146     -0.755      0.450      -0.396       0.176
    x17            0.8017      0.223      3.587      0.000       0.364       1.240
    x18            0.0907      0.094      0.965      0.335      -0.094       0.275
    x19           -0.1829      0.084     -2.171      0.030      -0.348      -0.018
    x20           -0.1245      0.072     -1.726      0.084      -0.266       0.017
    x21           -0.7503      2.273     -0.330      0.741      -5.206       3.706
    x22           -0.4297      1.674     -0.257      0.797      -3.712       2.853
    x23            0.2632      0.111      2.370      0.018       0.045       0.481
    x24            0.2214      0.330      0.671      0.502      -0.426       0.868
    x25           -0.5037      1.540     -0.327      0.744      -3.522       2.515
    x26            0.1573      0.092      1.711      0.087      -0.023       0.337
    x27            0.1457      0.078      1.861      0.063      -0.008       0.299
    x28            0.0786      0.102      0.768      0.443      -0.122       0.279
    x29            0.0228      0.095      0.239      0.811      -0.164       0.210
    x30            0.5584      0.111      5.018      0.000       0.340       0.777
    x31           -0.6916      0.213     -3.251      0.001      -1.109      -0.275
    ==============================================================================
    Omnibus:                     5547.517   Durbin-Watson:                   2.048
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           393767.319
    Skew:                           3.898   Prob(JB):                         0.00
    Kurtosis:                      40.976   Cond. No.                     2.32e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.32e+03. This might indicate that
    there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.33
    ----- Test ------
    LREG Root mean squared error: 5.59
    ----- Training ------
    Root mean squared error: 2.61
    ----- Test ------
    Root mean squared error: 4.49
    ----- Test ------
    Average Error: 0.12
    [12:59:05] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.543
    Model:                            OLS   Adj. R-squared:                  0.541
    Method:                 Least Squares   F-statistic:                     239.9
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:05   Log-Likelihood:                -19582.
    No. Observations:                6288   AIC:                         3.923e+04
    Df Residuals:                    6256   BIC:                         3.944e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2803      0.069     33.087      0.000       2.145       2.415
    x1            -1.3658      0.306     -4.457      0.000      -1.967      -0.765
    x2             1.5491      0.235      6.603      0.000       1.089       2.009
    x3             0.7525      0.431      1.747      0.081      -0.092       1.597
    x4            -4.8897      0.263    -18.590      0.000      -5.405      -4.374
    x5             9.0769      0.181     50.120      0.000       8.722       9.432
    x6            -0.4900      0.160     -3.068      0.002      -0.803      -0.177
    x7           -40.1718     49.751     -0.807      0.419    -137.701      57.357
    x8             0.2762      2.444      0.113      0.910      -4.516       5.068
    x9            23.5022     28.925      0.813      0.417     -33.201      80.205
    x10            0.2140      0.377      0.568      0.570      -0.525       0.953
    x11           32.4564     40.427      0.803      0.422     -46.794     111.706
    x12           -0.1938      2.442     -0.079      0.937      -4.980       4.593
    x13           -0.0579      0.079     -0.733      0.463      -0.213       0.097
    x14            0.0085      0.215      0.040      0.968      -0.413       0.430
    x15           -0.2803      0.141     -1.985      0.047      -0.557      -0.004
    x16           -0.0131      0.149     -0.088      0.930      -0.305       0.279
    x17            0.7171      0.209      3.431      0.001       0.307       1.127
    x18            0.1663      0.097      1.716      0.086      -0.024       0.356
    x19           -0.1948      0.089     -2.182      0.029      -0.370      -0.020
    x20           -0.1744      0.073     -2.388      0.017      -0.318      -0.031
    x21           -1.7582      2.378     -0.739      0.460      -6.419       2.903
    x22           -1.2396      1.749     -0.709      0.479      -4.669       2.190
    x23            0.3240      0.117      2.778      0.005       0.095       0.553
    x24            0.0388      0.343      0.113      0.910      -0.634       0.711
    x25           -1.1272      1.612     -0.699      0.484      -4.287       2.032
    x26            0.1949      0.093      2.090      0.037       0.012       0.378
    x27            0.0720      0.083      0.871      0.384      -0.090       0.234
    x28            0.1497      0.092      1.626      0.104      -0.031       0.330
    x29           -0.0432      0.098     -0.441      0.659      -0.235       0.149
    x30            0.5226      0.113      4.606      0.000       0.300       0.745
    x31           -0.6150      0.193     -3.179      0.001      -0.994      -0.236
    ==============================================================================
    Omnibus:                     5698.037   Durbin-Watson:                   1.968
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           423248.728
    Skew:                           4.061   Prob(JB):                         0.00
    Kurtosis:                      42.363   Cond. No.                     2.28e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.28e+03. This might indicate that
    there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.45
    ----- Test ------
    LREG Root mean squared error: 5.11
    ----- Training ------
    Root mean squared error: 2.62
    ----- Test ------
    Root mean squared error: 4.26
    ----- Test ------
    Average Error: 0.13
    [12:59:05] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.562
    Model:                            OLS   Adj. R-squared:                  0.560
    Method:                 Least Squares   F-statistic:                     258.7
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:06   Log-Likelihood:                -19511.
    No. Observations:                6288   AIC:                         3.909e+04
    Df Residuals:                    6256   BIC:                         3.930e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2492      0.068     33.008      0.000       2.116       2.383
    x1            -1.0923      0.313     -3.492      0.000      -1.706      -0.479
    x2             1.6234      0.233      6.965      0.000       1.166       2.080
    x3             0.4265      0.436      0.979      0.328      -0.428       1.281
    x4            -5.0791      0.258    -19.698      0.000      -5.585      -4.574
    x5             9.2229      0.176     52.398      0.000       8.878       9.568
    x6            -0.5582      0.157     -3.558      0.000      -0.866      -0.251
    x7           -19.0159     49.122     -0.387      0.699    -115.313      77.281
    x8             0.9677      2.274      0.426      0.670      -3.490       5.426
    x9            11.2151     28.560      0.393      0.695     -44.773      67.203
    x10            0.0765      0.373      0.205      0.837      -0.654       0.807
    x11           15.3143     39.916      0.384      0.701     -62.935      93.564
    x12           -0.8377      2.275     -0.368      0.713      -5.297       3.621
    x13           -0.1360      0.078     -1.749      0.080      -0.288       0.016
    x14           -0.0347      0.213     -0.163      0.871      -0.452       0.383
    x15           -0.2109      0.139     -1.516      0.130      -0.484       0.062
    x16           -0.0285      0.148     -0.193      0.847      -0.318       0.261
    x17            0.6583      0.230      2.864      0.004       0.208       1.109
    x18            0.0902      0.096      0.941      0.347      -0.098       0.278
    x19           -0.1045      0.085     -1.225      0.221      -0.272       0.063
    x20           -0.1581      0.081     -1.951      0.051      -0.317       0.001
    x21           -1.5467      2.255     -0.686      0.493      -5.967       2.874
    x22           -1.1680      1.660     -0.704      0.482      -4.422       2.086
    x23            0.3305      0.112      2.956      0.003       0.111       0.550
    x24           -0.0984      0.327     -0.301      0.764      -0.740       0.543
    x25           -0.9692      1.528     -0.634      0.526      -3.965       2.026
    x26            0.1641      0.102      1.607      0.108      -0.036       0.364
    x27            0.0023      0.076      0.030      0.976      -0.147       0.152
    x28            0.1720      0.089      1.931      0.054      -0.003       0.347
    x29            0.1125      0.097      1.158      0.247      -0.078       0.303
    x30            0.6648      0.112      5.934      0.000       0.445       0.884
    x31           -0.6153      0.222     -2.770      0.006      -1.051      -0.180
    ==============================================================================
    Omnibus:                     5703.552   Durbin-Watson:                   1.991
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           445912.281
    Skew:                           4.047   Prob(JB):                         0.00
    Kurtosis:                      43.453   Cond. No.                     2.29e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.29e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.39
    ----- Test ------
    LREG Root mean squared error: 5.37
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 4.59
    ----- Test ------
    Average Error: -0.06
    [12:59:06] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.564
    Model:                            OLS   Adj. R-squared:                  0.562
    Method:                 Least Squares   F-statistic:                     261.4
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:07   Log-Likelihood:                -19366.
    No. Observations:                6288   AIC:                         3.880e+04
    Df Residuals:                    6256   BIC:                         3.901e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2334      0.067     33.543      0.000       2.103       2.364
    x1            -1.1307      0.297     -3.803      0.000      -1.714      -0.548
    x2             1.2548      0.224      5.609      0.000       0.816       1.693
    x3             0.6434      0.415      1.552      0.121      -0.169       1.456
    x4            -4.8231      0.253    -19.062      0.000      -5.319      -4.327
    x5             9.0488      0.173     52.434      0.000       8.710       9.387
    x6            -0.5304      0.154     -3.439      0.001      -0.833      -0.228
    x7           -33.3774     47.491     -0.703      0.482    -126.475      59.720
    x8             0.7285      2.361      0.309      0.758      -3.900       5.357
    x9            19.5188     27.613      0.707      0.480     -34.612      73.649
    x10            0.1815      0.361      0.502      0.616      -0.527       0.890
    x11           26.9923     38.590      0.699      0.484     -48.658     102.642
    x12           -0.6234      2.359     -0.264      0.792      -5.248       4.001
    x13           -0.1075      0.075     -1.426      0.154      -0.255       0.040
    x14           -0.0278      0.207     -0.134      0.893      -0.434       0.378
    x15           -0.2120      0.137     -1.549      0.121      -0.480       0.056
    x16           -0.0091      0.144     -0.063      0.950      -0.292       0.273
    x17            0.8511      0.205      4.146      0.000       0.449       1.254
    x18            0.1893      0.093      2.033      0.042       0.007       0.372
    x19           -0.2195      0.084     -2.601      0.009      -0.385      -0.054
    x20           -0.1354      0.069     -1.948      0.051      -0.272       0.001
    x21           -1.4612      2.291     -0.638      0.524      -5.952       3.030
    x22           -0.9989      1.686     -0.592      0.554      -4.305       2.307
    x23            0.2744      0.110      2.485      0.013       0.058       0.491
    x24           -0.0544      0.333     -0.164      0.870      -0.706       0.597
    x25           -0.9818      1.552     -0.633      0.527      -4.023       2.060
    x26            0.1774      0.090      1.978      0.048       0.002       0.353
    x27            0.1657      0.080      2.063      0.039       0.008       0.323
    x28            0.0287      0.088      0.326      0.744      -0.144       0.202
    x29            0.1289      0.097      1.333      0.182      -0.061       0.318
    x30            0.4621      0.111      4.173      0.000       0.245       0.679
    x31           -0.6856      0.194     -3.543      0.000      -1.065      -0.306
    ==============================================================================
    Omnibus:                     5072.621   Durbin-Watson:                   1.958
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           263760.896
    Skew:                           3.472   Prob(JB):                         0.00
    Kurtosis:                      33.960   Cond. No.                     2.27e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.27e+03. This might indicate that
    there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.26
    ----- Test ------
    LREG Root mean squared error: 5.85
    ----- Training ------
    Root mean squared error: 2.57
    ----- Test ------
    Root mean squared error: 5.03
    ----- Test ------
    Average Error: 0.01
    [12:59:07] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.554
    Model:                            OLS   Adj. R-squared:                  0.551
    Method:                 Least Squares   F-statistic:                     250.3
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:07   Log-Likelihood:                -19360.
    No. Observations:                6288   AIC:                         3.878e+04
    Df Residuals:                    6256   BIC:                         3.900e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.1942      0.067     32.984      0.000       2.064       2.325
    x1            -1.2261      0.301     -4.076      0.000      -1.816      -0.636
    x2             1.2672      0.221      5.746      0.000       0.835       1.699
    x3             0.8009      0.414      1.937      0.053      -0.010       1.612
    x4            -4.7143      0.253    -18.619      0.000      -5.211      -4.218
    x5             8.8286      0.173     51.070      0.000       8.490       9.167
    x6            -0.4663      0.154     -3.024      0.003      -0.769      -0.164
    x7            10.4882     47.870      0.219      0.827     -83.353     104.329
    x8             0.5459      2.328      0.235      0.815      -4.017       5.109
    x9            -5.9809     27.832     -0.215      0.830     -60.541      48.580
    x10           -0.1294      0.363     -0.356      0.722      -0.841       0.583
    x11           -8.6014     38.897     -0.221      0.825     -84.853      67.651
    x12           -0.4701      2.327     -0.202      0.840      -5.032       4.092
    x13           -0.1361      0.074     -1.833      0.067      -0.282       0.009
    x14           -0.1680      0.210     -0.802      0.423      -0.579       0.243
    x15           -0.1544      0.139     -1.112      0.266      -0.427       0.118
    x16            0.0676      0.145      0.467      0.641      -0.216       0.352
    x17            0.7120      0.222      3.203      0.001       0.276       1.148
    x18            0.0651      0.093      0.701      0.484      -0.117       0.247
    x19           -0.0897      0.083     -1.081      0.280      -0.252       0.073
    x20           -0.1613      0.074     -2.187      0.029      -0.306      -0.017
    x21           -1.3649      2.228     -0.613      0.540      -5.732       3.002
    x22           -0.9276      1.641     -0.565      0.572      -4.144       2.289
    x23            0.2955      0.110      2.690      0.007       0.080       0.511
    x24            0.0429      0.324      0.133      0.895      -0.592       0.678
    x25           -0.8726      1.510     -0.578      0.563      -3.832       2.087
    x26            0.1633      0.090      1.806      0.071      -0.014       0.341
    x27           -0.0355      0.078     -0.453      0.651      -0.189       0.118
    x28            0.2192      0.088      2.499      0.012       0.047       0.391
    x29           -0.0051      0.095     -0.053      0.958      -0.191       0.181
    x30            0.5070      0.110      4.608      0.000       0.291       0.723
    x31           -0.5971      0.211     -2.824      0.005      -1.012      -0.183
    ==============================================================================
    Omnibus:                     5591.514   Durbin-Watson:                   1.987
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           380246.407
    Skew:                           3.967   Prob(JB):                         0.00
    Kurtosis:                      40.261   Cond. No.                     2.29e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.29e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.26
    ----- Test ------
    LREG Root mean squared error: 5.86
    ----- Training ------
    Root mean squared error: 2.70
    ----- Test ------
    Root mean squared error: 4.69
    ----- Test ------
    Average Error: -0.15
    [12:59:07] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.564
    Model:                            OLS   Adj. R-squared:                  0.561
    Method:                 Least Squares   F-statistic:                     260.6
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:08   Log-Likelihood:                -19505.
    No. Observations:                6288   AIC:                         3.907e+04
    Df Residuals:                    6256   BIC:                         3.929e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2443      0.068     32.966      0.000       2.111       2.378
    x1            -1.2127      0.303     -4.004      0.000      -1.806      -0.619
    x2             1.7161      0.228      7.539      0.000       1.270       2.162
    x3             0.3636      0.424      0.858      0.391      -0.467       1.194
    x4            -4.9479      0.257    -19.276      0.000      -5.451      -4.445
    x5             9.1463      0.175     52.352      0.000       8.804       9.489
    x6            -0.5431      0.157     -3.450      0.001      -0.852      -0.234
    x7           -35.8617     48.566     -0.738      0.460    -131.067      59.343
    x8             0.0088      2.261      0.004      0.997      -4.424       4.441
    x9            21.0025     28.237      0.744      0.457     -34.351      76.356
    x10            0.2001      0.369      0.542      0.588      -0.524       0.924
    x11           28.9908     39.464      0.735      0.463     -48.372     106.353
    x12            0.1304      2.262      0.058      0.954      -4.304       4.565
    x13           -0.0919      0.078     -1.174      0.241      -0.245       0.062
    x14           -0.0577      0.212     -0.273      0.785      -0.472       0.357
    x15           -0.2309      0.141     -1.641      0.101      -0.507       0.045
    x16           -0.0126      0.147     -0.086      0.931      -0.300       0.275
    x17            0.8584      0.213      4.034      0.000       0.441       1.276
    x18            0.1311      0.095      1.384      0.166      -0.055       0.317
    x19           -0.2142      0.084     -2.550      0.011      -0.379      -0.050
    x20           -0.1398      0.073     -1.926      0.054      -0.282       0.002
    x21           -0.4051      2.371     -0.171      0.864      -5.054       4.244
    x22           -0.2221      1.746     -0.127      0.899      -3.644       3.200
    x23            0.3257      0.112      2.901      0.004       0.106       0.546
    x24            0.1413      0.340      0.415      0.678      -0.526       0.809
    x25           -0.2129      1.608     -0.132      0.895      -3.365       2.939
    x26            0.1810      0.093      1.948      0.051      -0.001       0.363
    x27           -0.0137      0.079     -0.173      0.863      -0.169       0.142
    x28            0.2406      0.089      2.695      0.007       0.066       0.416
    x29            0.0681      0.099      0.688      0.491      -0.126       0.262
    x30            0.4785      0.111      4.303      0.000       0.260       0.696
    x31           -0.6925      0.200     -3.469      0.001      -1.084      -0.301
    ==============================================================================
    Omnibus:                     5469.403   Durbin-Watson:                   1.984
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           373783.491
    Skew:                           3.820   Prob(JB):                         0.00
    Kurtosis:                      39.990   Cond. No.                     2.28e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.28e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.38
    ----- Test ------
    LREG Root mean squared error: 5.38
    ----- Training ------
    Root mean squared error: 2.67
    ----- Test ------
    Root mean squared error: 4.54
    ----- Test ------
    Average Error: -0.19
    [12:59:08] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.545
    Model:                            OLS   Adj. R-squared:                  0.542
    Method:                 Least Squares   F-statistic:                     241.3
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:09   Log-Likelihood:                -19626.
    No. Observations:                6288   AIC:                         3.932e+04
    Df Residuals:                    6256   BIC:                         3.953e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2588      0.069     32.541      0.000       2.123       2.395
    x1            -1.3187      0.311     -4.238      0.000      -1.929      -0.709
    x2             1.4996      0.230      6.516      0.000       1.048       1.951
    x3             0.7045      0.433      1.628      0.104      -0.144       1.553
    x4            -4.8108      0.265    -18.128      0.000      -5.331      -4.291
    x5             9.0553      0.182     49.797      0.000       8.699       9.412
    x6            -0.4893      0.162     -3.023      0.003      -0.807      -0.172
    x7           -43.2529     50.166     -0.862      0.389    -141.596      55.090
    x8             1.5968      2.615      0.611      0.541      -3.529       6.723
    x9            25.3313     29.168      0.868      0.385     -31.847      82.510
    x10            0.2553      0.381      0.670      0.503      -0.492       1.002
    x11           34.9419     40.765      0.857      0.391     -44.971     114.855
    x12           -1.4992      2.612     -0.574      0.566      -6.619       3.621
    x13           -0.1703      0.080     -2.118      0.034      -0.328      -0.013
    x14           -0.0556      0.219     -0.254      0.799      -0.485       0.373
    x15           -0.2878      0.143     -2.017      0.044      -0.567      -0.008
    x16            0.0984      0.150      0.655      0.513      -0.196       0.393
    x17            0.6427      0.217      2.968      0.003       0.218       1.067
    x18            0.1023      0.097      1.053      0.292      -0.088       0.293
    x19           -0.1503      0.088     -1.708      0.088      -0.323       0.022
    x20           -0.1872      0.089     -2.103      0.035      -0.362      -0.013
    x21           -2.3278      2.440     -0.954      0.340      -7.112       2.456
    x22           -1.5879      1.797     -0.884      0.377      -5.110       1.934
    x23            0.2179      0.116      1.880      0.060      -0.009       0.445
    x24           -0.0733      0.350     -0.209      0.834      -0.760       0.613
    x25           -1.4407      1.655     -0.870      0.384      -4.686       1.804
    x26            0.1296      0.094      1.372      0.170      -0.056       0.315
    x27           -0.0501      0.085     -0.592      0.554      -0.216       0.116
    x28            0.1975      0.091      2.163      0.031       0.019       0.376
    x29            0.0832      0.098      0.853      0.394      -0.108       0.274
    x30            0.4986      0.116      4.296      0.000       0.271       0.726
    x31           -0.6033      0.202     -2.992      0.003      -0.998      -0.208
    ==============================================================================
    Omnibus:                     5585.136   Durbin-Watson:                   2.001
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           392506.447
    Skew:                           3.947   Prob(JB):                         0.00
    Kurtosis:                      40.892   Cond. No.                     2.29e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.29e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.49
    ----- Test ------
    LREG Root mean squared error: 4.94
    ----- Training ------
    Root mean squared error: 2.82
    ----- Test ------
    Root mean squared error: 4.11
    ----- Test ------
    Average Error: 0.09
    [12:59:09] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.547
    Model:                            OLS   Adj. R-squared:                  0.545
    Method:                 Least Squares   F-statistic:                     244.0
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:09   Log-Likelihood:                -19522.
    No. Observations:                6288   AIC:                         3.911e+04
    Df Residuals:                    6256   BIC:                         3.932e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2702      0.068     33.259      0.000       2.136       2.404
    x1            -1.0844      0.310     -3.493      0.000      -1.693      -0.476
    x2             1.6834      0.227      7.419      0.000       1.239       2.128
    x3             0.3063      0.430      0.712      0.476      -0.537       1.149
    x4            -4.7953      0.261    -18.382      0.000      -5.307      -4.284
    x5             9.0000      0.180     50.053      0.000       8.647       9.352
    x6            -0.4956      0.158     -3.128      0.002      -0.806      -0.185
    x7           -65.2530     49.498     -1.318      0.187    -162.286      31.780
    x8             0.5250      2.338      0.224      0.822      -4.059       5.109
    x9            38.0679     28.778      1.323      0.186     -18.347      94.482
    x10            0.4149      0.376      1.103      0.270      -0.323       1.152
    x11           52.8512     40.220      1.314      0.189     -25.994     131.697
    x12           -0.4375      2.338     -0.187      0.852      -5.022       4.147
    x13           -0.1390      0.077     -1.795      0.073      -0.291       0.013
    x14           -0.1893      0.215     -0.882      0.378      -0.610       0.231
    x15           -0.1304      0.142     -0.920      0.358      -0.408       0.148
    x16            0.0635      0.147      0.431      0.666      -0.225       0.352
    x17            0.6246      0.206      3.039      0.002       0.222       1.027
    x18            0.1499      0.095      1.570      0.116      -0.037       0.337
    x19           -0.1582      0.084     -1.883      0.060      -0.323       0.006
    x20           -0.1885      0.082     -2.303      0.021      -0.349      -0.028
    x21           -2.0098      2.313     -0.869      0.385      -6.545       2.525
    x22           -1.4624      1.703     -0.859      0.391      -4.802       1.877
    x23            0.3321      0.113      2.952      0.003       0.112       0.553
    x24           -0.0714      0.335     -0.213      0.831      -0.729       0.586
    x25           -1.2278      1.568     -0.783      0.434      -4.301       1.846
    x26            0.2082      0.091      2.278      0.023       0.029       0.387
    x27           -0.0219      0.078     -0.283      0.777      -0.174       0.130
    x28            0.1483      0.082      1.818      0.069      -0.012       0.308
    x29           -0.0342      0.094     -0.362      0.717      -0.219       0.151
    x30            0.5915      0.113      5.242      0.000       0.370       0.813
    x31           -0.5273      0.190     -2.781      0.005      -0.899      -0.156
    ==============================================================================
    Omnibus:                     5715.145   Durbin-Watson:                   2.023
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           430937.796
    Skew:                           4.076   Prob(JB):                         0.00
    Kurtosis:                      42.728   Cond. No.                     2.32e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.32e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.40
    ----- Test ------
    LREG Root mean squared error: 5.32
    ----- Training ------
    Root mean squared error: 2.68
    ----- Test ------
    Root mean squared error: 4.50
    ----- Test ------
    Average Error: 0.14
    [12:59:09] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.555
    Model:                            OLS   Adj. R-squared:                  0.553
    Method:                 Least Squares   F-statistic:                     252.2
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:10   Log-Likelihood:                -19520.
    No. Observations:                6288   AIC:                         3.910e+04
    Df Residuals:                    6256   BIC:                         3.932e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2537      0.068     33.025      0.000       2.120       2.387
    x1            -1.2260      0.302     -4.066      0.000      -1.817      -0.635
    x2             1.4059      0.227      6.189      0.000       0.961       1.851
    x3             0.6859      0.420      1.633      0.103      -0.138       1.509
    x4            -4.8934      0.258    -18.999      0.000      -5.398      -4.388
    x5             9.0952      0.177     51.440      0.000       8.749       9.442
    x6            -0.5210      0.156     -3.330      0.001      -0.828      -0.214
    x7           -51.6980     48.621     -1.063      0.288    -147.012      43.616
    x8             0.5955      2.329      0.256      0.798      -3.970       5.161
    x9            30.1865     28.268      1.068      0.286     -25.229      85.602
    x10            0.3118      0.369      0.845      0.398      -0.411       1.035
    x11           41.8540     39.509      1.059      0.289     -35.596     119.304
    x12           -0.4936      2.329     -0.212      0.832      -5.060       4.072
    x13           -0.0892      0.077     -1.152      0.250      -0.241       0.063
    x14           -0.0854      0.214     -0.399      0.690      -0.505       0.334
    x15           -0.2260      0.140     -1.619      0.105      -0.500       0.048
    x16            0.0399      0.149      0.267      0.789      -0.253       0.333
    x17            0.6528      0.226      2.892      0.004       0.210       1.095
    x18            0.1557      0.096      1.627      0.104      -0.032       0.343
    x19           -0.1682      0.087     -1.925      0.054      -0.340       0.003
    x20           -0.1657      0.073     -2.268      0.023      -0.309      -0.022
    x21           -2.3404      2.355     -0.994      0.320      -6.958       2.277
    x22           -1.6686      1.733     -0.963      0.336      -5.066       1.729
    x23            0.2150      0.115      1.874      0.061      -0.010       0.440
    x24           -0.0657      0.340     -0.193      0.847      -0.733       0.602
    x25           -1.5175      1.597     -0.950      0.342      -4.648       1.612
    x26            0.1267      0.093      1.362      0.173      -0.056       0.309
    x27           -0.0125      0.075     -0.167      0.868      -0.160       0.135
    x28            0.1301      0.089      1.455      0.146      -0.045       0.305
    x29            0.1419      0.099      1.428      0.153      -0.053       0.337
    x30            0.5484      0.112      4.900      0.000       0.329       0.768
    x31           -0.5094      0.220     -2.314      0.021      -0.941      -0.078
    ==============================================================================
    Omnibus:                     5703.492   Durbin-Watson:                   2.045
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           438210.449
    Skew:                           4.054   Prob(JB):                         0.00
    Kurtosis:                      43.085   Cond. No.                     2.27e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.27e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.39
    ----- Test ------
    LREG Root mean squared error: 5.33
    ----- Training ------
    Root mean squared error: 2.66
    ----- Test ------
    Root mean squared error: 4.50
    ----- Test ------
    Average Error: 0.03
    [12:59:10] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.539
    Model:                            OLS   Adj. R-squared:                  0.537
    Method:                 Least Squares   F-statistic:                     235.9
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:11   Log-Likelihood:                -19642.
    No. Observations:                6288   AIC:                         3.935e+04
    Df Residuals:                    6256   BIC:                         3.956e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2386      0.070     32.179      0.000       2.102       2.375
    x1            -1.4817      0.311     -4.767      0.000      -2.091      -0.872
    x2             1.3244      0.232      5.702      0.000       0.869       1.780
    x3             1.0777      0.432      2.493      0.013       0.230       1.925
    x4            -4.7855      0.265    -18.032      0.000      -5.306      -4.265
    x5             8.9286      0.182     49.163      0.000       8.573       9.285
    x6            -0.4903      0.160     -3.061      0.002      -0.804      -0.176
    x7           -58.0705     50.286     -1.155      0.248    -156.648      40.507
    x8             1.1784      2.472      0.477      0.634      -3.668       6.025
    x9            33.9240     29.236      1.160      0.246     -23.388      91.237
    x10            0.3454      0.382      0.905      0.366      -0.403       1.093
    x11           46.9707     40.861      1.150      0.250     -33.131     127.072
    x12           -1.0734      2.470     -0.435      0.664      -5.915       3.768
    x13           -0.1014      0.079     -1.279      0.201      -0.257       0.054
    x14            0.0777      0.219      0.354      0.723      -0.352       0.508
    x15           -0.2745      0.143     -1.924      0.054      -0.554       0.005
    x16           -0.0679      0.151     -0.449      0.653      -0.365       0.229
    x17            0.6942      0.241      2.881      0.004       0.222       1.167
    x18            0.1964      0.098      2.009      0.045       0.005       0.388
    x19           -0.1439      0.087     -1.651      0.099      -0.315       0.027
    x20           -0.1312      0.074     -1.763      0.078      -0.277       0.015
    x21           -3.1101      2.446     -1.271      0.204      -7.905       1.685
    x22           -2.2381      1.800     -1.243      0.214      -5.767       1.291
    x23            0.2903      0.116      2.512      0.012       0.064       0.517
    x24           -0.1962      0.352     -0.557      0.577      -0.886       0.494
    x25           -2.0335      1.659     -1.226      0.220      -5.285       1.218
    x26            0.1594      0.094      1.702      0.089      -0.024       0.343
    x27            0.0408      0.082      0.498      0.619      -0.120       0.201
    x28            0.0405      0.084      0.485      0.628      -0.123       0.204
    x29           -0.0379      0.099     -0.384      0.701      -0.232       0.156
    x30            0.5957      0.117      5.111      0.000       0.367       0.824
    x31           -0.6624      0.229     -2.895      0.004      -1.111      -0.214
    ==============================================================================
    Omnibus:                     5627.545   Durbin-Watson:                   2.002
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           379258.824
    Skew:                           4.015   Prob(JB):                         0.00
    Kurtosis:                      40.190   Cond. No.                     2.31e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.31e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.50
    ----- Test ------
    LREG Root mean squared error: 4.88
    ----- Training ------
    Root mean squared error: 2.71
    ----- Test ------
    Root mean squared error: 4.19
    ----- Test ------
    Average Error: 0.02
    [12:59:11] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.547
    Model:                            OLS   Adj. R-squared:                  0.545
    Method:                 Least Squares   F-statistic:                     243.9
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:11   Log-Likelihood:                -19573.
    No. Observations:                6288   AIC:                         3.921e+04
    Df Residuals:                    6256   BIC:                         3.943e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2847      0.069     33.201      0.000       2.150       2.420
    x1            -1.3510      0.307     -4.402      0.000      -1.953      -0.749
    x2             1.4345      0.229      6.257      0.000       0.985       1.884
    x3             0.7941      0.428      1.854      0.064      -0.046       1.634
    x4            -4.7136      0.260    -18.108      0.000      -5.224      -4.203
    x5             8.9857      0.180     49.985      0.000       8.633       9.338
    x6            -0.5018      0.158     -3.171      0.002      -0.812      -0.192
    x7           -43.5573     49.523     -0.880      0.379    -140.640      53.525
    x8             0.9762      2.591      0.377      0.706      -4.103       6.056
    x9            25.4509     28.793      0.884      0.377     -30.994      81.896
    x10            0.3446      0.378      0.911      0.362      -0.397       1.086
    x11           35.2470     40.241      0.876      0.381     -43.639     114.133
    x12           -0.8834      2.589     -0.341      0.733      -5.959       4.192
    x13           -0.2423      0.081     -2.998      0.003      -0.401      -0.084
    x14           -0.2880      0.214     -1.346      0.178      -0.707       0.131
    x15           -0.1149      0.141     -0.813      0.416      -0.392       0.162
    x16            0.1269      0.148      0.857      0.392      -0.163       0.417
    x17            0.6749      0.208      3.239      0.001       0.266       1.083
    x18            0.0612      0.096      0.638      0.523      -0.127       0.249
    x19           -0.0960      0.087     -1.108      0.268      -0.266       0.074
    x20           -0.1375      0.071     -1.934      0.053      -0.277       0.002
    x21           -2.1725      2.408     -0.902      0.367      -6.892       2.547
    x22           -1.5770      1.773     -0.890      0.374      -5.052       1.898
    x23            0.2107      0.114      1.846      0.065      -0.013       0.434
    x24           -0.0305      0.347     -0.088      0.930      -0.710       0.649
    x25           -1.4643      1.633     -0.897      0.370      -4.665       1.736
    x26            0.1729      0.093      1.851      0.064      -0.010       0.356
    x27           -0.1198      0.087     -1.378      0.168      -0.290       0.051
    x28            0.1514      0.083      1.829      0.067      -0.011       0.314
    x29            0.1095      0.096      1.140      0.255      -0.079       0.298
    x30            0.6424      0.114      5.642      0.000       0.419       0.866
    x31           -0.5620      0.195     -2.883      0.004      -0.944      -0.180
    ==============================================================================
    Omnibus:                     5600.881   Durbin-Watson:                   2.047
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           397755.377
    Skew:                           3.961   Prob(JB):                         0.00
    Kurtosis:                      41.149   Cond. No.                     2.30e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.3e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.44
    ----- Test ------
    LREG Root mean squared error: 5.15
    ----- Training ------
    Root mean squared error: 2.74
    ----- Test ------
    Root mean squared error: 4.00
    ----- Test ------
    Average Error: 0.23
    [12:59:11] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.548
    Model:                            OLS   Adj. R-squared:                  0.546
    Method:                 Least Squares   F-statistic:                     245.1
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:12   Log-Likelihood:                -19536.
    No. Observations:                6288   AIC:                         3.914e+04
    Df Residuals:                    6256   BIC:                         3.935e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2646      0.068     33.112      0.000       2.131       2.399
    x1            -1.3407      0.303     -4.424      0.000      -1.935      -0.747
    x2             1.7455      0.230      7.582      0.000       1.294       2.197
    x3             0.5060      0.425      1.192      0.233      -0.326       1.338
    x4            -4.7442      0.259    -18.309      0.000      -5.252      -4.236
    x5             9.0094      0.179     50.386      0.000       8.659       9.360
    x6            -0.4243      0.158     -2.693      0.007      -0.733      -0.115
    x7           -46.5517     48.399     -0.962      0.336    -141.431      48.327
    x8             0.8582      2.371      0.362      0.717      -3.790       5.506
    x9            27.2434     28.140      0.968      0.333     -27.920      82.407
    x10            0.2520      0.367      0.686      0.493      -0.468       0.972
    x11           37.6453     39.328      0.957      0.338     -39.450     114.741
    x12           -0.7737      2.369     -0.327      0.744      -5.417       3.870
    x13           -0.0857      0.077     -1.111      0.267      -0.237       0.066
    x14            0.0219      0.214      0.102      0.919      -0.398       0.442
    x15           -0.2776      0.141     -1.964      0.050      -0.555      -0.001
    x16           -0.0252      0.148     -0.170      0.865      -0.315       0.265
    x17            0.7797      0.217      3.587      0.000       0.354       1.206
    x18            0.1489      0.096      1.559      0.119      -0.038       0.336
    x19           -0.0929      0.085     -1.091      0.275      -0.260       0.074
    x20           -0.1301      0.071     -1.841      0.066      -0.269       0.008
    x21           -0.3107      2.310     -0.134      0.893      -4.839       4.218
    x22           -0.1520      1.700     -0.089      0.929      -3.485       3.181
    x23            0.3460      0.112      3.086      0.002       0.126       0.566
    x24            0.1947      0.333      0.584      0.559      -0.459       0.848
    x25           -0.1546      1.566     -0.099      0.921      -3.224       2.915
    x26            0.1537      0.092      1.671      0.095      -0.027       0.334
    x27            0.0304      0.079      0.383      0.702      -0.125       0.186
    x28            0.0892      0.081      1.097      0.273      -0.070       0.249
    x29            0.0841      0.096      0.872      0.383      -0.105       0.273
    x30            0.5558      0.113      4.910      0.000       0.334       0.778
    x31           -0.6834      0.204     -3.353      0.001      -1.083      -0.284
    ==============================================================================
    Omnibus:                     5417.889   Durbin-Watson:                   1.989
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           341616.239
    Skew:                           3.791   Prob(JB):                         0.00
    Kurtosis:                      38.304   Cond. No.                     2.27e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.27e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.41
    ----- Test ------
    LREG Root mean squared error: 5.28
    ----- Training ------
    Root mean squared error: 2.72
    ----- Test ------
    Root mean squared error: 4.34
    ----- Test ------
    Average Error: 0.09
    [12:59:12] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.564
    Model:                            OLS   Adj. R-squared:                  0.561
    Method:                 Least Squares   F-statistic:                     260.6
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:13   Log-Likelihood:                -19414.
    No. Observations:                6288   AIC:                         3.889e+04
    Df Residuals:                    6256   BIC:                         3.911e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2548      0.067     33.610      0.000       2.123       2.386
    x1            -1.2016      0.298     -4.037      0.000      -1.785      -0.618
    x2             1.7796      0.226      7.879      0.000       1.337       2.222
    x3             0.2763      0.417      0.663      0.507      -0.541       1.093
    x4            -4.6707      0.255    -18.330      0.000      -5.170      -4.171
    x5             8.9042      0.173     51.409      0.000       8.565       9.244
    x6            -0.4675      0.155     -3.011      0.003      -0.772      -0.163
    x7             1.6161     47.365      0.034      0.973     -91.235      94.467
    x8             0.7635      2.323      0.329      0.742      -3.791       5.318
    x9            -0.7706     27.539     -0.028      0.978     -54.756      53.214
    x10           -0.0346      0.360     -0.096      0.923      -0.740       0.671
    x11           -1.4680     38.488     -0.038      0.970     -76.917      73.981
    x12           -0.6504      2.322     -0.280      0.779      -5.202       3.901
    x13           -0.1192      0.077     -1.558      0.119      -0.269       0.031
    x14           -0.1807      0.213     -0.849      0.396      -0.598       0.237
    x15           -0.2094      0.140     -1.492      0.136      -0.484       0.066
    x16            0.0731      0.147      0.498      0.618      -0.215       0.361
    x17            0.6871      0.213      3.227      0.001       0.270       1.105
    x18            0.1380      0.094      1.463      0.144      -0.047       0.323
    x19           -0.1174      0.084     -1.390      0.164      -0.283       0.048
    x20           -0.1453      0.079     -1.838      0.066      -0.300       0.010
    x21           -1.4296      2.283     -0.626      0.531      -5.906       3.047
    x22           -1.0711      1.680     -0.638      0.524      -4.365       2.222
    x23            0.2190      0.111      1.975      0.048       0.002       0.436
    x24           -0.0499      0.329     -0.151      0.880      -0.695       0.596
    x25           -0.9234      1.547     -0.597      0.551      -3.957       2.110
    x26            0.0976      0.091      1.072      0.284      -0.081       0.276
    x27           -0.0571      0.079     -0.724      0.469      -0.212       0.097
    x28            0.1925      0.088      2.192      0.028       0.020       0.365
    x29            0.1573      0.097      1.614      0.107      -0.034       0.348
    x30            0.6745      0.111      6.075      0.000       0.457       0.892
    x31           -0.6053      0.203     -2.978      0.003      -1.004      -0.207
    ==============================================================================
    Omnibus:                     5200.236   Durbin-Watson:                   2.011
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           294257.254
    Skew:                           3.583   Prob(JB):                         0.00
    Kurtosis:                      35.738   Cond. No.                     2.26e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.26e+03. This might indicate that
    there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.30
    ----- Test ------
    LREG Root mean squared error: 5.69
    ----- Training ------
    Root mean squared error: 2.66
    ----- Test ------
    Root mean squared error: 4.59
    ----- Test ------
    Average Error: 0.08
    [12:59:13] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.561
    Model:                            OLS   Adj. R-squared:                  0.559
    Method:                 Least Squares   F-statistic:                     258.0
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:13   Log-Likelihood:                -19404.
    No. Observations:                6288   AIC:                         3.887e+04
    Df Residuals:                    6256   BIC:                         3.909e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.2173      0.067     33.101      0.000       2.086       2.349
    x1            -1.1416      0.299     -3.821      0.000      -1.727      -0.556
    x2             1.3077      0.224      5.835      0.000       0.868       1.747
    x3             0.6275      0.415      1.513      0.130      -0.185       1.440
    x4            -4.9437      0.253    -19.525      0.000      -5.440      -4.447
    x5             9.1066      0.173     52.566      0.000       8.767       9.446
    x6            -0.5990      0.155     -3.872      0.000      -0.902      -0.296
    x7           -53.2359     48.814     -1.091      0.275    -148.927      42.456
    x8             0.2507      2.302      0.109      0.913      -4.262       4.763
    x9            31.1045     28.381      1.096      0.273     -24.531      86.740
    x10            0.3551      0.370      0.959      0.337      -0.371       1.081
    x11           43.0574     39.665      1.086      0.278     -34.700     120.815
    x12           -0.1875      2.302     -0.081      0.935      -4.699       4.324
    x13           -0.1142      0.076     -1.504      0.133      -0.263       0.035
    x14           -0.2317      0.210     -1.105      0.269      -0.643       0.179
    x15           -0.1437      0.138     -1.043      0.297      -0.414       0.127
    x16            0.0772      0.144      0.536      0.592      -0.205       0.360
    x17            0.7558      0.213      3.549      0.000       0.338       1.173
    x18            0.1641      0.094      1.753      0.080      -0.019       0.348
    x19           -0.1576      0.084     -1.875      0.061      -0.322       0.007
    x20           -0.1447      0.073     -1.988      0.047      -0.287      -0.002
    x21           -2.7201      2.302     -1.181      0.237      -7.233       1.793
    x22           -1.9609      1.694     -1.157      0.247      -5.282       1.360
    x23            0.1833      0.112      1.636      0.102      -0.036       0.403
    x24           -0.1877      0.332     -0.566      0.571      -0.838       0.462
    x25           -1.8035      1.561     -1.155      0.248      -4.864       1.257
    x26            0.2704      0.099      2.729      0.006       0.076       0.465
    x27            0.0007      0.079      0.009      0.992      -0.154       0.156
    x28            0.0433      0.080      0.542      0.588      -0.113       0.200
    x29            0.0258      0.094      0.274      0.784      -0.158       0.210
    x30            0.5309      0.112      4.753      0.000       0.312       0.750
    x31           -0.6128      0.202     -3.027      0.002      -1.010      -0.216
    ==============================================================================
    Omnibus:                     5530.502   Durbin-Watson:                   2.005
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           386064.653
    Skew:                           3.884   Prob(JB):                         0.00
    Kurtosis:                      40.592   Cond. No.                     2.31e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.31e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.30
    ----- Test ------
    LREG Root mean squared error: 5.72
    ----- Training ------
    Root mean squared error: 2.60
    ----- Test ------
    Root mean squared error: 4.81
    ----- Test ------
    Average Error: 0.07
    [12:59:13] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core, or some parameter
      actually being used
      but getting flagged wrongly here. Please open an issue if you find
      any such cases.
                                OLS Regression Results
    ==============================================================================
    Dep. Variable:                 y_norm   R-squared:                       0.551
    Model:                            OLS   Adj. R-squared:                  0.549
    Method:                 Least Squares   F-statistic:                     247.6
    Date:                Wed, 13 Sep 2023   Prob (F-statistic):               0.00
    Time:                        12:59:14   Log-Likelihood:                -19527.
    No. Observations:                6288   AIC:                         3.912e+04
    Df Residuals:                    6256   BIC:                         3.933e+04
    Df Model:                          31
    Covariance Type:            nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          2.1795      0.068     31.906      0.000       2.046       2.313
    x1            -1.2629      0.305     -4.144      0.000      -1.860      -0.665
    x2             1.3776      0.226      6.084      0.000       0.934       1.822
    x3             0.7036      0.424      1.659      0.097      -0.128       1.535
    x4            -4.9023      0.259    -18.944      0.000      -5.410      -4.395
    x5             9.0082      0.177     50.989      0.000       8.662       9.355
    x6            -0.5418      0.158     -3.426      0.001      -0.852      -0.232
    x7           -55.4107     49.223     -1.126      0.260    -151.904      41.083
    x8             0.6442      2.405      0.268      0.789      -4.071       5.359
    x9            32.3798     28.619      1.131      0.258     -23.723      88.482
    x10            0.3718      0.375      0.992      0.321      -0.363       1.107
    x11           44.8754     39.998      1.122      0.262     -33.533     123.284
    x12           -0.5661      2.406     -0.235      0.814      -5.282       4.150
    x13           -0.1068      0.078     -1.364      0.173      -0.260       0.047
    x14           -0.2067      0.214     -0.965      0.335      -0.627       0.213
    x15           -0.1949      0.141     -1.382      0.167      -0.471       0.081
    x16            0.0835      0.147      0.568      0.570      -0.205       0.372
    x17            0.6747      0.204      3.301      0.001       0.274       1.075
    x18            0.1772      0.096      1.843      0.065      -0.011       0.366
    x19           -0.1656      0.087     -1.898      0.058      -0.337       0.005
    x20           -0.1765      0.080     -2.200      0.028      -0.334      -0.019
    x21           -2.6873      2.394     -1.123      0.262      -7.380       2.006
    x22           -1.9273      1.762     -1.094      0.274      -5.381       1.527
    x23            0.2363      0.112      2.118      0.034       0.018       0.455
    x24           -0.2124      0.346     -0.613      0.540      -0.892       0.467
    x25           -1.8074      1.621     -1.115      0.265      -4.986       1.371
    x26            0.1549      0.091      1.699      0.089      -0.024       0.334
    x27           -0.0192      0.077     -0.251      0.802      -0.169       0.131
    x28            0.1449      0.082      1.773      0.076      -0.015       0.305
    x29            0.0325      0.099      0.328      0.743      -0.162       0.227
    x30            0.6143      0.113      5.429      0.000       0.392       0.836
    x31           -0.5799      0.190     -3.052      0.002      -0.952      -0.207
    ==============================================================================
    Omnibus:                     5791.079   Durbin-Watson:                   2.038
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):           451613.776
    Skew:                           4.156   Prob(JB):                         0.00
    Kurtosis:                      43.677   Cond. No.                     2.29e+03
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The condition number is large, 2.29e+03. This might indicate that there are
    strong multicollinearity or other numerical problems.
    ----- Training ------
    LREG Root mean squared error: 5.40
    ----- Test ------
    LREG Root mean squared error: 5.31
    ----- Training ------
    Root mean squared error: 2.64
    ----- Test ------
    Root mean squared error: 4.59
    ----- Test ------
    Average Error: -0.21

```python
# Define a function to plot RMSEs
def rmse_bin_plot(te_rmse, tr_rmse, te_ave, min_rg, max_rg, step):

    m_test_rmse = statistics.mean(te_rmse)
    plt.figure(figsize=(4, 3))
    plt.axvline(m_test_rmse, color="b", linestyle="dashed")
    plt.hist(
        te_rmse,
        bins=np.arange(min_rg, max_rg, step),
        edgecolor="k",
        histtype="bar",
        density=True,
    )
    sd_test_rmse = statistics.stdev(te_rmse)

    m_train_rmse = statistics.mean(tr_rmse)
    plt.axvline(m_train_rmse, color="red", linestyle="dashed")
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

    # Average Error
    m_test_ave = statistics.mean(te_ave)
    sd_test_ave = statistics.stdev(te_ave)

    print(f"mean_RMSE_test: {m_test_rmse:.2f}")
    # print(f"mean_RMSE_train: {m_train_rmse:.2f}")

    print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
    # print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

    print(f"mean_AVE_test: {m_test_ave:.2f}")

    print(f"stdev_AVE_test: {sd_test_ave:.2f}")

    # create legend
    labels = ["Mean_test", "Mean_train", "test", "train"]
    plt.legend(labels)

    plt.xlabel("The RMSE error")
    plt.ylabel("Frequency")
    plt.title("histogram of the RMSE distribution")
    plt.show()
```

```python
# RMSE in Total

print("RMSE in total", "\n")
rmse_bin_plot(test_RMSE["all"], train_RMSE["all"], test_AVE["all"], 0.5, 7.0, 0.45)
```

RMSE in total

mean_RMSE_test: 4.47
stdev_RMSE_test: 0.26
mean_AVE_test: 0.03
stdev_AVE_test: 0.11

![png](output_17_1.png)

```python
bin_params = {
    1: (0.05, 1.0, 0.07),
    2: (0.5, 2.5, 0.1),
    3: (0.5, 5, 0.35),
    4: (-6.0, 16.0, 1.5),
    5: (-21.0, 35.0, 4),
}


for bin_num in range(1, 6):

    print(f"RMSE per bin {bin_num}\n")
    rmse_bin_plot(
        test_RMSE[bin_num], train_RMSE[bin_num], test_AVE[bin_num],
        *bin_params[bin_num]
    )
```

RMSE per bin 1

mean_RMSE_test: 0.26
stdev_RMSE_test: 0.06
mean_AVE_test: 0.07
stdev_AVE_test: 0.01

![png](output_18_1.png)

RMSE per bin 2

mean_RMSE_test: 2.19
stdev_RMSE_test: 0.25
mean_AVE_test: 0.93
stdev_AVE_test: 0.08

![png](output_18_3.png)

RMSE per bin 3

mean_RMSE_test: 4.71
stdev_RMSE_test: 0.55
mean_AVE_test: 1.04
stdev_AVE_test: 0.34

![png](output_18_5.png)

RMSE per bin 4

mean_RMSE_test: 13.24
stdev_RMSE_test: 1.11
mean_AVE_test: -6.09
stdev_AVE_test: 1.37

![png](output_18_7.png)

RMSE per bin 5

mean_RMSE_test: 27.99
stdev_RMSE_test: 4.07
mean_AVE_test: -20.24
stdev_AVE_test: 3.24

![png](output_18_9.png)

```python

```

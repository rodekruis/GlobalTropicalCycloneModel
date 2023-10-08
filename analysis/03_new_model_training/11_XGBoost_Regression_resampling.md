# Regression Model (Data Resampling)

We utilized the SMOTE technique to address the class
imbalance in the data by oversampling the minority class.
In order to estimate the accuracy of a resampled regression model,
we created a binary target variable to serve as an auxiliary variable
for resampling the training data.
The binary target variable was used solely for the purpose of
resampling the data and was ignored during the estimation of the
root mean squared error (RMSE) for the new resampled dataset.

```python
%load_ext jupyter_black
```

```python
from collections import defaultdict, Counter
import statistics

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import colorama
from colorama import Fore

from utils import get_training_dataset
```

```python
# Read csv file and import to df
df = get_training_dataset()
df.head()
```

```python
# For simplicity in the later steps of the code,
# I removed last feature from last column and insert it before target
first_column = df.pop("percent_houses_damaged_5years")
df.insert(20, "percent_houses_damaged_5years", first_column)
```

```python
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)
```

```python
# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```

```python
# define a threshold to separate target into damaged and not_damaged
thres = 10.0

for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] >= thres:
        df.at[i, "binary_damage"] = 1
    else:
        df.at[i, "binary_damage"] = 0

df["binary_damage"] = df["binary_damage"].astype("int")
df
```

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)].reset_index(drop=True)
df = df.drop(columns=["grid_point_id", "typhoon_year"])
df.head()
```

```python
# Define bin
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
print(samples_per_bin2)
print(binsP2)
```

[38901  7232  2552   925   144]
[0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]

```python
# Check the bins' intervalls
df["binary_damage"].value_counts(bins=binsP2)
```

(-0.001, 9e-05]    48685
(9e-05, 1.0]        1069
(1.0, 10.0]            0
(10.0, 50.0]           0
(50.0, 101.0]          0
Name: binary_damage, dtype: int64

```python
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
""" We keep the continuous target in the dataset, since we only want to
 use binary target to resample dataset and
after that we will remove it and use continuous target as the main target
"""
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
    "percent_houses_damaged",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["binary_damage"]
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
        'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
        'water', 'total_pop', 'percent_houses_damaged_5years',
        'percent_houses_damaged'],
      dtype='object')

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, df["binary_damage"], stratify=y_input_strat, test_size=0.2
)
```

```python
# Check train data before resampling
print(Counter(y_train))
```

Counter({0: 38948, 1: 855})

```python
# Create an oversampled training data
smote = SMOTE()
# random_state=101
X_train, y_train = smote.fit_resample(X_train, y_train)
```

```python
# Check train data after resampling
print(Counter(y_train))
```

Counter({0: 38948, 1: 38948})

```python
# Insert X_train into a df
df_train = pd.DataFrame(X_train)
df_train
```

```python
# Insert test set into a df
df_test = pd.DataFrame(X_test)
df_test
```

```python
# Show histogram of damage for train data
df_train.hist(column="percent_houses_damaged", figsize=(4, 3))
plt.title("percent_houses_damaged for train_data")

# Show histogram of damage for test data
df_test.hist(column="percent_houses_damaged", figsize=(4, 3))
plt.title("percent_houses_damaged for test_data")
```

Text(0.5, 1.0, 'percent_houses_damaged for test_data')

![png](output_21_1.png)

![png](output_21_2.png)

```python
# We use this features to train regression model
features_new = [
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
# Split X and y from train dataframe
X_train = df_train[features_new]
display(X_train.columns)
y_train = df_train["percent_houses_damaged"]
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
        'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
        'water', 'total_pop', 'percent_houses_damaged_5years'],
      dtype='object')

```python
# Split X and y from test dataframe
X_test = df_test[features_new]
display(X_test.columns)
y_test = df_test["percent_houses_damaged"]
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
        'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
        'water', 'total_pop', 'percent_houses_damaged_5years'],
      dtype='object')

```python
sc = preprocessing.StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
```

```python
X_train = X_train_sc
X_test = X_test_sc
y_train = y_train
y_test = y_test
```

```python
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
xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

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


print(f"LREG Root mean squared error(training): {rmse_train_LREG:.2f}")
print(f"LREG Root mean squared error(test): {rmse_LREG:.2f}")


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

print(f"RMSE_train_in_total: {rmse_train:.2f}")
print(f"RMSE_test_in_total: {rmse:.2f}")


# Calculate RMSE per bins

bin_index_test = np.digitize(y_test, bins=binsP2)
bin_index_train = np.digitize(y_train, bins=binsP2)


for bin_num in range(1, 6):

    # Estimation of RMSE for train data
    mse_train_idx = mean_squared_error(
        y_train[bin_index_train == bin_num],
        y_pred_train_clipped[bin_index_train == bin_num],
    )
    rmse_train = np.sqrt(mse_train_idx)

    # Estimation of RMSE for test data
    mse_idx = mean_squared_error(
        y_test[bin_index_test == bin_num], y_pred_clipped[bin_index_test == bin_num]
    )
    rmse = np.sqrt(mse_idx)

    print(f"RMSE_train: {rmse_train:.2f}")
    print(f"RMSE_test: {rmse:.2f}")
```

WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905
/work/src/learner.cc:576:
Parameters: { "early_stopping_rounds" } might not be used.
This could be a false alarm, with some parameters getting used by language
bindings but then being mistakenly passed down to XGBoost core, or some
parameter actually being used but getting flagged wrongly here.
Please open an issue if you find any such cases.
OLS Regression Results

Dep. Variable:     percent_houses_damaged   R-squared:                    0.616
Model:                                OLS   Adj. R-squared:               0.616
Method:                     Least Squares   F-statistic:                  7356.
Date:                    Wed, 22 Mar 2023   Prob (F-statistic):            0.00
Time:                            21:51:22   Log-Likelihood:         -2.9724e+05
No. Observations:                   77896   AIC:                      5.945e+05
Df Residuals:                       77878   BIC:                      5.947e+05
Df Model:                              17
Covariance Type:                nonrobust

coef    std err          t      P>|t|      [0.025      0.975]

const         14.4614      0.041    351.698      0.000      14.381      14.542
x1            16.1189      0.075    214.567      0.000      15.972      16.266
x2             3.2518      0.076     42.701      0.000       3.103       3.401
x3             0.4538      0.145      3.131      0.002       0.170       0.738
x4             1.9351      0.103     18.743      0.000       1.733       2.137
x5            -1.7601      0.101    -17.472      0.000      -1.958      -1.563
x6             0.0751      0.067      1.121      0.262      -0.056       0.206
x7            -8.2655      0.764    -10.824      0.000      -9.762      -6.769
x8             0.2073      0.357      0.581      0.561      -0.492       0.907
x9             7.1466      0.725      9.860      0.000       5.726       8.567
x10           -0.0323      0.312     -0.103      0.918      -0.644       0.580
x11           -0.1445      0.085     -1.695      0.090      -0.312       0.023
x12            0.6583      0.049     13.382      0.000       0.562       0.755
x13           -1.0258      0.075    -13.698      0.000      -1.173      -0.879
x14        -3.264e+12   4.47e+12     -0.731      0.465    -1.2e+13    5.49e+12
x15        -6.081e+12   8.32e+12     -0.731      0.465   -2.24e+13    1.02e+13
x16        -6.278e+12   8.59e+12     -0.731      0.465   -2.31e+13    1.06e+13
x17           -0.3668      0.144     -2.551      0.011      -0.649      -0.085
x18            0.4351      0.040     10.986      0.000       0.357       0.513

Omnibus:                    20014.673   Durbin-Watson:                   1.903
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            82746.608
Skew:                           1.219   Prob(JB):                         0.00
Kurtosis:                       7.422   Cond. No.                     7.79e+14

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly
specified.
[2] The smallest eigenvalue is 7.41e-25. This might indicate that there are
strong multicollinearity problems or that the design matrix is singular.
LREG Root mean squared error(training): 10.99
LREG Root mean squared error(test): 7.90
RMSE_train_in_total: 8.08
RMSE_test_in_total: 5.17
RMSE_train: 3.03
RMSE_test: 2.96
RMSE_train: 6.95
RMSE_test: 6.98
RMSE_train: 11.66
RMSE_test: 11.74
RMSE_train: 8.51
RMSE_test: 10.78
RMSE_train: 21.62
RMSE_test: 35.09

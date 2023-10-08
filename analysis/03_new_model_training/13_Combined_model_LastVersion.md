# Combined  Model (XGBoost Undersampling + XGBoost Regression)

We developed a hybrid model using both xgboost regression and xgboost
classification(while undersampling technique was implemented to enhance
its performance). Subsequently, we evaluated the performance of this
combined model on the test dataset and compared it with the result of
the simple xgboost regression model.

```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import imblearn
import statsmodels.api as sm

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
from sty import fg, rs

from sklearn.metrics import confusion_matrix
from matplotlib import cm
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from utils import get_training_dataset
```

```python
# Read csv file and import to df
df = get_training_dataset()
df.head()
```

```python
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)

# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```

```python
# Remove zeros from wind_speed
df = (df[(df[["wind_speed"]] != 0).any(axis=1)]).reset_index(drop=True)
df = df.drop(columns=["grid_point_id", "typhoon_year"])
df.head()
```

```python
# Define bins for data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
bins_eval = [0, 1, 10, 20, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
```

```python
# Check the bins' intervalls (first bin means all zeros,
# second bin means 0 < values <= 1)
df["percent_houses_damaged"].value_counts(bins=binsP2)
```

(-0.001, 9e-05]    38901
(9e-05, 1.0]        7232
(1.0, 10.0]         2552
(10.0, 50.0]         925
(50.0, 101.0]        144
Name: percent_houses_damaged, dtype: int64

```python
print(samples_per_bin2)
print(binsP2)
```

[38901  7232  2552   925   144]
[0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]

```python
bin_index2 = np.digitize(df["percent_houses_damaged"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
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

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["percent_houses_damaged"]
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
        'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
        'water', 'total_pop', 'percent_houses_damaged_5years'],
      dtype='object')

```python
# Define train and test data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    df["percent_houses_damaged"],
    test_size=0.2,
    stratify=y_input_strat,
)
```

## First step is to train XGBoost Regression model for train data

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
```

```python
# Make prediction on train and test data
y_pred_train = xgb.predict(X_train)
y_pred = xgb.predict(X_test)
```

```python
# Calculate RMSE in total

mse_train_idx = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train_idx)

mse_idx = mean_squared_error(y_test, y_pred)
rmseM1 = np.sqrt(mse_idx)

print(f"RMSE_test_in_total: {rmseM1:.2f}")
print(f"RMSE_train_in_total: {rmse_train:.2f}")
```

RMSE_test_in_total: 3.08
RMSE_train_in_total: 2.65

```python
# Calculate RMSE per bins

bin_index_test = np.digitize(y_test, bins=bins_eval)
bin_index_train = np.digitize(y_train, bins=bins_eval)

RSME_test_model1 = np.zeros(len(bins_eval) - 1)

for bin_num in range(1, len(bins_eval)):

    # Estimation of RMSE for train data
    mse_train_idx = mean_squared_error(
        y_train[bin_index_train == bin_num], y_pred_train[bin_index_train == bin_num]
    )
    rmse_train = np.sqrt(mse_train_idx)

    # Estimation of RMSE for test data
    mse_idx = mean_squared_error(
        y_test[bin_index_test == bin_num], y_pred[bin_index_test == bin_num]
    )
    RSME_test_model1[bin_num - 1] = np.sqrt(mse_idx)

    print(
        f"RMSE_test  [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {RSME_test_model1[bin_num-1]:.2f}"
    )
    print(
        f"RMSE_train [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}): {rmse_train:.2f}"
    )
```

RMSE_test  [0,1): 1.17
RMSE_train [0,1): 0.94
RMSE_test  [1,10): 4.54
RMSE_train [1,10): 3.93
RMSE_test  [10,20): 9.31
RMSE_train [10,20): 9.03
RMSE_test  [20,50): 19.75
RMSE_train [20,50): 15.83
RMSE_test  [50,101): 33.02
RMSE_train [50,101): 28.50

## Second step is to train XGBoost Binary model for same train data

```python
# Define a threshold to separate target into damaged and not_damaged
thres = 10.0
y_test_bool = y_test >= thres
y_train_bool = y_train >= thres
y_test_bin = (y_test_bool) * 1
y_train_bin = (y_train_bool) * 1
```

```python
sum(y_train_bin)
```

855

```python
print(Counter(y_train_bin))
```

Counter({0: 38948, 1: 855})

```python
# Undersampling

# Define undersampling strategy
under = RandomUnderSampler(sampling_strategy=0.1)
# Fit and apply the transform
X_train_us, y_train_us = under.fit_resample(X_train, y_train_bin)

print(Counter(y_train_us))
```

Counter({0: 8550, 1: 855})

```python
# Use XGBClassifier as a Machine Learning model to fit the data
xgb_model = XGBClassifier(eval_metric=["error", "logloss"])

# eval_set = [(X_train, y_train), (X_train, y_train)]
eval_set = [(X_test, y_test_bin)]
xgb_model.fit(
    X_train_us,
    y_train_us,
    eval_set=eval_set,
    verbose=False,
)
```

```python
# Make prediction on test data
y_pred_test = xgb_model.predict(X_test)
```

```python
# Print Confusion Matrix
cm = confusion_matrix(y_test_bin, y_pred_test)
cm
```

array([[9601,  136],
        [  61,  153]])

```python
# Classification Report
print(metrics.classification_report(y_test_bin, y_pred_test))
print(metrics.confusion_matrix(y_test_bin, y_pred_test))
```

precision    recall  f1-score   support

0       0.99      0.99      0.99      9737
1       0.53      0.71      0.61       214

accuracy                           0.98      9951
macro avg       0.76      0.85      0.80      9951
weighted avg       0.98      0.98      0.98      9951

[[9601  136]
[  61  153]]

```python
# Make prediction on train data
y_pred_train = xgb_model.predict(X_train)
```

```python
# Print Confusion Matrix
cm = confusion_matrix(y_train_bin, y_pred_train)
cm
```

array([[38510,   438],
        [    0,   855]])

```python
# Classification Report
print(metrics.classification_report(y_train_bin, y_pred_train))
print(metrics.confusion_matrix(y_train_bin, y_pred_train))
```

precision    recall  f1-score   support

0       1.00      0.99      0.99     38948
1       0.66      1.00      0.80       855

accuracy                           0.99     39803
macro avg       0.83      0.99      0.90     39803
weighted avg       0.99      0.99      0.99     39803

[[38510   438]
[    0   855]]

```python
reduced_df = X_train.copy()
```

```python
reduced_df["percent_houses_damaged"] = y_train.values
reduced_df["predicted_value"] = y_pred_train
```

```python
fliterd_df = reduced_df[reduced_df.predicted_value == 1]
```

```python
fliterd_df
```

### Third step is to train XGBoost regression model for this reduced train data

### (including damg>10.0%)

```python
# Define bins for data stratification in regression model
bins2 = [0, 1, 10, 20, 50, 101]
samples_per_bin2, binsP2 = np.histogram(
    fliterd_df["percent_houses_damaged"], bins=bins2
)

print(samples_per_bin2)
print(binsP2)
```

[168 270 373 367 115]
[  0   1  10  20  50 101]

```python
bin_index2 = np.digitize(fliterd_df["percent_houses_damaged"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
# Split X and y from dataframe features
X_r = fliterd_df[features]
display(X.columns)
y_r = fliterd_df["percent_houses_damaged"]
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
        'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
        'water', 'total_pop', 'percent_houses_damaged_5years'],
      dtype='object')

```python
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
    verbosity=1,
    eval_metric=["rmse", "logloss"],
    random_state=0,
)

eval_set = [(X_r, y_r)]
xgbR_model = xgbR.fit(X_r, y_r, eval_set=eval_set, verbose=False)
```

```python
# Make prediction on train and global test data
y_pred_r = xgbR.predict(X_r)
y_pred_test_total = xgbR.predict(X_test)
```

```python
# Calculate RMSE in total

mse_train_idxR = mean_squared_error(y_r, y_pred_r)
rmse_trainR = np.sqrt(mse_train_idxR)


mse_idxR = mean_squared_error(y_test, y_pred_test_total)
rmseR = np.sqrt(mse_idxR)

print(f"RMSE_test_in_total MR: {rmseR:.2f}")
print(f"RMSE_test_in_total M1: {rmseM1:.2f}")
print(f"RMSE_train_in_reduced: {rmse_trainR:.2f}")
```

RMSE_test_in_total MR: 15.44
RMSE_test_in_total M1: 3.08
RMSE_train_in_reduced: 10.01

```python
# Calculate RMSE per bins
bin_index_r = np.digitize(y_r, bins=bins_eval)

RSME_test_model1R = np.zeros(len(bins_eval) - 1)
for bin_num in range(1, len(bins_eval)):

    # Estimation of RMSE for train data
    mse_train_idxR = mean_squared_error(
        y_r[bin_index_r == bin_num], y_pred_r[bin_index_r == bin_num]
    )
    rmse_trainR = np.sqrt(mse_train_idxR)

    # Estimation of RMSE for test data
    mse_idxR = mean_squared_error(
        y_test[bin_index_test == bin_num], y_pred_test_total[bin_index_test == bin_num]
    )
    RSME_test_model1R[bin_num - 1] = np.sqrt(mse_idxR)

    # print(f"RMSE_test: {rmse:.2f}")
    print(
        f"RMSE_train_reduced [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}):
        {rmse_trainR:.2f}"
    )
    print(
        f"RMSE_test_total_MR [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}):
        {RSME_test_model1R[bin_num-1]:.2f}"
    )
    print(
        f"RMSE_test_total_M1 [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}):
        {RSME_test_model1[bin_num-1]:.2f}"
    )
    RSME_test_model1
    # print(f"RMSE_train: {rmse_train:.2f}")
```

RMSE_train_reduced [0,1): 11.16
RMSE_test_total_MR [0,1): 15.60
RMSE_test_total_M1 [0,1): 1.17
RMSE_train_reduced [1,10): 8.32
RMSE_test_total_MR [1,10): 12.26
RMSE_test_total_M1 [1,10): 4.54
RMSE_train_reduced [10,20): 4.63
RMSE_test_total_MR [10,20): 6.80
RMSE_test_total_M1 [10,20): 9.31
RMSE_train_reduced [20,50): 9.69
RMSE_test_total_MR [20,50): 15.19
RMSE_test_total_M1 [20,50): 19.75
RMSE_train_reduced [50,101): 20.31
RMSE_test_total_MR [50,101): 30.13
RMSE_test_total_M1 [50,101): 33.02

## Last step is to add model combination (model M1 with model MR)

```python
# Check the result of classifier for test set
reduced_test_df = X_test.copy()
```

```python
# joined X_test with countinous target and binary predicted values
reduced_test_df["percent_houses_damaged"] = y_test.values
reduced_test_df["predicted_value"] = y_pred_test

reduced_test_df
```

```python
# damaged prediction
fliterd_test_df1 = reduced_test_df[reduced_test_df.predicted_value == 1]

# not damaged prediction
fliterd_test_df0 = reduced_test_df[reduced_test_df.predicted_value == 0]
```

```python
# Use X0 and X1 for the M1 and MR models' predictions
X1 = fliterd_test_df1[features]
X0 = fliterd_test_df0[features]
```

```python
# For the output equal to 1 apply MR to evaluate the performance
y1_pred = xgbR.predict(X1)
y1 = fliterd_test_df1["percent_houses_damaged"]
```

```python
# For the output equal to 0 apply M1 to evaluate the performance
y0_pred = xgb.predict(X0)
y0 = fliterd_test_df0["percent_houses_damaged"]
```

```python
## Combined the two outputs
```

```python
fliterd_test_df0["predicted_percent_damage"] = y0_pred
fliterd_test_df0
```

```python
fliterd_test_df1["predicted_percent_damage"] = y1_pred
fliterd_test_df1
```

```python
# Join two dataframes together

join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])
join_test_dfs
```

```python
# join_test_dfs = join_test_dfs.reset_index(drop=True)
```

### Compare performance of M1 with combined model

```python
# Calculate RMSE in total

mse_combined_model = mean_squared_error(
    join_test_dfs["percent_houses_damaged"], join_test_dfs["predicted_percent_damage"]
)
rmse_combined_model = np.sqrt(mse_combined_model)


print(fg.red + f"RMSE_in_total(combined_model): {rmse_combined_model:.2f}" + fg.rs)
print(f"RMSE_in_total(M1_model): {rmseM1:.2f}")
```

[31mRMSE_in_total(combined_model): 3.20[39m
RMSE_in_total(M1_model): 3.08

```python
# Calculate RMSE per bin

y_join = join_test_dfs["percent_houses_damaged"]
y_pred_join = join_test_dfs["predicted_percent_damage"]

bin_index_test = np.digitize(y_join, bins=bins_eval)

RSME_combined_model = np.zeros(len(bins_eval) - 1)

for bin_num in range(1, len(bins_eval)):

    mse_combined_model = mean_squared_error(
        y_join[bin_index_test == bin_num],
        y_pred_join[bin_index_test == bin_num],
    )
    RSME_combined_model[bin_num - 1] = np.sqrt(mse_combined_model)

    print(
        fg.red
        + f"RMSE_combined_model [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}):
        {RSME_combined_model[bin_num-1]:.2f}"
        + fg.rs
    )

    print(
        f"RMSE_M1_model       [{bins_eval[bin_num-1]:.0f},{bins_eval[bin_num]:.0f}):
        {RSME_test_model1[bin_num-1]:.2f}"
    )
    print("\n")
```

[31mRMSE_combined_model [0,1): 1.55[39m
RMSE_M1_model       [0,1): 1.17

[31mRMSE_combined_model [1,10): 5.56[39m
RMSE_M1_model       [1,10): 4.54

[31mRMSE_combined_model [10,20): 9.10[39m
RMSE_M1_model       [10,20): 9.31

[31mRMSE_combined_model [20,50): 17.79[39m
RMSE_M1_model       [20,50): 19.75

[31mRMSE_combined_model [50,101): 32.35[39m
RMSE_M1_model       [50,101): 33.02

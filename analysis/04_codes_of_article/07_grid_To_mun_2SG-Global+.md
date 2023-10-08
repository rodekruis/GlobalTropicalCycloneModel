# Converting from grid-based to municipality-based

## GridGlobal++

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
from sty import fg, rs

from sklearn.metrics import confusion_matrix
from matplotlib import cm
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from utils import get_training_dataset, weight_file
```

```python
# Hide all warnings
warnings.filterwarnings("ignore")
```

```python
# Import the created dataset to a df
df = get_training_dataset()
df
```

```python
for i in range(len(df)):
    df.at[i, "typhoon_year"] = str(df.loc[i]["typhoon_year"])
```

```python
df["typhoon_name"] = df["typhoon_name"] + df["typhoon_year"]
```

```python
# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```

```python
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)
```

```python
# Read the new weight CSV file and import to a df
df_weight = weight_file("/ggl_grid_to_mun_weights.csv")
df_weight.head()
```

```python
# Change name of column ['id'] to ['grid_point_id'] the same name as in input df
df_weight.rename(columns={"id": "grid_point_id"}, inplace=True)
df_weight.head()
```

### Following Steps are to convert grid_based model into Municipality based one

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df_data = df.drop(columns=["grid_point_id", "typhoon_year"])
```

```python
display(df.head())
display(df_data.head())
```

```python
# Read municipality dataset which already merged with y_norm converted ground truth
df_mun_merged = pd.read_csv("data/df_merged_2.csv")

# Remove the duplicated rows
df_mun_merged.drop_duplicates(keep="first", inplace=True)
df_mun_merged = df_mun_merged.reset_index(drop=True)

# Make the name of typhoons to uppercase
df_mun_merged["typhoon"] = df_mun_merged["typhoon"].str.upper()

# Rename y_norm column
df_mun_merged = df_mun_merged.rename(columns={"y_norm": "y_norm_mun"})

df_mun_merged
```

```python
# Specify features
features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "rwi",
    # "strong_roof_strong_wall",
    # "strong_roof_light_wall",
    # "strong_roof_salvage_wall",
    # "light_roof_strong_wall",
    # "light_roof_light_wall",
    # "light_roof_salvage_wall",
    # "salvaged_roof_strong_wall",
    # "salvaged_roof_light_wall",
    # "salvaged_roof_salvage_wall",
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
X = df_data[features]
display(X.columns)
y = df_data["percent_houses_damaged"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
'water', 'total_pop', 'percent_houses_damaged_5years'],
dtype='object')

```python
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df_data["percent_houses_damaged"], bins=bins2)
```

```python
# Define range of for loop
num_exp = 20

# Define number of bins
num_bins = len(bins2)
```

```python
M1_RMSE_lst = defaultdict(list)
Combined_RMSE_lst = defaultdict(list)
```

```python
# Define empty list to save RMSE in combined model
test_RMSE_lst = np.zeros(num_exp)
test_RMSE_bin = np.zeros((num_exp, num_bins))

# Define empty list to save RMSE in model1
test_RMSE_lst_M1 = np.zeros(num_exp)
test_RMSE_bin_M1 = np.zeros((num_exp, num_bins))
```

```python
# Defin two lists to save RMSE and Average Error

RMSE = defaultdict(list)
AVE = defaultdict(list)
```

```python
for run_ix in range(num_exp):

    bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
    y_input_strat = bin_index2

    X = df_data[features]
    y = df_data["percent_houses_damaged"]

    # Define train and test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, df_data["percent_houses_damaged"], test_size=0.2, stratify=y_input_strat
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
    test_RMSE_lst_M1[run_ix] = rmseM1

    # Calculate RMSE per bins
    bin_index_test = np.digitize(y_test, bins=binsP2)
    bin_index_train = np.digitize(y_train, bins=binsP2)

    RSME_test_model1 = np.zeros(num_bins - 1)

    for bin_num in range(1, num_bins):

        # Estimation of RMSE for train data
        mse_train_idx = mean_squared_error(
            y_train[bin_index_train == bin_num],
            y_pred_train[bin_index_train == bin_num],
        )
        rmse_train = np.sqrt(mse_train_idx)

        # Estimation of RMSE for test data
        mse_idx = mean_squared_error(
            y_test[bin_index_test == bin_num], y_pred[bin_index_test == bin_num]
        )

        RSME_test_model1 = np.sqrt(mse_idx)

        # Add RMSE of Model1 to the list of each bin
        test_RMSE_bin_M1[run_ix, bin_num] = RSME_test_model1
        M1_RMSE_lst[bin_num].append(RSME_test_model1)

    ## Second step is to train XGBoost Binary model for same train data

    # Define a threshold to separate target into damaged and not_damaged
    thres = 10.0
    y_test_bool = y_test >= thres
    y_train_bool = y_train >= thres
    y_test_bin = (y_test_bool) * 1
    y_train_bin = (y_train_bool) * 1

    sum(y_train_bin)

    # Define undersampling strategy
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
    cm = confusion_matrix(y_test_bin, y_pred_test)

    # Make prediction on train data and print Confusion Matrix
    y_pred_train = xgb_model.predict(X_train)
    cm = confusion_matrix(y_train_bin, y_pred_train)

    reduced_df = X_train.copy()

    reduced_df["percent_houses_damaged"] = y_train.values
    reduced_df["predicted_value"] = y_pred_train

    fliterd_df = reduced_df[reduced_df.predicted_value == 1]

    ### Third step is to train XGBoost regression model for this
    # reduced train data (including damg>10.0%)
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

    # Calculate RMSE per bins
    bin_index_r = np.digitize(y_r, bins=binsP2)

    RSME_test_model1R = np.zeros(num_bins - 1)

    for bin_num in range(1, num_bins):

        # Estimation of RMSE for train data
        mse_train_idxR = mean_squared_error(
            y_r[bin_index_r == bin_num], y_pred_r[bin_index_r == bin_num]
        )
        rmse_trainR = np.sqrt(mse_train_idxR)

        # Estimation of RMSE for test data
        mse_idxR = mean_squared_error(
            y_test[bin_index_test == bin_num],
            y_pred_test_total[bin_index_test == bin_num],
        )
        RSME_test_model1R[bin_num - 1] = np.sqrt(mse_idxR)

    #### Last step is to add model combination (model M1 with model MR)

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
    y1_pred = y1_pred.clip(0, 100)
    y1 = fliterd_test_df1["percent_houses_damaged"]

    # For the output equal to 0 apply M1 to evaluate the performance
    y0_pred = xgb.predict(X0)
    y0_pred = y0_pred.clip(0, 100)
    y0 = fliterd_test_df0["percent_houses_damaged"]

    fliterd_test_df0["predicted_percent_damage"] = y0_pred
    fliterd_test_df1["predicted_percent_damage"] = y1_pred

    # Join two dataframes together
    join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])

    y_join = join_test_dfs["percent_houses_damaged"]
    y_pred_join = join_test_dfs["predicted_percent_damage"]

    pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])
    pred_df["y_all"] = y_join
    pred_df["y_pred_all"] = y_pred_join

    # bin_index = np.digitize(pred_df["y_all"], bins=binsP2)

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
    agg_df = join_df.groupby(["ADM3_PCODE", "typhoon_name", "typhoon_year"]).agg("sum")

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
    final_df = final_df.reset_index()
    print(len(final_df))
    # print(final_df)

    # Intersection of two datasets grid and municipality
    # Rename a column
    final_df = final_df.rename(
        columns={"ADM3_PCODE": "Mun_Code", "typhoon_name": "typhoon"}
    )

    # print(final_df)
    # Merge DataFrames based on 'typhoon_name' and 'Mun_Code'
    merged_df = pd.merge(
        final_df, df_mun_merged, on=["Mun_Code", "typhoon"], how="inner"
    )
    print(len(merged_df))
    # print(merged_df)

    # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based
    #if (len(merged_df["y_norm"])) != 0:
    rmse = sqrt(mean_squared_error(merged_df["y_norm"], merged_df["y_pred_norm"]))
    ave = (merged_df["y_pred_norm"] - merged_df["y_norm"]).sum() / len(
        merged_df["y_norm"]
    )

    print(f"RMSE for grid_based model: {rmse:.2f}")
    print(f"Average Error for grid_based model: {ave:.2f}")

    RMSE["all"].append(rmse)
    AVE["all"].append(ave)

    bin_index = np.digitize(merged_df["y_norm"], bins=binsP2)

    #for bin_num in range(1, 6):
    if len(merged_df["y_norm"][bin_index == bin_num]) > 0:

        mse_idx = mean_squared_error(
            merged_df["y_norm"][bin_index == bin_num],
            merged_df["y_pred_norm"][bin_index == bin_num],
        )
        rmse = np.sqrt(mse_idx)

        ave = (
            merged_df["y_pred_norm"][bin_index == bin_num]
            - merged_df["y_norm"][bin_index == bin_num]
        ).sum() / len(merged_df["y_norm"][bin_index == bin_num])

        RMSE[bin_num].append(rmse)
        AVE[bin_num].append(ave)
```

16260
5002
RMSE for grid_based model: 4.56
Average Error for grid_based model: -0.06
16345
4936
RMSE for grid_based model: 5.13
Average Error for grid_based model: -0.06
16388
5020
RMSE for grid_based model: 4.74
Average Error for grid_based model: -0.14
16445
4986
RMSE for grid_based model: 5.12
Average Error for grid_based model: -0.21
16345
5008
RMSE for grid_based model: 4.78
Average Error for grid_based model: -0.28
16284
4962
RMSE for grid_based model: 4.58
Average Error for grid_based model: -0.07
16179
5013
RMSE for grid_based model: 4.69
Average Error for grid_based model: 0.04
16456
5030
RMSE for grid_based model: 5.10
Average Error for grid_based model: -0.13
16152
4917
RMSE for grid_based model: 4.68
Average Error for grid_based model: 0.01
16400
5083
RMSE for grid_based model: 4.59
Average Error for grid_based model: 0.02
16042
4945
RMSE for grid_based model: 4.48
Average Error for grid_based model: -0.08
16269
4871
RMSE for grid_based model: 4.25
Average Error for grid_based model: -0.02
16282
5013
RMSE for grid_based model: 4.79
Average Error for grid_based model: -0.09
16301
5096
RMSE for grid_based model: 4.59
Average Error for grid_based model: 0.11
16409
4978
RMSE for grid_based model: 4.54
Average Error for grid_based model: -0.18
16482
4939
RMSE for grid_based model: 5.42
Average Error for grid_based model: -0.04
16344
4900
RMSE for grid_based model: 4.41
Average Error for grid_based model: -0.06
16311
4955
RMSE for grid_based model: 4.69
Average Error for grid_based model: 0.01
16249
4921
RMSE for grid_based model: 4.93
Average Error for grid_based model: -0.05
16264
4958
RMSE for grid_based model: 4.52
Average Error for grid_based model: 0.04

```python
# Define a function to plot RMSEs
def rmse_ave_mean(rmse, ave):

    # Mean of RMSE and Standard deviation
    m_rmse = statistics.mean(rmse)
    sd_rmse = statistics.stdev(rmse)

    m_ave = statistics.mean(ave)
    sd_ave = statistics.stdev(ave)

    print(f"mean_RMSE: {m_rmse:.2f}")
    print(f"stdev_RMSE: {sd_rmse:.2f}")

    print(f"mean_average_error: {m_ave:.2f}")
    print(f"stdev_average_error: {sd_ave:.2f}")
```

```python
print("RMSE and Average Error in total", "\n")
rmse_ave_mean(RMSE["all"], AVE["all"])
```

RMSE and Average Error in total

mean_RMSE: 4.73
stdev_RMSE: 0.28
mean_average_error: -0.06
stdev_average_error: 0.09

```python
for bin_num in range(1, 6):

    print(f"\n RMSE and Average Error per bin {bin_num}\n")
    rmse_ave_mean(RMSE[bin_num], AVE[bin_num])
```

RMSE and Average Error per bin 1

mean_RMSE: 0.39
stdev_RMSE: 0.20
mean_average_error: 0.04
stdev_average_error: 0.01

RMSE and Average Error per bin 2

mean_RMSE: 1.94
stdev_RMSE: 0.17
mean_average_error: 0.67
stdev_average_error: 0.05

RMSE and Average Error per bin 3

mean_RMSE: 5.64
stdev_RMSE: 0.59
mean_average_error: 1.00
stdev_average_error: 0.35

RMSE and Average Error per bin 4

mean_RMSE: 12.48
stdev_RMSE: 0.90
mean_average_error: -4.53
stdev_average_error: 0.97

RMSE and Average Error per bin 5

mean_RMSE: 31.67
stdev_RMSE: 3.97
mean_average_error: -25.39
stdev_average_error: 3.97

```python

```

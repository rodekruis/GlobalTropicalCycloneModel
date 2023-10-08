# Naive baseline

## Converting from grid-based to municipality-based

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
# Import the created dataset to a df
df = get_training_dataset()
df
```

```python
# Add the typhoon year to typhoon names
for i in range(len(df)):
    df.at[i, "typhoon_year"] = str(df.loc[i]["typhoon_year"])

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
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df_data["percent_houses_damaged"], bins=bins2)
```

```python
bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
# Define the length of for loop
num_exp = 20
```

```python
# Define X and y data
# We define a vector of all 0 with same length of y for X data
X = pd.Series([0] * 49754)
y = df_data["percent_houses_damaged"]
```

```python
# Define two lists to save RMSE and Average Error

RMSE = defaultdict(list)
AVE = defaultdict(list)
```

```python
for i in range(num_exp):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df_data["percent_houses_damaged"],
        stratify=y_input_strat,
        test_size=0.2,
    )

    # create a dummy regressor
    dummy_reg = DummyRegressor(strategy="mean")

    # fit it on the training set
    dummy_reg.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = dummy_reg.predict(X_test)
    y_pred_clipped = y_pred.clip(0, 100)

    pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])
    pred_df["y_all"] = y_test
    pred_df["y_pred_all"] = y_pred_clipped

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

    # Intersection of two datasets grid and municipality
    # Rename a column
    final_df = final_df.rename(
        columns={"ADM3_PCODE": "Mun_Code", "typhoon_name": "typhoon"}
    )

    # Merge DataFrames based on 'typhoon_name' and 'Mun_Code'
    merged_df = pd.merge(
        final_df, df_mun_merged, on=["Mun_Code", "typhoon"], how="inner"
    )
    print(len(merged_df))

    # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based
    rmse = sqrt(mean_squared_error(merged_df["y_norm"], merged_df["y_pred_norm"]))
    ave = (merged_df["y_pred_norm"] - merged_df["y_norm"]).sum() / len(
        merged_df["y_norm"]
    )

    print(f"RMSE for grid_based model: {rmse:.2f}")
    print(f"Average Error for grid_based model: {ave:.2f}")

    RMSE["all"].append(rmse)
    AVE["all"].append(ave)

    bin_index = np.digitize(merged_df["y_norm"], bins=binsP2)

    for bin_num in range(1, 6):

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

16277
4936
RMSE for grid_based model: 8.74
Average Error for grid_based model: -1.57
16460
4994
RMSE for grid_based model: 8.20
Average Error for grid_based model: -1.45
16317
4959
RMSE for grid_based model: 8.32
Average Error for grid_based model: -1.47
16327
4948
RMSE for grid_based model: 8.36
Average Error for grid_based model: -1.42
16310
5013
RMSE for grid_based model: 8.25
Average Error for grid_based model: -1.44
16314
4981
RMSE for grid_based model: 8.58
Average Error for grid_based model: -1.48
16107
4891
RMSE for grid_based model: 8.03
Average Error for grid_based model: -1.40
16309
4972
RMSE for grid_based model: 8.31
Average Error for grid_based model: -1.43
16222
4899
RMSE for grid_based model: 7.93
Average Error for grid_based model: -1.40
16289
4902
RMSE for grid_based model: 8.00
Average Error for grid_based model: -1.38
16327
5012
RMSE for grid_based model: 8.18
Average Error for grid_based model: -1.39
16521
5055
RMSE for grid_based model: 8.37
Average Error for grid_based model: -1.43
16300
4905
RMSE for grid_based model: 8.55
Average Error for grid_based model: -1.47
16384
4989
RMSE for grid_based model: 8.35
Average Error for grid_based model: -1.39
16547
4986
RMSE for grid_based model: 7.97
Average Error for grid_based model: -1.36
16341
4946
RMSE for grid_based model: 8.03
Average Error for grid_based model: -1.44
16281
4859
RMSE for grid_based model: 7.87
Average Error for grid_based model: -1.34
16305
5057
RMSE for grid_based model: 8.30
Average Error for grid_based model: -1.41
16221
4925
RMSE for grid_based model: 8.54
Average Error for grid_based model: -1.48
16156
4951
RMSE for grid_based model: 8.14
Average Error for grid_based model: -1.41

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

mean_RMSE: 8.25
stdev_RMSE: 0.24
mean_average_error: -1.43
stdev_average_error: 0.05

```python
for bin_num in range(1, 6):

    print(f"\n RMSE and Average Error per bin {bin_num}\n")
    rmse_ave_mean(RMSE[bin_num], AVE[bin_num])
```

  RMSE and Average Error per bin 1

mean_RMSE: 0.83
stdev_RMSE: 0.00
mean_average_error: 0.83
stdev_average_error: 0.00

  RMSE and Average Error per bin 2

mean_RMSE: 0.68
stdev_RMSE: 0.01
mean_average_error: 0.64
stdev_average_error: 0.01

  RMSE and Average Error per bin 3

mean_RMSE: 3.69
stdev_RMSE: 0.12
mean_average_error: -2.80
stdev_average_error: 0.12

  RMSE and Average Error per bin 4

mean_RMSE: 25.34
stdev_RMSE: 0.92
mean_average_error: -22.57
stdev_average_error: 0.81

  RMSE and Average Error per bin 5

mean_RMSE: 63.97
stdev_RMSE: 2.35
mean_average_error: -62.72
stdev_average_error: 2.16

```python

```

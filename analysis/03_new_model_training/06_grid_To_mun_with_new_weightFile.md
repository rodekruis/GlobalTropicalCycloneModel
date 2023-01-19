# Converting from grid-based to municipality-based

The following steps were done to convert created grid_based dataset to municipality_based one:

import the weight file to a dataframe

assign the values to each grid 

multiply the damaged values with the weights

Aggregate the values by municipality and typhoon_name 

```python
%load_ext jupyter_black
```

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

from utils import get_training_dataset, weight_file
```

```python
# Import the created dataset to a df
df = get_training_dataset()
df
```

```python
# Read the weight CSV file and import to a df
df_weight_m2c = weight_file("/phl_bld_weight_matrix.csv")
df_weight_m2c.drop("Unnamed: 0", axis=1, inplace=True)

# Convert id columns from float to int
df_weight_m2c["grid_point_id"] = df_weight_m2c["id"].astype(np.int64)
df_weight_m2c.drop(columns="id", inplace=True)
df_weight_m2c
```

```python
# check if weight_sum per municiplaity == 1
agg_m2c = df_weight_m2c.groupby(["ADM3_PCODE"]).agg("sum")
agg_m2c.sort_values("weight", ascending=True)
```

```python
# values not summing up to 1
agg_m2c[(agg_m2c["weight"] < 0.99)].sort_values("weight", ascending=True)
```

```python
# Read the new weight CSV file and import to a df
df_weight = weight_file("/phl_bld_grid_to_municip_matrix.csv")
df_weight.drop("Unnamed: 0", axis=1, inplace=True)

# Convert id columns from float to int
df_weight["grid_point_id"] = df_weight["id"].astype(np.int64)
df_weight.drop(columns="id", inplace=True)
df_weight
```

```python
# check if weight_sum per cell == 1
agg_c2m = df_weight.groupby(["grid_point_id"]).agg("sum")
agg_c2m.sort_values("weight", ascending=True)
```

```python
# values not summing up to 1
agg_c2m[(agg_c2m["weight"] < 0.99) & (agg_c2m["weight"] > 0)].sort_values(
    "weight", ascending=True
)
```

### Following Steps are to convert grid_based model into Municipality based one

```python
# Hist plot after data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_buildings_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

```python
# Check the bins' intervalls
df["percent_buildings_damaged"].value_counts(bins=binsP2)
```

```python
# Remove zeros from wind_speed
# df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df_data = df.drop(columns=["grid_point_id", "typhoon_year"])
```

```python
display(df.head())
display(df_data.head())
```

```python
# Hist plot after removing rows where windspeed is 0
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(
    df_data["percent_buildings_damaged"], bins=bins2
)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

```python
print(samples_per_bin2)
print(binsP2)
```

```python
# Check the bins' intervalls
df_data["percent_buildings_damaged"].value_counts(bins=binsP2)
```

```python
bin_index2 = np.digitize(df_data["percent_buildings_damaged"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
features = [
    "wind_speed",
    # "track_distance",
    # "total_buildings",
    # "rainfall_max_6h",
    # "rainfall_max_24h"
]

# Split X and y from dataframe features
X = df_data[features]
display(X.columns)
y = df_data["percent_buildings_damaged"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```

```python
# Define train-test-split function

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df_data["percent_buildings_damaged"],
    stratify=y_input_strat,
    test_size=0.2,
)

# Define XGBoost Reduced Overfitting model
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
```

```python
# Combine test and training set to have predictiob for all
y_pred_all = xgb.predict(X_scaled)
pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])
pred_df["y_all"] = df_data["percent_buildings_damaged"]
pred_df["y_pred_all"] = y_pred_all

pred_df
```

```python
rmse = sqrt(mean_squared_error(pred_df["y_all"], pred_df["y_pred_all"]))
print(f"RMSE for grid_based model: {rmse:.2f}")
```

```python
bin_index = np.digitize(pred_df["y_all"], bins=binsP2)
```

```python
# Join data with y_all and y_all_pred
df_data_w_pred = pd.merge(pred_df, df_data, left_index=True, right_index=True)
# Join data with grid_point_id typhoon_year
df_data_w_pred_grid = pd.merge(
    df[["grid_point_id", "typhoon_year"]],
    df_data_w_pred,
    left_index=True,
    right_index=True,
)
```

```python
df_data_w_pred_grid.sort_values("y_pred_all", ascending=False)
```

```python
# join with weights df
join_df = df_data_w_pred_grid.merge(df_weight, on="grid_point_id", how="left")
join_df
```

```python
# Indicate where values are valid and not missing
join_df = join_df.loc[join_df["weight"].notna()]
join_df
```

```python
# Multiply weight by y_all and y_pred_all
join_df["weight*y_pred*buildings"] = (
    join_df["y_pred_all"] * join_df["weight"] * join_df["total_buildings"] / 100
)
join_df["weight*y*buildings"] = (
    join_df["y_all"] * join_df["weight"] * join_df["total_buildings"] / 100
)
join_df["weight*buildings"] = join_df["weight"] * join_df["total_buildings"]
join_df
```

```python
join_df.sort_values("y_pred_all", ascending=False)
```

```python
# Groupby by municipality and typhoon_name with sum as the aggregation function
agg_df = join_df.groupby(["ADM3_PCODE", "typhoon_name", "typhoon_year"]).agg("sum")

# Normalize by the sum of the weights
agg_df["y_pred_norm"] = (
    agg_df["weight*y_pred*buildings"] / agg_df["weight*buildings"] * 100
)
agg_df["y_norm"] = agg_df["weight*y*buildings"] / agg_df["weight*buildings"] * 100

# Drop not required column y and y_pred before multiplying by weight
agg_df.drop("y_all", axis=1, inplace=True)
agg_df.drop("y_pred_all", axis=1, inplace=True)

agg_df
```

```python
# agg_df.isnull().values.any()

# Remove rows with NaN after normalization
final_df = agg_df.dropna()
final_df
```

```python
# Calculate RMSE in total for converted grid_based model to Mun_based

rmse = sqrt(mean_squared_error(final_df["y_norm"], final_df["y_pred_norm"]))
print(f"RMSE for grid_based model: {rmse:.2f}")
```

```python
bin_index = np.digitize(final_df["y_norm"], bins=binsP2)
```

### Calculate RMSE per bin for converted grid_based model to Mun_based

```python
# Define a function to estimate RMSE per bin


def rmse_bin(n):
    mse = mean_squared_error(
        final_df["y_norm"][bin_index == n],
        final_df["y_pred_norm"][bin_index == n],
    )
    rmse = np.sqrt(mse)
    print(f"RMSE per bin_{n}: {rmse:.2f}")


for bin_num in range(1, 6):
    rmse_bin(bin_num)
```

### Check if y_norm is the the same as the damage ground truth in the original model

```python
# Read the weight CSV file and import to df
df_old_data = pd.read_csv("data/old_data.csv")
df_old_data.drop("Unnamed: 0", axis=1, inplace=True)
df_old_data.columns = df_old_data.columns.str.replace("Mun_Code", "ADM3_PCODE")
df_old_data.head()
```

```python
# Capitalize strings typhoon column and change the typhoon column's name
for i in range(len(df_old_data)):
    df_old_data.at[i, "typhoon_name"] = df_old_data.loc[i, "typhoon"].upper()

del df_old_data["typhoon"]
df_old_data
```

```python
df_old_data["typhoon_year"] = df_old_data["typhoon_name"].str[-4:]
df_old_data["typhoon_name"] = df_old_data["typhoon_name"].str[:-4]
df_old_data["typhoon_year"] = df_old_data["typhoon_year"].astype("int64")
df_old_data
```

```python
# Remove year from typhoons' name
# for i in range(len(df_old_data)):
#    if df_old_data.at[i, "typhoon_name"] == "GONI2015":
#        df_old_data.loc[i, "typhoon_name"] = "GONI2015"

#    elif df_old_data.at[i, "typhoon_name"] == "GONI2020":
#        df_old_data.loc[i, "typhoon_name"] = "GONI2020"

#    else:
#        df_old_data.loc[i, "typhoon_name"] = df_old_data.loc[i, "typhoon_name"][:-4]
# df_old_data
```

```python
agg_df_old_data = df_old_data.groupby(["ADM3_PCODE"]).agg("sum")
agg_df_old_data
```

```python
df_merged = df_old_data.merge(
    final_df,
    how="left",
    on=["ADM3_PCODE", "typhoon_name", "typhoon_year"],
)

df_merged
```

```python
print(df_merged["DAM_perc_dmg"].corr(df_merged["y_norm"]))
print(df_merged["DAM_perc_dmg"].corr(df_merged["y_pred_norm"]))
```

```python
x = df_merged["DAM_perc_dmg"]
y = df_merged["y_norm"]
plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 100})
plt.scatter(x, y, c=y, cmap="Spectral")
plt.colorbar()
plt.title("Scatter plot of damaged in original model and damaged in grid_to_mun")
plt.xlabel("damaged_data")
plt.ylabel("y_norm")
plt.show()
```

```python
# df_merged.isnull().sum()
df_merged_2 = df_merged.dropna()
```

```python
df_merged_2.reset_index(drop=True, inplace=True)
df_merged_2
```

```python
for i in range(len(df_merged_2)):
    if df_merged_2.loc[i, "DAM_perc_dmg"] == df_merged_2.loc[i, "y_norm"]:
        df_merged_2.at[i, "compare"] = "True"
    elif df_merged_2.loc[i, "DAM_perc_dmg"] != df_merged_2.loc[i, "y_norm"]:
        df_merged_2.at[i, "compare"] = "False"


df_merged_2
```

```python
df_merged_2 = df_merged_2[~df_merged_2.select_dtypes(["object"]).eq("True").any(1)]
df_merged_2
```

```python
print(df_merged_2["DAM_perc_dmg"].corr(df_merged_2["y_norm"]))
```

```python
x = df_merged_2["DAM_perc_dmg"]
y = df_merged_2["y_norm"]
plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 100})
plt.scatter(x, y, c=y, cmap="Spectral")
plt.colorbar()
plt.title("Scatter plot of damaged in original model and damaged in grid_to_mun")
plt.xlabel("damaged_data")
plt.ylabel("y_norm")
plt.show()
```

```python
diff = df_merged["y_norm"] - df_merged["DAM_perc_dmg"]
diff.hist(bins=40, figsize=(8, 6))
plt.title("Histogram of diff")
plt.xlabel("diff")
plt.ylabel("Frequency")
```

```python

```

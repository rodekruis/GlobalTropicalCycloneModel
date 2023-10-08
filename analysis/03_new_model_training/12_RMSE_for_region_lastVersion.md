# RMSE estimation for region(adm1)

We used a weight and adm3_area files to join weight and adm_1(region)
and adm_3(municipality) to our main dataset.
We prepared a dataframe that represents real and damaged value per region(ADM1).
Then we train our model(XGBoost Reduced Overfitting) to this input data while
we splitted five typhoons(randomly selected) as the test set and the rest of
them as the train set.
The final goal is to estimate the difference between real and predicted damage
value per region with respect to each typhoon, to check how the model performs
for a wide area.

```python
%load_ext jupyter_black
```

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import statistics

from utils import get_training_dataset, weight_file
```

```python
# Read csv file and import to df
df = get_training_dataset()

# Move target to be the last column for simplicity
df = df.reindex(
    columns=[col for col in df.columns if col != "percent_houses_damaged"]
    + ["percent_houses_damaged"]
)

df.head()
```

```python
# df.loc[df["typhoon_name"] == "GONI"]
```

```python
# Fill the missing values of RWI with mean value
df["rwi"].fillna(df["rwi"].mean(), inplace=True)
```

```python
# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df.reset_index(drop=True, inplace=True)
df = df.drop(columns=["typhoon_year"])
df.head()
```

```python
# Define bins for data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
```

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
# Use MinMaxScaler function for data standardization
# (it normalaize data in range of [0,1] and not negative values)

# Separate typhoon from other features
dfs = np.split(df, [2], axis=1)
dfa = np.split(dfs[1], [27], axis=1)
# print(dfs[0], dfs[1], dfa[0], dfa[1])

# Standardaize data
scaler = MinMaxScaler().fit(dfa[0])
X1 = scaler.transform(dfa[0])
Xnew = pd.DataFrame(X1)
Xnew_per_pred = pd.DataFrame(X1)
display(Xnew)
```

```python
# All df without target column
dfa[0]
```

```python
dfa[1] = dfa[1].astype(float)
```

```python
Xnew = pd.concat([Xnew.reset_index(drop=True), dfa[1].reset_index(drop=True)],
    axis=1)
Xnew
```

```python
features = [
    "wind_speed",
    "track_distance",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "total_houses",
    "rwi",
    "strong_roof_strong_wall",
    "strong_roof_light_wall",
    "strong_roof_salvage_wall",
    "light_roof_strong_wall",
    "light_roof_light_wall",
    "light_roof_salvage_wall",
    "salvaged_roof_strong_wall",
    "salvaged_roof_light_wall",
    "salvaged_roof_salvage_wall",
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
# Add the features to the columns' headers after standardization
i = 0
for feature in features:
    Xnew = Xnew.rename(columns={i: feature})
    i += 1

Xnew = pd.concat([dfs[0].reset_index(drop=True), Xnew.reset_index(drop=True)],
    axis=1)
Xnew
```

```python
df["typhoon_name"].unique()
```

array(['DURIAN', 'FENGSHEN', 'KETSANA', 'CONSON', 'NESAT', 'BOPHA',
        'NARI', 'KROSA', 'HAIYAN', 'USAGI', 'UTOR', 'JANGMI', 'KALMAEGI',
        'RAMMASUN', 'HAGUPIT', 'FUNG-WONG', 'LINGLING', 'MUJIGAE', 'MELOR',
        'NOUL', 'GONI', 'LINFA', 'KOPPU', 'MEKKHALA', 'HAIMA', 'TOKAGE',
        'MERANTI', 'NOCK-TEN', 'SARIKA', 'MANGKHUT', 'YUTU', 'KAMMURI',
        'NAKRI', 'PHANFONE', 'SAUDEL', 'VAMCO', 'VONGFONG', 'MOLAVE'],
      dtype=object)

```python
# Define a test_list (including 5 typhoons) randomly were chosen
test_list_1 = ["FENGSHEN", "DURIAN", "NESAT", "VONGFONG", "MOLAVE"]

test_list_2 = ["YUTU", "KAMMURI", "SARIKA", "TOKAGE", "LINGLING"]

test_list_3 = ["SAUDEL", "MANGKHUT", "HAIMA", "BOPHA", "KETSANA"]

test_list_4 = ["GONI", "LINFA", "NOCK-TEN", "NOUL", "JANGMI"]

test_list_5 = ["NAKRI", "UTOR", "HAIYAN", "RAMMASUN", "CONSON"]

test_list_6 = ["PHANFONE", "VAMCO", "KOPPU", "FUNG-WONG", "HAGUPIT"]

test_list_7 = ["MEKKHALA", "NARI", "KROSA", "USAGI", "KALMAEGI"]
```

```python
# Extract the column of unique ids
grid_id = df["grid_point_id"]
```

```python
df_test = pd.DataFrame(
    Xnew,
    columns=[
        "typhoon_name",
        "grid_point_id",
        "wind_speed",
        "track_distance",
        "rainfall_max_6h",
        "rainfall_max_24h",
        "total_houses",
        "rwi",
        "strong_roof_strong_wall",
        "strong_roof_light_wall",
        "strong_roof_salvage_wall",
        "light_roof_strong_wall",
        "light_roof_light_wall",
        "light_roof_salvage_wall",
        "salvaged_roof_strong_wall",
        "salvaged_roof_light_wall",
        "salvaged_roof_salvage_wall",
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
    ],
)

df_test = Xnew[Xnew["typhoon_name"] == test_list_3[4]]
df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[3]])
df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[2]])
df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[1]])
df_test = df_test.append(Xnew[Xnew["typhoon_name"] == test_list_3[0]])

Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[4]], inplace=True)
Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[3]], inplace=True)
Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[2]], inplace=True)
Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[1]], inplace=True)
Xnew.drop(Xnew.index[Xnew["typhoon_name"] == test_list_3[0]], inplace=True)

display(df_test)
df_train = Xnew
display(df_train)
```

```python
df_test["typhoon_name"].unique()
```

array(['KETSANA', 'BOPHA', 'HAIMA', 'MANGKHUT', 'SAUDEL'], dtype=object)

```python
# Split X and y from dataframe features
X_test = df_test[features]
X_train = df_train[features]

y_train = df_train["percent_houses_damaged"]
y_test = df_test["percent_houses_damaged"]
```

```python
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

/Users/mersedehkooshki/opt/anaconda3/envs/global-storm/lib/python3.8/site-packages/
xgboost/data.py:250: FutureWarning: pandas.Int64Index is deprecated and will be
removed from pandas in a future version. Use pandas.Index with the appropriate
dtype instead.
  elif isinstance(data.columns, (pd.Int64Index, pd.RangeIndex)):
    [15:36:29] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.229
    Model:                                OLS   Adj. R-squared:           0.228
    Method:                     Least Squares   F-statistic:              479.1
    Date:                    Fri, 09 Jun 2023   Prob (F-statistic):        0.00
    Time:                            15:36:34   Log-Likelihood:     -1.2583e+05
    No. Observations:                   43643   AIC:                  2.517e+05
    Df Residuals:                       43615   BIC:                  2.520e+05
    Df Model:                              27
    Covariance Type:                nonrobust
    ===========================================================================
                            coef   std err       t    P>|t|    [0.025    0.975]
    ---------------------------------------------------------------------------
    const            -4.905e+10  3.97e+11   -0.124    0.902 -8.27e+11  7.29e+11
    wind_speed          17.4519     0.193   90.623    0.000    17.074    17.829
    track_distance       3.4347     0.113   30.286    0.000     3.212     3.657
    rainfall_max_6h      3.6435     0.559    6.514    0.000     2.547     4.740
    rainfall_max_24     -4.2706     0.546   -7.817    0.000    -5.341    -3.200
    total_houses        -0.1335     1.553   -0.086    0.932    -3.178     2.911
    rwi                 -0.0095     0.256   -0.037    0.971    -0.511     0.492
    strong_roof_strong_wall 7.0063  3.442    2.035    0.042     0.259    13.753
    strong_roof_light_wall 8.4713   3.055    2.773    0.006     2.483    14.460
    strong_roof_salvage_wall 5.9594 0.404   14.736    0.000     5.167     6.752
    light_roof_strong_wall  4.1391  1.158    3.574    0.000     1.869     6.409
    light_roof_light_wall    6.9589 2.968    2.345    0.019     1.142    12.776
    light_roof_salvage_wall -1.5073 0.875   -1.722    0.085    -3.223     0.209
    salvaged_roof_strong_wall 0.9716 0.565   1.718    0.086    -0.137     2.080
    salvaged_roof_light_wall -0.4198 0.863  -0.486    0.627    -2.112     1.272
    salvaged_roof_salvage_wall 0.0442 0.391  0.113    0.910    -0.722     0.810
    mean_slope         -0.9250      2.046   -0.452    0.651    -4.935     3.085
    std_slope          -1.5320      1.024   -1.496    0.135    -3.539     0.475
    mean_tri            0.7997      2.122    0.377    0.706    -3.360     4.959
    std_tri             2.5473      1.194    2.134    0.033     0.207     4.887
    mean_elev          -0.3685      0.287   -1.282    0.200    -0.932     0.195
    coast_length        0.6033      0.256    2.359    0.018     0.102     1.105
    with_coast         -0.0761      0.074   -1.025    0.305    -0.222     0.069
    urban            4.905e+10   3.97e+11    0.124    0.902 -7.29e+11  8.27e+11
    rural            4.905e+10   3.97e+11    0.124    0.902 -7.29e+11  8.27e+11
    water            4.905e+10   3.97e+11    0.124    0.902 -7.29e+11  8.27e+11
    total_pop          -0.2064      1.706   -0.121    0.904    -3.549     3.136
    percent_houses_damaged_5years -3.4584 2.176 -1.589 0.112   -7.724     0.807
    ===========================================================================
    Omnibus:                    61282.876   Durbin-Watson:                0.648
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):      20701380.945
    Skew:                           8.299   Prob(JB):                      0.00
    Kurtosis:                     108.397   Cond. No.                  6.66e+13
    ===========================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 2.97e-23. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.

```python
# Make prediction
y_pred_train = xgb.predict(X_train)
y_pred_train_clipped = y_pred_train.clip(0, 100)

y_pred = xgb.predict(X_test)
y_pred_clipped = y_pred.clip(0, 100)
```

```python
y_pred = y_pred_clipped.tolist()
y_true = df_test["percent_houses_damaged"].tolist()
```

```python
df_test.reset_index(drop=True, inplace=True)
for i in range(len(df_test)):
    df_test.at[i, "y_pred"] = y_pred[i]
df_test
```

```python
# Read a CSV file including grid_id and mun_code and import to a df
df_weight = weight_file("/ggl_grid_to_mun_weights.csv")
df_weight.head()
```

```python
# Change name of column ['id'] to ['grid_point_id'] the same name as in input df
df_weight.rename(columns={"id": "grid_point_id"}, inplace=True)
```

```python
# join main df to the weight df based on grid_point_id
join_final = df_test.merge(df_weight, on="grid_point_id", how="left")
```

```python
# Remove all columns between column index 21 to 25
join_final.drop(join_final.iloc[:, 23:27], inplace=True, axis=1)
```

```python
# Multiply %damg and also %predicted_damg with total_houses and weight
join_final["weight*%damg*houses"] = (
    join_final["weight"]
    * join_final["percent_houses_damaged"]
    * join_final["total_houses"]
) / 100
join_final["weight*%predicted_damg*houses"] = (
    join_final["weight"] * join_final["y_pred"] * join_final["total_houses"]
) / 100

# Multiply total_houses with weight
join_final["weight*houses"] = (join_final["weight"] * join_final["total_houses"])
    / 100

join_final
```

```python
# Read CSV file which includes regoin name and code
region_df = pd.read_csv("data/adm3_area.csv", index_col=0)
region_df.head()
```

```python
# join regoin_code column to the main df(join_final) based on mun_code
join_region_df = join_final.merge(
    region_df[["ADM1_EN", "ADM1_PCODE", "ADM3_PCODE"]], on="ADM3_PCODE", how="left"
)
join_region_df
```

```python
# Groupby by municipality with sum as the aggregation function
agg_df = join_region_df.groupby(["ADM3_PCODE", "ADM1_PCODE", "typhoon_name"]).agg(
    {
        "weight*%damg*houses": "sum",
        "weight*%predicted_damg*houses": "sum",
        "weight": "sum",
        "weight*houses": "sum",
    }
)
agg_df
```

```python
# Normalize by the sum of the weights
agg_df["damg_houses_per_mun"] = agg_df["weight*%damg*houses"] / agg_df["weight"]
agg_df["predicted_damg_houses_per_mun"] = (
    agg_df["weight*%predicted_damg*houses"] / agg_df["weight"]
)

agg_df["sum_of_weight_mun"] = agg_df["weight*houses"] / agg_df["weight"]
```

```python
# Keep only %damg_normalized and %pred_damg_normalized columns
agg_df.drop(agg_df.columns[:4], inplace=True, axis=1)
```

```python
# Groupby by regin with sum as the aggregation function
agg_df_1 = agg_df.groupby(["ADM1_PCODE", "typhoon_name"]).agg(
    {
        "damg_houses_per_mun": "sum",
        "predicted_damg_houses_per_mun": "sum",
        "sum_of_weight_mun": "sum",
    }
)
agg_df_1.head()
```

```python
# Rename columns' names
agg_df_1 = agg_df_1.rename(
    columns={
        "damg_houses_per_mun": "damg_houses_per_Region",
        "predicted_damg_houses_per_mun": "predicted_damg_houses_per_Region",
        "sum_of_weight_mun": "sum_of_weight_region",
    }
)
```

```python
# reset indexex
agg_df_2 = agg_df_1.reset_index()
```

```python
# Estimate the percent difference of real and predicted damaged values
# (First way)
agg_df_2["Percent_Difference_total_houses_based"] = (
    (agg_df_2["damg_houses_per_Region"] -
    agg_df_2["predicted_damg_houses_per_Region"])
    / (
        agg_df_2["sum_of_weight_region"]
    )  # (agg_df_2["damg_houses_per_Region"] + np.finfo(float).eps)
) * 100
```

```python
# Estimate the percent difference of real and predicted damaged values
# (Second way)
difference = (
    agg_df_2["damg_houses_per_Region"] -
    agg_df_2["predicted_damg_houses_per_Region"]
)
ave = (
    agg_df_2["damg_houses_per_Region"] +
    agg_df_2["predicted_damg_houses_per_Region"]
) / 2

agg_df_2["Percent_Difference_average_based"] = (difference / ave) * 100
```

```python
agg_df_2 = agg_df_2[
    [
        "ADM1_PCODE",
        "typhoon_name",
        "Percent_Difference_total_houses_based",
        "Percent_Difference_average_based",
    ]
]
```

```python
df_sorted = agg_df_2.sort_values(by=["typhoon_name"], ascending=-True)
    .reset_index(
    drop=True
)
df_sorted
```

```python

```

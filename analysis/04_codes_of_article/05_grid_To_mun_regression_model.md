# Converting from grid-based to municipality-based

## GridGlobal,  GridGlobal+

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
# Specify features
global_features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    "rainfall_max_24h",
    # "rwi",
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
    # "urban",
    # "rural",
    # "water",
    # "total_pop",
    # "percent_houses_damaged_5years",
]

global_plus_features = [
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
# Ask the user whether to use global features set or global+ features set
feature_set = int(
    input("Enter 1 for global features, 2 for global+ features: "))
```

Enter 1 for global features, 2 for global+ features: 2

```python
if feature_set == 1:
    features = global_features
    print(len(features))

elif feature_set == 2:
    features = global_plus_features
    print(len(features))

else:
    print("Invalid input. Please enter 1 or 2")
```

18

```python
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
# Defin two lists to save RMSE and Average Error

RMSE = defaultdict(list)
AVE = defaultdict(list)
```

```python
for i in range(num_exp):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        df_data["percent_houses_damaged"],
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

    eval_set = [(X_train, y_train)]
    xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    print(est2.summary())

    y_pred_all = xgb.predict(X_test)
    y_pred_all_clipped = y_pred_all.clip(0, 100)

    pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])
    pred_df["y_all"] = y_test
    pred_df["y_pred_all"] = y_pred_all_clipped

    # bin_index = np.digitize(pred_df["y_all"], bins=binsP2)

    # Join data with y_all and y_all_pred
    df_data_w_pred = pd.merge(pred_df, df_data, left_index=True,
        right_index=True)
    # Join data with grid_point_id typhoon_year
    df_data_w_pred_grid = pd.merge(
        df[["grid_point_id", "typhoon_year"]],
        df_data_w_pred,
        left_index=True,
        right_index=True,
    )
    df_data_w_pred_grid.sort_values("y_pred_all", ascending=False)

    # join with weights df
    join_df = df_data_w_pred_grid.merge(df_weight, on="grid_point_id",
        how="left")

    # Indicate where values are valid and not missing
    join_df = join_df.loc[join_df["weight"].notna()]

    # Multiply weight by y_all and y_pred_all
    join_df["weight*y_pred*houses"] = (
        join_df["y_pred_all"] * join_df["weight"] * join_df["total_houses"]
            / 100
    )
    join_df["weight*y*houses"] = (
        join_df["y_all"] * join_df["weight"] * join_df["total_houses"]
            / 100
    )
    join_df["weight*houses"] = join_df["weight"] * join_df["total_houses"]

    join_df.sort_values("y_pred_all", ascending=False)

    # Groupby by municipality and typhoon_name with sum as the aggregation function
    agg_df = join_df.groupby(["ADM3_PCODE", "typhoon_name", "typhoon_year"]).agg("sum")

    # Normalize by the sum of the weights
    agg_df["y_pred_norm"] = (
        agg_df["weight*y_pred*houses"] / agg_df["weight*houses"]
            * 100
    )
    agg_df["y_norm"] = agg_df["weight*y*houses"] / agg_df["weight*houses"]
        * 100

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

    # Calculate RMSE & Average Error in total for converted grid_based model
    # to Mun_based
    rmse = sqrt(mean_squared_error(merged_df["y_norm"],
        merged_df["y_pred_norm"]))
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

[20:44:34] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/work
    /src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.

This could be a false alarm, with some parameters getting used by language
bindings but then being mistakenly passed down to XGBoost core, or some
parameter actually being used but getting flagged wrongly here.
Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.202
    Model:                                OLS   Adj. R-squared:           0.202
    Method:                     Least Squares   F-statistic:              593.4
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:37   Log-Likelihood:     -1.1609e+05
    No. Observations:                   39803   AIC:                  2.322e+05
    Df Residuals:                       39785   BIC:                  2.324e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ===========================================================================
                     coef    std err          t      P>|t|      [0.025   0.975]
    ---------------------------------------------------------------------------
    const          0.8259      0.023     36.532      0.000       0.782    0.870
    x1             2.7897      0.034     82.198      0.000       2.723    2.856
    x2             0.9442      0.034     27.721      0.000       0.877    1.011
    x3             0.0222      0.081      0.273      0.785      -0.137    0.182
    x4             0.5558      0.058      9.525      0.000       0.441    0.670
    x5            -0.4841      0.058     -8.308      0.000      -0.598   -0.370
    x6            -0.1283      0.038     -3.378      0.001      -0.203   -0.054
    x7            -1.2233      0.449     -2.722      0.006      -2.104   -0.342
    x8            -0.2762      0.205     -1.347      0.178      -0.678    0.126
    x9             1.1285      0.433      2.609      0.009       0.281    1.976
    x10            0.3681      0.180      2.046      0.041       0.015    0.721
    x11           -0.1145      0.045     -2.521      0.012      -0.203   -0.025
    x12            0.0897      0.032      2.807      0.005       0.027    0.152
    x13           -0.0199      0.045     -0.447      0.655      -0.107    0.068
    x14        -2.511e+12   3.26e+12     -0.771      0.441    -8.9e+12 3.87e+12
    x15        -4.308e+12   5.59e+12     -0.771      0.441   -1.53e+13 6.65e+12
    x16        -4.285e+12   5.56e+12     -0.771      0.441   -1.52e+13 6.61e+12
    x17           -0.0324      0.080     -0.406      0.685      -0.189    0.124
    x18            0.0686      0.022      3.162      0.002       0.026    0.111
    ===========================================================================
    Omnibus:                    58161.134   Durbin-Watson:                2.017
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):      23572452.604
    Skew:                           8.913   Prob(JB):                      0.00
    Kurtosis:                     120.880   Cond. No.                  9.25e+14
    ===========================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 2.75e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16232
    5076
    RMSE for grid_based model: 4.59
    Average Error for grid_based model: -0.28
    [20:44:37] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.207
    Model:                                OLS   Adj. R-squared:           0.206
    Method:                     Least Squares   F-statistic:              610.1
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:39   Log-Likelihood:     -1.1631e+05
    No. Observations:                   39803   AIC:                  2.327e+05
    Df Residuals:                       39785   BIC:                  2.328e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8236      0.023     36.457      0.000       0.779       0.868
    x1             2.8564      0.034     83.728      0.000       2.790       2.923
    x2             0.9699      0.034     28.323      0.000       0.903       1.037
    x3            -0.0053      0.082     -0.064      0.949      -0.167       0.156
    x4             0.5738      0.059      9.740      0.000       0.458       0.689
    x5            -0.5315      0.059     -9.022      0.000      -0.647      -0.416
    x6            -0.0867      0.048     -1.822      0.068      -0.180       0.007
    x7            -1.5946      0.450     -3.543      0.000      -2.477      -0.713
    x8            -0.2191      0.207     -1.056      0.291      -0.626       0.188
    x9             1.5267      0.433      3.526      0.000       0.678       2.375
    x10            0.2982      0.183      1.628      0.103      -0.061       0.657
    x11           -0.1364      0.047     -2.904      0.004      -0.228      -0.044
    x12            0.1039      0.029      3.597      0.000       0.047       0.161
    x13           -0.0323      0.040     -0.801      0.423      -0.111       0.047
    x14         4.435e+12   3.24e+12      1.369      0.171   -1.91e+12    1.08e+13
    x15         7.609e+12   5.56e+12      1.369      0.171   -3.28e+12    1.85e+13
    x16         7.567e+12   5.53e+12      1.369      0.171   -3.27e+12    1.84e+13
    x17           -0.0228      0.080     -0.283      0.777      -0.180       0.135
    x18            0.0811      0.022      3.698      0.000       0.038       0.124
    ==============================================================================
    Omnibus:                    57782.948   Durbin-Watson:                   2.001
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22484987.538
    Skew:                           8.813   Prob(JB):                         0.00
    Kurtosis:                     118.096   Cond. No.                     9.14e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 2.81e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16226
    4957
    RMSE for grid_based model: 4.81
    Average Error for grid_based model: -0.21
    [20:44:39] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/
    work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but then being mistakenly passed down to XGBoost core,
      or some parameter actually being used but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.205
    Model:                                OLS   Adj. R-squared:           0.205
    Method:                     Least Squares   F-statistic:              605.2
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:41   Log-Likelihood:     -1.1640e+05
    No. Observations:                   39803   AIC:                  2.328e+05
    Df Residuals:                       39785   BIC:                  2.330e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.7601      0.029     26.527      0.000       0.704       0.816
    x1             2.8549      0.034     83.779      0.000       2.788       2.922
    x2             0.9696      0.034     28.227      0.000       0.902       1.037
    x3             0.0443      0.086      0.517      0.605      -0.124       0.212
    x4             0.5660      0.059      9.613      0.000       0.451       0.681
    x5            -0.5267      0.059     -8.924      0.000      -0.642      -0.411
    x6            -0.1045      0.038     -2.755      0.006      -0.179      -0.030
    x7            -1.0719      0.450     -2.380      0.017      -1.955      -0.189
    x8            -0.3169      0.208     -1.527      0.127      -0.724       0.090
    x9             1.0875      0.433      2.509      0.012       0.238       1.937
    x10            0.3667      0.183      2.006      0.045       0.008       0.725
    x11           -0.1450      0.046     -3.162      0.002      -0.235      -0.055
    x12            0.1219      0.029      4.198      0.000       0.065       0.179
    x13            0.0303      0.042      0.727      0.467      -0.051       0.112
    x14         1.088e+13   2.85e+12      3.821      0.000     5.3e+12    1.65e+13
    x15         1.867e+13   4.89e+12      3.821      0.000    9.09e+12    2.82e+13
    x16         1.857e+13   4.86e+12      3.821      0.000    9.04e+12    2.81e+13
    x17           -0.0509      0.087     -0.588      0.557      -0.221       0.119
    x18            0.1053      0.026      4.087      0.000       0.055       0.156
    ==============================================================================
    Omnibus:                    57222.185   Durbin-Watson:                   1.987
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21016627.797
    Skew:                           8.666   Prob(JB):                         0.00
    Kurtosis:                     114.229   Cond. No.                     8.02e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 3.65e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16294
    4897
    RMSE for grid_based model: 4.61
    Average Error for grid_based model: -0.34
    [20:44:41] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/
    work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.204
    Model:                                OLS   Adj. R-squared:           0.204
    Method:                     Least Squares   F-statistic:              600.9
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:43   Log-Likelihood:     -1.1662e+05
    No. Observations:                   39803   AIC:                  2.333e+05
    Df Residuals:                       39785   BIC:                  2.334e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8367      0.024     34.283      0.000       0.789       0.885
    x1             2.8354      0.034     82.501      0.000       2.768       2.903
    x2             0.9595      0.034     27.825      0.000       0.892       1.027
    x3             0.0257      0.089      0.289      0.773      -0.149       0.200
    x4             0.6573      0.059     11.058      0.000       0.541       0.774
    x5            -0.5691      0.059     -9.625      0.000      -0.685      -0.453
    x6            -0.1524      0.041     -3.757      0.000      -0.232      -0.073
    x7            -1.0814      0.457     -2.365      0.018      -1.978      -0.185
    x8            -0.2427      0.216     -1.125      0.261      -0.666       0.180
    x9             1.0317      0.436      2.364      0.018       0.176       1.887
    x10            0.2998      0.187      1.604      0.109      -0.067       0.666
    x11           -0.1240      0.052     -2.372      0.018      -0.226      -0.022
    x12            0.1228      0.029      4.229      0.000       0.066       0.180
    x13           -0.0007      0.041     -0.018      0.986      -0.082       0.080
    x14        -1.943e+11   2.64e+12     -0.074      0.941   -5.38e+12    4.99e+12
    x15        -3.334e+11   4.53e+12     -0.074      0.941   -9.22e+12    8.56e+12
    x16        -3.316e+11   4.51e+12     -0.074      0.941   -9.17e+12    8.51e+12
    x17           -0.0341      0.089     -0.381      0.703      -0.209       0.141
    x18            0.0853      0.024      3.511      0.000       0.038       0.133
    ==============================================================================
    Omnibus:                    57785.784   Durbin-Watson:                   2.009
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22613887.812
    Skew:                           8.811   Prob(JB):                         0.00
    Kurtosis:                     118.434   Cond. No.                     7.39e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 4.29e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16426
    4920
    RMSE for grid_based model: 4.34
    Average Error for grid_based model: -0.24
    [20:44:43] WARNING: /Users/runner/miniforge3/conda-bld/xgboost-split_1637426408905/
    work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but then being mistakenly passed down to XGBoost core,
      or some parameter actually being used but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.204
    Model:                                OLS   Adj. R-squared:           0.204
    Method:                     Least Squares   F-statistic:              599.9
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:46   Log-Likelihood:     -1.1617e+05
    No. Observations:                   39803   AIC:                  2.324e+05
    Df Residuals:                       39785   BIC:                  2.325e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8256      0.022     36.758      0.000       0.782       0.870
    x1             2.8086      0.034     82.795      0.000       2.742       2.875
    x2             0.9504      0.034     27.903      0.000       0.884       1.017
    x3            -0.0048      0.080     -0.061      0.952      -0.161       0.152
    x4             0.5743      0.059      9.809      0.000       0.460       0.689
    x5            -0.5065      0.059     -8.656      0.000      -0.621      -0.392
    x6            -0.1040      0.037     -2.782      0.005      -0.177      -0.031
    x7            -1.4111      0.447     -3.154      0.002      -2.288      -0.534
    x8            -0.2553      0.203     -1.257      0.209      -0.654       0.143
    x9             1.3386      0.431      3.107      0.002       0.494       2.183
    x10            0.3378      0.179      1.889      0.059      -0.013       0.688
    x11           -0.1097      0.046     -2.384      0.017      -0.200      -0.020
    x12            0.0846      0.029      2.931      0.003       0.028       0.141
    x13           -0.0301      0.040     -0.753      0.452      -0.108       0.048
    x14           -0.0436      0.032     -1.351      0.177      -0.107       0.020
    x15           -0.0312      0.025     -1.250      0.211      -0.080       0.018
    x16            0.0570      0.020      2.807      0.005       0.017       0.097
    x17           -0.0098      0.079     -0.124      0.901      -0.164       0.144
    x18            0.1105      0.025      4.410      0.000       0.061       0.160
    ==============================================================================
    Omnibus:                    57829.242   Durbin-Watson:                   1.991
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22578633.178
    Skew:                           8.826   Prob(JB):                         0.00
    Kurtosis:                     118.337   Cond. No.                     1.27e+15
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 1.46e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16306
    4953
    RMSE for grid_based model: 4.79
    Average Error for grid_based model: -0.45
    [20:44:46] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.204
    Model:                                OLS   Adj. R-squared:           0.204
    Method:                     Least Squares   F-statistic:              599.8
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:48   Log-Likelihood:     -1.1671e+05
    No. Observations:                   39803   AIC:                  2.334e+05
    Df Residuals:                       39785   BIC:                  2.336e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8299      0.026     31.686      0.000       0.779       0.881
    x1             2.8640      0.034     83.223      0.000       2.797       2.931
    x2             0.9808      0.035     28.366      0.000       0.913       1.049
    x3             0.0533      0.087      0.613      0.540      -0.117       0.224
    x4             0.5750      0.060      9.655      0.000       0.458       0.692
    x5            -0.5203      0.059     -8.756      0.000      -0.637      -0.404
    x6            -0.1227      0.041     -2.964      0.003      -0.204      -0.042
    x7            -0.9365      0.457     -2.048      0.041      -1.833      -0.040
    x8            -0.3742      0.211     -1.773      0.076      -0.788       0.040
    x9             0.9187      0.439      2.092      0.036       0.058       1.779
    x10            0.3839      0.186      2.066      0.039       0.020       0.748
    x11           -0.1351      0.048     -2.810      0.005      -0.229      -0.041
    x12            0.1116      0.029      3.831      0.000       0.055       0.169
    x13           -0.0119      0.042     -0.283      0.777      -0.094       0.070
    x14         -2.92e+11   1.99e+12     -0.147      0.883   -4.19e+12    3.61e+12
    x15        -5.009e+11   3.42e+12     -0.147      0.883    -7.2e+12    6.19e+12
    x16        -4.982e+11    3.4e+12     -0.147      0.883   -7.16e+12    6.16e+12
    x17           -0.0697      0.087     -0.802      0.423      -0.240       0.101
    x18            0.0821      0.022      3.760      0.000       0.039       0.125
    ==============================================================================
    Omnibus:                    58536.702   Durbin-Watson:                   2.013
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         24543754.642
    Skew:                           9.016   Prob(JB):                         0.00
    Kurtosis:                     123.308   Cond. No.                     5.56e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 7.59e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16126
    4962
    RMSE for grid_based model: 4.45
    Average Error for grid_based model: -0.31
    [20:44:48] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ==========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:               0.204
    Model:                                OLS   Adj. R-squared:          0.203
    Method:                     Least Squares   F-statistic:             598.0
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):       0.00
    Time:                            20:44:50   Log-Likelihood:    -1.1621e+05
    No. Observations:                   39803   AIC:                 2.325e+05
    Df Residuals:                       39785   BIC:                 2.326e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8214      0.027     30.445      0.000       0.769       0.874
    x1             2.8218      0.034     82.731      0.000       2.755       2.889
    x2             0.9593      0.034     28.045      0.000       0.892       1.026
    x3             0.0535      0.086      0.624      0.533      -0.114       0.221
    x4             0.5218      0.059      8.880      0.000       0.407       0.637
    x5            -0.4483      0.059     -7.642      0.000      -0.563      -0.333
    x6            -0.1397      0.039     -3.568      0.000      -0.216      -0.063
    x7            -1.1208      0.451     -2.484      0.013      -2.005      -0.236
    x8            -0.3058      0.207     -1.478      0.139      -0.711       0.100
    x9             1.0428      0.434      2.404      0.016       0.193       1.893
    x10            0.4012      0.182      2.204      0.028       0.044       0.758
    x11           -0.1224      0.048     -2.575      0.010      -0.216      -0.029
    x12            0.1003      0.029      3.434      0.001       0.043       0.158
    x13           -0.0289      0.040     -0.722      0.471      -0.107       0.050
    x14           1.1e+12   3.21e+12      0.343      0.732   -5.18e+12    7.38e+12
    x15         1.887e+12    5.5e+12      0.343      0.732   -8.89e+12    1.27e+13
    x16         1.877e+12   5.47e+12      0.343      0.732   -8.84e+12    1.26e+13
    x17           -0.0558      0.086     -0.651      0.515      -0.224       0.112
    x18            0.0882      0.023      3.869      0.000       0.044       0.133
    ==============================================================================
    Omnibus:                    58523.010   Durbin-Watson:                   2.003
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         24416024.850
    Skew:                           9.014   Prob(JB):                         0.00
    Kurtosis:                     122.988   Cond. No.                     9.05e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 2.86e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16510
    4990
    RMSE for grid_based model: 5.12
    Average Error for grid_based model: -0.29
    [20:44:50] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    =========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:               0.204
    Model:                                OLS   Adj. R-squared:          0.203
    Method:                     Least Squares   F-statistic:             599.2
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):       0.00
    Time:                            20:44:52   Log-Likelihood:    -1.1662e+05
    No. Observations:                   39803   AIC:                 2.333e+05
    Df Residuals:                       39785   BIC:                 2.334e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8392      0.025     33.038      0.000       0.789       0.889
    x1             2.8513      0.034     82.971      0.000       2.784       2.919
    x2             0.9606      0.035     27.807      0.000       0.893       1.028
    x3             0.0340      0.085      0.399      0.690      -0.133       0.201
    x4             0.5373      0.059      9.070      0.000       0.421       0.653
    x5            -0.4808      0.059     -8.127      0.000      -0.597      -0.365
    x6            -0.1171      0.038     -3.074      0.002      -0.192      -0.042
    x7            -1.2756      0.456     -2.796      0.005      -2.170      -0.381
    x8            -0.2324      0.211     -1.102      0.271      -0.646       0.181
    x9             1.2434      0.440      2.829      0.005       0.382       2.105
    x10            0.2475      0.186      1.329      0.184      -0.117       0.613
    x11           -0.1289      0.047     -2.727      0.006      -0.221      -0.036
    x12            0.1063      0.030      3.501      0.000       0.047       0.166
    x13           -0.0301      0.048     -0.630      0.529      -0.124       0.064
    x14         1.105e+12   3.05e+12      0.363      0.717   -4.87e+12    7.08e+12
    x15         1.896e+12   5.23e+12      0.363      0.717   -8.35e+12    1.21e+13
    x16         1.886e+12    5.2e+12      0.363      0.717    -8.3e+12    1.21e+13
    x17           -0.0460      0.084     -0.545      0.586      -0.211       0.119
    x18            0.0827      0.023      3.665      0.000       0.038       0.127
    ==============================================================================
    Omnibus:                    58253.799   Durbin-Watson:                   2.008
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23636240.690
    Skew:                           8.942   Prob(JB):                         0.00
    Kurtosis:                     121.034   Cond. No.                     8.51e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 3.23e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16261
    5012
    RMSE for grid_based model: 4.67
    Average Error for grid_based model: -0.32
    [20:44:52] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.206
    Model:                                OLS   Adj. R-squared:           0.206
    Method:                     Least Squares   F-statistic:              608.2
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:55   Log-Likelihood:     -1.1660e+05
    No. Observations:                   39803   AIC:                  2.332e+05
    Df Residuals:                       39785   BIC:                  2.334e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8188      0.029     27.886      0.000       0.761       0.876
    x1             2.8646      0.034     83.636      0.000       2.797       2.932
    x2             0.9857      0.034     28.606      0.000       0.918       1.053
    x3             0.0271      0.089      0.304      0.761      -0.148       0.202
    x4             0.5885      0.059      9.931      0.000       0.472       0.705
    x5            -0.5252      0.059     -8.865      0.000      -0.641      -0.409
    x6            -0.1004      0.038     -2.643      0.008      -0.175      -0.026
    x7            -1.1420      0.455     -2.508      0.012      -2.034      -0.249
    x8            -0.0573      0.210     -0.273      0.785      -0.469       0.355
    x9             1.0942      0.438      2.499      0.012       0.236       1.952
    x10            0.1135      0.185      0.613      0.540      -0.250       0.477
    x11           -0.1433      0.046     -3.091      0.002      -0.234      -0.052
    x12            0.1156      0.030      3.910      0.000       0.058       0.173
    x13           -0.0185      0.040     -0.457      0.647      -0.098       0.061
    x14         1.179e+12   2.33e+12      0.506      0.613   -3.38e+12    5.74e+12
    x15         2.023e+12      4e+12      0.506      0.613   -5.81e+12    9.85e+12
    x16         2.012e+12   3.97e+12      0.506      0.613   -5.78e+12     9.8e+12
    x17           -0.0502      0.091     -0.551      0.581      -0.229       0.128
    x18            0.0815      0.023      3.527      0.000       0.036       0.127
    ==============================================================================
    Omnibus:                    57859.266   Durbin-Watson:                   2.011
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22765543.274
    Skew:                           8.832   Prob(JB):                         0.00
    Kurtosis:                     118.823   Cond. No.                     6.51e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 5.52e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16440
    4908
    RMSE for grid_based model: 4.71
    Average Error for grid_based model: -0.33
    [20:44:55] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.205
    Model:                                OLS   Adj. R-squared:           0.205
    Method:                     Least Squares   F-statistic:              603.8
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:57   Log-Likelihood:     -1.1607e+05
    No. Observations:                   39803   AIC:                  2.322e+05
    Df Residuals:                       39785   BIC:                  2.323e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8246      0.027     30.730      0.000       0.772       0.877
    x1             2.8235      0.034     83.225      0.000       2.757       2.890
    x2             0.9615      0.034     28.162      0.000       0.895       1.028
    x3             0.0233      0.084      0.276      0.782      -0.142       0.188
    x4             0.5520      0.058      9.467      0.000       0.438       0.666
    x5            -0.4856      0.058     -8.342      0.000      -0.600      -0.372
    x6            -0.1119      0.039     -2.901      0.004      -0.188      -0.036
    x7            -1.1425      0.448     -2.549      0.011      -2.021      -0.264
    x8            -0.2515      0.207     -1.217      0.224      -0.657       0.154
    x9             1.0478      0.432      2.425      0.015       0.201       1.895
    x10            0.3376      0.182      1.851      0.064      -0.020       0.695
    x11           -0.0928      0.046     -2.038      0.042      -0.182      -0.004
    x12            0.1255      0.031      4.041      0.000       0.065       0.186
    x13           -0.0527      0.048     -1.107      0.268      -0.146       0.041
    x14        -7.442e+11   2.63e+12     -0.283      0.777   -5.89e+12    4.41e+12
    x15        -1.277e+12   4.51e+12     -0.283      0.777   -1.01e+13    7.56e+12
    x16         -1.27e+12   4.48e+12     -0.283      0.777   -1.01e+13    7.52e+12
    x17           -0.0379      0.084     -0.449      0.653      -0.203       0.128
    x18            0.0906      0.022      4.069      0.000       0.047       0.134
    ==============================================================================
    Omnibus:                    57674.160   Durbin-Watson:                   1.980
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22135901.411
    Skew:                           8.786   Prob(JB):                         0.00
    Kurtosis:                     117.186   Cond. No.                     7.46e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 4.22e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16206
    4937
    RMSE for grid_based model: 5.05
    Average Error for grid_based model: -0.38
    [20:44:57] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.205
    Model:                                OLS   Adj. R-squared:           0.205
    Method:                     Least Squares   F-statistic:              603.5
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:44:59   Log-Likelihood:     -1.1644e+05
    No. Observations:                   39803   AIC:                  2.329e+05
    Df Residuals:                       39785   BIC:                  2.331e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8355      0.025     32.839      0.000       0.786       0.885
    x1             2.8441      0.034     83.110      0.000       2.777       2.911
    x2             0.9588      0.034     27.894      0.000       0.891       1.026
    x3             0.0394      0.085      0.466      0.642      -0.127       0.205
    x4             0.5168      0.059      8.748      0.000       0.401       0.633
    x5            -0.4563      0.059     -7.728      0.000      -0.572      -0.341
    x6            -0.1041      0.042     -2.491      0.013      -0.186      -0.022
    x7            -1.4962      0.453     -3.304      0.001      -2.384      -0.609
    x8            -0.1974      0.210     -0.938      0.348      -0.610       0.215
    x9             1.4747      0.436      3.386      0.001       0.621       2.328
    x10            0.2382      0.185      1.288      0.198      -0.124       0.601
    x11           -0.1613      0.048     -3.335      0.001      -0.256      -0.067
    x12            0.1234      0.032      3.862      0.000       0.061       0.186
    x13           -0.0270      0.046     -0.592      0.554      -0.116       0.062
    x14         6.217e+11   2.65e+12      0.234      0.815   -4.58e+12    5.82e+12
    x15         1.067e+12   4.55e+12      0.234      0.815   -7.86e+12    9.99e+12
    x16         1.061e+12   4.53e+12      0.234      0.815   -7.82e+12    9.94e+12
    x17           -0.0525      0.083     -0.632      0.527      -0.215       0.110
    x18            0.0839      0.022      3.804      0.000       0.041       0.127
    ==============================================================================
    Omnibus:                    57828.292   Durbin-Watson:                   1.994
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22571101.883
    Skew:                           8.826   Prob(JB):                         0.00
    Kurtosis:                     118.317   Cond. No.                     7.46e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 4.22e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16611
    5029
    RMSE for grid_based model: 4.76
    Average Error for grid_based model: -0.26
    [20:44:59] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.204
    Model:                                OLS   Adj. R-squared:           0.204
    Method:                     Least Squares   F-statistic:              601.3
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:45:01   Log-Likelihood:     -1.1672e+05
    No. Observations:                   39803   AIC:                  2.335e+05
    Df Residuals:                       39785   BIC:                  2.336e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8498      0.030     28.355      0.000       0.791       0.909
    x1             2.8533      0.034     82.867      0.000       2.786       2.921
    x2             0.9730      0.035     28.105      0.000       0.905       1.041
    x3             0.0424      0.082      0.518      0.605      -0.118       0.203
    x4             0.5703      0.059      9.591      0.000       0.454       0.687
    x5            -0.4988      0.059     -8.409      0.000      -0.615      -0.383
    x6            -0.0804      0.039     -2.067      0.039      -0.157      -0.004
    x7            -1.0984      0.458     -2.400      0.016      -1.995      -0.202
    x8            -0.2370      0.210     -1.127      0.260      -0.649       0.175
    x9             1.0402      0.439      2.368      0.018       0.179       1.901
    x10            0.2959      0.185      1.602      0.109      -0.066       0.658
    x11           -0.1344      0.049     -2.739      0.006      -0.231      -0.038
    x12            0.1346      0.029      4.611      0.000       0.077       0.192
    x13           -0.0547      0.042     -1.306      0.192      -0.137       0.027
    x14        -1.911e+12   2.19e+12     -0.871      0.384   -6.21e+12    2.39e+12
    x15        -3.279e+12   3.76e+12     -0.871      0.384   -1.07e+13     4.1e+12
    x16        -3.261e+12   3.74e+12     -0.871      0.384   -1.06e+13    4.08e+12
    x17           -0.0574      0.081     -0.708      0.479      -0.216       0.101
    x18            0.1110      0.025      4.455      0.000       0.062       0.160
    ==============================================================================
    Omnibus:                    57426.967   Durbin-Watson:                   2.012
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21333205.639
    Skew:                           8.725   Prob(JB):                         0.00
    Kurtosis:                     115.066   Cond. No.                     6.14e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 6.26e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16207
    5000
    RMSE for grid_based model: 4.87
    Average Error for grid_based model: -0.27
    [20:45:01] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.208
    Model:                                OLS   Adj. R-squared:           0.207
    Method:                     Least Squares   F-statistic:              613.3
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:45:04   Log-Likelihood:     -1.1646e+05
    No. Observations:                   39803   AIC:                  2.330e+05
    Df Residuals:                       39785   BIC:                  2.331e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8365      0.023     36.965      0.000       0.792       0.881
    x1             2.8738      0.034     84.003      0.000       2.807       2.941
    x2             0.9873      0.034     28.708      0.000       0.920       1.055
    x3             0.0198      0.084      0.236      0.813      -0.144       0.184
    x4             0.6113      0.059     10.305      0.000       0.495       0.728
    x5            -0.5505      0.059     -9.309      0.000      -0.666      -0.435
    x6            -0.0836      0.038     -2.213      0.027      -0.158      -0.010
    x7            -1.1628      0.451     -2.577      0.010      -2.047      -0.278
    x8            -0.0497      0.207     -0.240      0.811      -0.456       0.357
    x9             1.0745      0.435      2.472      0.013       0.223       1.926
    x10            0.1252      0.183      0.685      0.493      -0.233       0.483
    x11           -0.1253      0.046     -2.733      0.006      -0.215      -0.035
    x12            0.1186      0.029      4.072      0.000       0.062       0.176
    x13           -0.0254      0.040     -0.631      0.528      -0.104       0.053
    x14           -0.0433      0.032     -1.336      0.182      -0.107       0.020
    x15           -0.0104      0.025     -0.414      0.679      -0.060       0.039
    x16            0.0359      0.020      1.754      0.079      -0.004       0.076
    x17           -0.0342      0.084     -0.410      0.682      -0.198       0.130
    x18            0.0856      0.022      3.866      0.000       0.042       0.129
    ==============================================================================
    Omnibus:                    57425.040   Durbin-Watson:                   1.977
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21654590.884
    Skew:                           8.716   Prob(JB):                         0.00
    Kurtosis:                     115.930   Cond. No.                     1.06e+15
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 2.1e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16279
    4966
    RMSE for grid_based model: 4.87
    Average Error for grid_based model: -0.28
    [20:45:04] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.202
    Model:                                OLS   Adj. R-squared:           0.201
    Method:                     Least Squares   F-statistic:              591.7
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:45:06   Log-Likelihood:     -1.1678e+05
    No. Observations:                   39803   AIC:                  2.336e+05
    Df Residuals:                       39785   BIC:                  2.338e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8344      0.023     36.578      0.000       0.790       0.879
    x1             2.8450      0.035     82.344      0.000       2.777       2.913
    x2             0.9626      0.035     27.800      0.000       0.895       1.030
    x3             0.0174      0.084      0.208      0.835      -0.147       0.181
    x4             0.5556      0.060      9.316      0.000       0.439       0.672
    x5            -0.4931      0.059     -8.306      0.000      -0.609      -0.377
    x6            -0.0968      0.038     -2.538      0.011      -0.172      -0.022
    x7            -1.0447      0.455     -2.294      0.022      -1.937      -0.152
    x8            -0.2181      0.211     -1.034      0.301      -0.632       0.195
    x9             0.9851      0.439      2.245      0.025       0.125       1.845
    x10            0.2722      0.186      1.464      0.143      -0.092       0.637
    x11           -0.1300      0.046     -2.810      0.005      -0.221      -0.039
    x12            0.1206      0.029      4.117      0.000       0.063       0.178
    x13           -0.0503      0.041     -1.236      0.217      -0.130       0.029
    x14           -0.0481      0.033     -1.468      0.142      -0.112       0.016
    x15           -0.0183      0.025     -0.722      0.470      -0.068       0.031
    x16            0.0466      0.021      2.255      0.024       0.006       0.087
    x17           -0.0295      0.082     -0.361      0.718      -0.190       0.131
    x18            0.0782      0.022      3.530      0.000       0.035       0.122
    ==============================================================================
    Omnibus:                    57910.344   Durbin-Watson:                   2.006
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22807705.261
    Skew:                           8.848   Prob(JB):                         0.00
    Kurtosis:                     118.928   Cond. No.                     1.02e+15
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 2.23e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16333
    4919
    RMSE for grid_based model: 4.58
    Average Error for grid_based model: -0.25
    [20:45:06] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ==========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:               0.204
    Model:                                OLS   Adj. R-squared:          0.203
    Method:                     Least Squares   F-statistic:             598.4
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):       0.00
    Time:                            20:45:09   Log-Likelihood:    -1.1683e+05
    No. Observations:                   39803   AIC:                 2.337e+05
    Df Residuals:                       39785   BIC:                 2.339e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8339      0.023     36.510      0.000       0.789       0.879
    x1             2.8544      0.035     82.658      0.000       2.787       2.922
    x2             0.9707      0.035     28.002      0.000       0.903       1.039
    x3             0.0126      0.087      0.144      0.885      -0.158       0.183
    x4             0.5648      0.060      9.487      0.000       0.448       0.681
    x5            -0.4898      0.059     -8.246      0.000      -0.606      -0.373
    x6            -0.0839      0.038     -2.202      0.028      -0.159      -0.009
    x7            -1.2660      0.456     -2.777      0.005      -2.160      -0.372
    x8            -0.2684      0.211     -1.274      0.203      -0.681       0.145
    x9             1.2386      0.439      2.821      0.005       0.378       2.099
    x10            0.3224      0.186      1.736      0.083      -0.042       0.686
    x11           -0.1523      0.047     -3.254      0.001      -0.244      -0.061
    x12            0.1067      0.029      3.671      0.000       0.050       0.164
    x13           -0.0232      0.041     -0.571      0.568      -0.103       0.057
    x14           -0.0601      0.033     -1.827      0.068      -0.125       0.004
    x15           -0.0103      0.025     -0.406      0.685      -0.060       0.040
    x16            0.0456      0.021      2.202      0.028       0.005       0.086
    x17           -0.0230      0.087     -0.265      0.791      -0.193       0.147
    x18            0.0825      0.023      3.663      0.000       0.038       0.127
    ==============================================================================
    Omnibus:                    57840.607   Durbin-Watson:                   2.006
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22420003.994
    Skew:                           8.834   Prob(JB):                         0.00
    Kurtosis:                     117.919   Cond. No.                     2.75e+15
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 3.07e-26. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16319
    4989
    RMSE for grid_based model: 4.64
    Average Error for grid_based model: -0.35
    [20:45:09] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.203
    Model:                                OLS   Adj. R-squared:           0.202
    Method:                     Least Squares   F-statistic:              595.5
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:45:11   Log-Likelihood:     -1.1663e+05
    No. Observations:                   39803   AIC:                  2.333e+05
    Df Residuals:                       39785   BIC:                  2.335e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8413      0.023     37.024      0.000       0.797       0.886
    x1             2.8335      0.034     82.502      0.000       2.766       2.901
    x2             0.9509      0.035     27.543      0.000       0.883       1.019
    x3             0.0327      0.084      0.389      0.697      -0.132       0.197
    x4             0.5178      0.059      8.721      0.000       0.401       0.634
    x5            -0.4630      0.059     -7.819      0.000      -0.579      -0.347
    x6            -0.0895      0.038     -2.347      0.019      -0.164      -0.015
    x7            -1.3832      0.454     -3.047      0.002      -2.273      -0.493
    x8            -0.0979      0.209     -0.469      0.639      -0.507       0.311
    x9             1.2815      0.438      2.927      0.003       0.423       2.140
    x10            0.1838      0.184      0.999      0.318      -0.177       0.544
    x11           -0.1169      0.046     -2.542      0.011      -0.207      -0.027
    x12            0.1236      0.029      4.262      0.000       0.067       0.180
    x13           -0.0404      0.040     -1.000      0.317      -0.119       0.039
    x14           -0.0559      0.033     -1.705      0.088      -0.120       0.008
    x15           -0.0156      0.025     -0.613      0.540      -0.065       0.034
    x16            0.0484      0.021      2.354      0.019       0.008       0.089
    x17           -0.0443      0.083     -0.531      0.595      -0.208       0.119
    x18            0.1042      0.025      4.160      0.000       0.055       0.153
    ==============================================================================
    Omnibus:                    57851.739   Durbin-Watson:                   2.016
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         22570448.546
    Skew:                           8.834   Prob(JB):                         0.00
    Kurtosis:                     118.313   Cond. No.                     1.67e+15
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 8.43e-26. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16119
    4929
    RMSE for grid_based model: 4.45
    Average Error for grid_based model: -0.27
    [20:45:11] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.202
    Model:                                OLS   Adj. R-squared:           0.202
    Method:                     Least Squares   F-statistic:              592.6
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:45:13   Log-Likelihood:     -1.1668e+05
    No. Observations:                   39803   AIC:                  2.334e+05
    Df Residuals:                       39785   BIC:                  2.335e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8445      0.025     34.458      0.000       0.796       0.893
    x1             2.8367      0.034     82.444      0.000       2.769       2.904
    x2             0.9696      0.035     28.005      0.000       0.902       1.037
    x3             0.0256      0.087      0.294      0.768      -0.145       0.196
    x4             0.5098      0.059      8.616      0.000       0.394       0.626
    x5            -0.4387      0.059     -7.424      0.000      -0.555      -0.323
    x6            -0.0603      0.048     -1.258      0.208      -0.154       0.034
    x7            -1.3131      0.455     -2.887      0.004      -2.205      -0.422
    x8            -0.1881      0.206     -0.911      0.362      -0.593       0.216
    x9             1.2338      0.438      2.815      0.005       0.375       2.093
    x10            0.2730      0.182      1.501      0.133      -0.083       0.629
    x11           -0.1174      0.047     -2.509      0.012      -0.209      -0.026
    x12            0.1296      0.029      4.418      0.000       0.072       0.187
    x13           -0.0310      0.041     -0.763      0.446      -0.111       0.049
    x14         2.583e+12   3.43e+12      0.753      0.451   -4.14e+12     9.3e+12
    x15         4.431e+12   5.88e+12      0.753      0.451    -7.1e+12     1.6e+13
    x16         4.407e+12   5.85e+12      0.753      0.451   -7.06e+12    1.59e+13
    x17           -0.0531      0.087     -0.607      0.544      -0.224       0.118
    x18            0.0885      0.024      3.617      0.000       0.041       0.136
    ==============================================================================
    Omnibus:                    58347.477   Durbin-Watson:                   1.996
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         23856380.502
    Skew:                           8.968   Prob(JB):                         0.00
    Kurtosis:                     121.587   Cond. No.                     9.60e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 2.56e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16414
    4912
    RMSE for grid_based model: 4.71
    Average Error for grid_based model: -0.44
    [20:45:13] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.205
    Model:                                OLS   Adj. R-squared:           0.204
    Method:                     Least Squares   F-statistic:              602.0
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:45:16   Log-Likelihood:     -1.1639e+05
    No. Observations:                   39803   AIC:                  2.328e+05
    Df Residuals:                       39785   BIC:                  2.330e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8313      0.023     36.804      0.000       0.787       0.876
    x1             2.8355      0.034     83.117      0.000       2.769       2.902
    x2             0.9615      0.034     28.038      0.000       0.894       1.029
    x3             0.0317      0.084      0.376      0.707      -0.134       0.197
    x4             0.5467      0.059      9.274      0.000       0.431       0.662
    x5            -0.4849      0.059     -8.212      0.000      -0.601      -0.369
    x6            -0.1038      0.038     -2.748      0.006      -0.178      -0.030
    x7            -1.3272      0.452     -2.934      0.003      -2.214      -0.441
    x8            -0.2048      0.208     -0.985      0.324      -0.612       0.203
    x9             1.2633      0.436      2.898      0.004       0.409       2.118
    x10            0.2535      0.183      1.382      0.167      -0.106       0.613
    x11           -0.1034      0.046     -2.257      0.024      -0.193      -0.014
    x12            0.0968      0.029      3.388      0.001       0.041       0.153
    x13           -0.0117      0.040     -0.292      0.770      -0.090       0.067
    x14           -0.0288      0.033     -0.884      0.377      -0.093       0.035
    x15           -0.0322      0.025     -1.279      0.201      -0.081       0.017
    x16            0.0492      0.020      2.409      0.016       0.009       0.089
    x17           -0.0483      0.084     -0.575      0.565      -0.213       0.116
    x18            0.0952      0.023      4.169      0.000       0.050       0.140
    ==============================================================================
    Omnibus:                    57623.210   Durbin-Watson:                   2.000
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21849372.762
    Skew:                           8.776   Prob(JB):                         0.00
    Kurtosis:                     116.430   Cond. No.                     1.47e+15
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 1.09e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16364
    5015
    RMSE for grid_based model: 4.49
    Average Error for grid_based model: -0.28
    [20:45:16] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.205
    Model:                                OLS   Adj. R-squared:           0.205
    Method:                     Least Squares   F-statistic:              604.8
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:45:18   Log-Likelihood:     -1.1608e+05
    No. Observations:                   39803   AIC:                  2.322e+05
    Df Residuals:                       39785   BIC:                  2.323e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8210      0.024     34.504      0.000       0.774       0.868
    x1             2.8303      0.034     83.374      0.000       2.764       2.897
    x2             0.9581      0.034     28.176      0.000       0.891       1.025
    x3             0.0268      0.082      0.329      0.742      -0.133       0.187
    x4             0.5274      0.058      9.027      0.000       0.413       0.642
    x5            -0.4808      0.058     -8.229      0.000      -0.595      -0.366
    x6            -0.0998      0.038     -2.632      0.008      -0.174      -0.025
    x7            -1.1760      0.450     -2.613      0.009      -2.058      -0.294
    x8            -0.0875      0.211     -0.415      0.678      -0.501       0.326
    x9             1.0398      0.432      2.408      0.016       0.193       1.886
    x10            0.1799      0.185      0.972      0.331      -0.183       0.543
    x11           -0.0890      0.050     -1.777      0.076      -0.187       0.009
    x12            0.1008      0.032      3.158      0.002       0.038       0.163
    x13           -0.0237      0.048     -0.493      0.622      -0.118       0.070
    x14        -1.944e+11   2.99e+12     -0.065      0.948   -6.06e+12    5.67e+12
    x15        -3.335e+11   5.13e+12     -0.065      0.948   -1.04e+13    9.73e+12
    x16        -3.317e+11   5.11e+12     -0.065      0.948   -1.03e+13    9.67e+12
    x17           -0.0350      0.080     -0.437      0.662      -0.192       0.122
    x18            0.1001      0.025      4.060      0.000       0.052       0.148
    ==============================================================================
    Omnibus:                    57633.603   Durbin-Watson:                   2.010
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21861796.720
    Skew:                           8.779   Prob(JB):                         0.00
    Kurtosis:                     116.462   Cond. No.                     8.47e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 3.26e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16364
    5015
    RMSE for grid_based model: 4.65
    Average Error for grid_based model: -0.29
    [20:45:18] WARNING: /Users/runner/miniforge3/conda-bld/
    xgboost-split_1637426408905/work/src/learner.cc:576:
    Parameters: { "early_stopping_rounds" } might not be used.
      This could be a false alarm, with some parameters getting used by
      language bindings but
      then being mistakenly passed down to XGBoost core,
      or some parameter actually being used
      but getting flagged wrongly here.
      Please open an issue if you find any such cases.
                                  OLS Regression Results
    ===========================================================================
    Dep. Variable:     percent_houses_damaged   R-squared:                0.206
    Model:                                OLS   Adj. R-squared:           0.205
    Method:                     Least Squares   F-statistic:              606.0
    Date:                    Wed, 28 Jun 2023   Prob (F-statistic):        0.00
    Time:                            20:45:20   Log-Likelihood:     -1.1598e+05
    No. Observations:                   39803   AIC:                  2.320e+05
    Df Residuals:                       39785   BIC:                  2.321e+05
    Df Model:                              17
    Covariance Type:                nonrobust
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const          0.8241      0.028     29.223      0.000       0.769       0.879
    x1             2.8197      0.034     83.394      0.000       2.753       2.886
    x2             0.9574      0.034     28.236      0.000       0.891       1.024
    x3             0.0229      0.081      0.284      0.776      -0.135       0.181
    x4             0.5728      0.058      9.827      0.000       0.459       0.687
    x5            -0.5011      0.058     -8.581      0.000      -0.616      -0.387
    x6            -0.1285      0.040     -3.243      0.001      -0.206      -0.051
    x7            -1.4454      0.448     -3.229      0.001      -2.323      -0.568
    x8            -0.0744      0.204     -0.364      0.716      -0.475       0.326
    x9             1.3695      0.430      3.183      0.001       0.526       2.213
    x10            0.1469      0.180      0.817      0.414      -0.205       0.499
    x11           -0.1152      0.048     -2.407      0.016      -0.209      -0.021
    x12            0.0972      0.030      3.272      0.001       0.039       0.155
    x13            0.0124      0.043      0.289      0.772      -0.072       0.097
    x14         4.841e+11   2.01e+12      0.240      0.810   -3.47e+12    4.43e+12
    x15         8.305e+11   3.46e+12      0.240      0.810   -5.94e+12    7.61e+12
    x16          8.26e+11   3.44e+12      0.240      0.810   -5.91e+12    7.56e+12
    x17           -0.0276      0.078     -0.351      0.725      -0.181       0.126
    x18            0.0804      0.022      3.675      0.000       0.038       0.123
    ==============================================================================
    Omnibus:                    57344.080   Durbin-Watson:                   2.007
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):         21134143.743
    Skew:                           8.703   Prob(JB):                         0.00
    Kurtosis:                     114.536   Cond. No.                     5.72e+14
    ==============================================================================
    Notes:
    [1] Standard Errors assume that the covariance matrix of the errors is
    correctly specified.
    [2] The smallest eigenvalue is 7.15e-25. This might indicate that there are
    strong multicollinearity problems or that the design matrix is singular.
    16174
    4947
    RMSE for grid_based model: 4.96
    Average Error for grid_based model: -0.28

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

mean_RMSE: 4.71
stdev_RMSE: 0.20
mean_average_error: -0.30
stdev_average_error: 0.06

```python
for bin_num in range(1, 6):

    print(f"\n RMSE and Average Error per bin {bin_num}\n")
    rmse_ave_mean(RMSE[bin_num], AVE[bin_num])
```

RMSE and Average Error per bin 1

mean_RMSE: 0.24
stdev_RMSE: 0.08
mean_average_error: 0.03
stdev_average_error: 0.01

RMSE and Average Error per bin 2

mean_RMSE: 1.50
stdev_RMSE: 0.13
mean_average_error: 0.58
stdev_average_error: 0.03

RMSE and Average Error per bin 3

mean_RMSE: 4.63
stdev_RMSE: 0.54
mean_average_error: 0.10
stdev_average_error: 0.25

RMSE and Average Error per bin 4

mean_RMSE: 14.02
stdev_RMSE: 0.66
mean_average_error: -6.66
stdev_average_error: 0.92

RMSE and Average Error per bin 5

mean_RMSE: 31.62
stdev_RMSE: 3.50
mean_average_error: -23.49
stdev_average_error: 3.52

```python

```

# Feature Importance

## NOTE: There is an error raised within this notebook

## when splitting training and test sets

ValueError: The least populated class in y has only 1 member,
which is too few.

SHAP values and XGBoost built-in feature importance are two
popular techniques for determining feature importance.

SHAP values are computed by analyzing the impact of each feature
on the model's output when that feature is included or excluded.

XGBoost is a gradient-boosting library that includes a built-in
feature importance function that ranks features based on how often
they are used to split the data in the boosting process.
The XGBoost feature importance function takes into account the
contribution of each feature to the model's accuracy.

Both SHAP values and XGBoost built-in feature importance provide
valuable insights into the importance of different features in a dataset.
These techniques may produce different rankings of feature importance so
it is useful to compare their results to get a more comprehensive understanding
of the importance of different features.

```python
%load_ext jupyter_black
```

```python
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from xgboost.sklearn import XGBRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
```

```python
# Read csv file and import to df
df = pd.read_csv("data/updated_corr.csv")
df.head()
```

```python
# Show histogram of damage
df.hist(column="percent_houses_damaged", figsize=(4, 3))
```

array([[<AxesSubplot:title={'center':'percent_houses_damaged'}>]],
      dtype=object)

![png](output_4_1.png)

```python
# Hist plot after data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

[<matplotlib.lines.Line2D at 0x7f9a48689310>]

![png](output_5_1.png)

```python
# Check the bins' intervalls (first bin means all zeros,
# second bin means 0 < values <= 1)
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
```

```python
# Hist plot after removing rows where windspeed is 0
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(
    df["percent_houses_damaged"], bins=bins2
)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

[<matplotlib.lines.Line2D at 0x7f99f8041910>]

![png](output_8_1.png)

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

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```

```python
# Split dataset into training set and test set
## NOTE: This causes an error.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    df["percent_houses_damaged"],
    stratify=y_input_strat,
    test_size=0.2,
)
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
```

```python
X_train4shapely = pd.DataFrame(data=X_train, columns=features)
```

```python
explainer_xgb = shap.Explainer(xgb_model, X_train4shapely)
shap_values_xgb = explainer_xgb(X_train4shapely)
```

97%|=================== | 38536/39803 [00:30<00:00]

```python
# Showing Barplot
plt.title("The Bar plot wrt Shap Values")
shap.plots.bar(shap_values_xgb, max_display=25, show=False)
plt.gcf().set_size_inches(9, 5)
plt.show()
```

![png](output_18_0.png)

```python
# Showing Beeswarm Plot
# plt.gcf().set_size_inches(4, 3)
shap.plots.beeswarm(
    shap_values_xgb,
    max_display=25,
    plot_size=0.7,
    # order=shap_values_xgb.abs.max(0)#, color="shap_red"
)
```

![png](output_19_0.png)

```python
# Xgboost Built-in Feature Importance

plt.rcParams.update({"figure.figsize": (8.0, 7.0)})
plt.rcParams.update({"font.size": 10})

sorted_idx = xgb.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], xgb.feature_importances_[sorted_idx])
plt.title("Xgboost built-in Feature Importance")
plt.xlabel("Feature Importance values")
```

Text(0.5, 0, 'Feature Importance values')

![png](output_20_1.png)

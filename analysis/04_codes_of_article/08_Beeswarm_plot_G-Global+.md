# Feature Importance

Beeswarm Plot based on SHAP value for G-Global+ model.

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
df = pd.read_csv("/Users/mersedehkooshki/03_new_model_training/data/updated_corr.csv")
df
```

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df = df.drop(columns=["grid_point_id", "typhoon_year"])
```

```python
# Hist plot after removing rows where windspeed is 0
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

[<matplotlib.lines.Line2D at 0x7fbf5910f430>]

![png](output_5_1.png)

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

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
        'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
        'water', 'total_pop', 'percent_houses_damaged_5years'],
        dtype='object')

```python
# Split dataset into training set and test set

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["percent_houses_damaged"], stratify=y_input_strat, test_size=0.2
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

eval_set = [(X_train, y_train)]
xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)
```

```python
X_train4shapely = pd.DataFrame(data=X_train, columns=features)
```

```python
explainer_xgb = shap.Explainer(xgb_model, X_train4shapely)
shap_values_xgb = explainer_xgb(X_train4shapely)
```

98%|===================| 38831/39803 [00:33<00:00]

```python
shap.summary_plot(shap_values_xgb, max_display=30, plot_size=0.7, show=False)
ax = plt.gca()

ax.set_xlabel("")
# You can change the min and max value of xaxis by changing the arguments of:
ax.set_xlim(-5, 10)
# plt.show()

plt.savefig("SHAP_updated1.pdf", bbox_inches="tight")
```

![png](output_15_0.png)

```python
# Calculate shap_values
shap.summary_plot(shap_values_xgb, max_display=30, plot_size=0.7, show=False)
ax = plt.gca()

ax.set_xlabel("")
# You can change the min and max value of xaxis by changing the arguments of:
ax.set_xlim(-10, 70)
# plt.show()


plt.savefig("SHAP_updated2.pdf", bbox_inches="tight")
```

![png](output_16_0.png)

```python

```

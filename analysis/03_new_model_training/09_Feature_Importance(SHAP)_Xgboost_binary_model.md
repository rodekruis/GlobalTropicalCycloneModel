# XGBoost Classification Model - Feature Importance by SHAP values

```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import shap

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
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
# define a threshold to separate target into damaged and not_damaged
thres = 10.0

for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] >= thres:
        df.at[i, "binary_damage"] = 1
    else:
        df.at[i, "binary_damage"] = 0

df["binary_damage"] = df["binary_damage"].astype("int")

# Remove previous target 'percent_buildings_damaged' from the dataframe
df = df.drop(["percent_houses_damaged"], axis=1)
df
```

```python
plt.figure(figsize=(4, 3))
sns.countplot(x="binary_damage", data=df, palette="hls")
plt.title("bar_plot (counts of observations)")
plt.show()
```

![png](output_6_0.png)

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df = df.drop(columns=["grid_point_id", "typhoon_year"])
df.head()
```

```python
# Show histogram of damage
df.hist(column="binary_damage", figsize=(4, 3))
```

array([[<AxesSubplot:title={'center':'binary_damage'}>]], dtype=object)

![png](output_8_1.png)

```python
# Define bins and data stratification
bins2 = [0, 0.1, 1]
samples_per_bin2, binsP2 = np.histogram(df["binary_damage"], bins=bins2)
```

```python
# Check the bins' intervalls (first bin means all zeros,
# second bin means 0 < values <= 1)
df["binary_damage"].value_counts(bins=binsP2)
```

(-0.001, 0.1]    48685
(0.1, 1.0]        1069
Name: binary_damage, dtype: int64

```python
print(samples_per_bin2)
print(binsP2)
```

[48685  1069]
[0.  0.1 1. ]

```python
bin_index2 = np.digitize(df["binary_damage"], bins=binsP2)
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
y = df["binary_damage"]
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
        'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
        'water', 'total_pop', 'percent_houses_damaged_5years'],
      dtype='object')

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, df["binary_damage"], stratify=y_input_strat, test_size=0.2
)
```

```python
# Use XGBClassifier as a Machine Learning model to fit the data
xgb_model = XGBClassifier()

eval_set = [(X_train, y_train), (X_test, y_test)]
# eval_set = [(X_test, y_test)]
xgb_model.fit(
    X_train,
    y_train,
    eval_metric=["error", "logloss"],
    eval_set=eval_set,
    verbose=False,
)
```

```python
# Make prediction on test data
y_pred = xgb_model.predict(X_test)
```

```python
# Xgboost Built-in Feature Importance
plt.rcParams.update({"figure.figsize": (8.0, 5.0)})
plt.rcParams.update({"font.size": 10})

sorted_idx = xgb_model.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], xgb_model.feature_importances_[sorted_idx])
plt.xlabel("Built_in Feature Importance")
plt.title("Xgboost")
plt.show()
```

![png](output_18_0.png)

```python
# Xgboost Feature Importance based on SHAP values
X_train4shapely = pd.DataFrame(data=X_train, columns=features)
```

```python
explainer_xgb = shap.Explainer(xgb_model, X_train4shapely)
shap_values_xgb = explainer_xgb(X_train4shapely)
```

99%|===================| 39553/39803 [01:18<00:00]

```python
# Showing Barplot
plt.title("The Bar plot wrt Shap Values")
shap.plots.bar(shap_values_xgb, max_display=30, show=False)
plt.gcf().set_size_inches(8, 5)
plt.show()
```

![png](output_21_0.png)

```python
# Showing Beeswarm Plot
# plt.gcf().set_size_inches(4, 3)
shap.plots.beeswarm(
    shap_values_xgb,
    max_display=30,
    plot_size=0.7,
    # order=shap_values_xgb.abs.max(0)#, color="shap_red"
)
```

![png](output_22_0.png)

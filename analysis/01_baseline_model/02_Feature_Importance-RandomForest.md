The estimation of Feature Importance based on a Random Forest regression model.
After removing the features with correlation value higher than 0.99, analyzing on the input data of the model
was done to estimate the most important features  with respect to a Random Forest.

The Feature Importance is done according to two different approaches: 1.SHAP values which used to explain how each feature affects the model and 2.Random Forest Built-in Feature Importance which is based on reducing in impurity for Random Forest.

```python
%load_ext jupyter_black
```

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import shap

from utils import get_clean_dataset
```

```python
df = get_clean_dataset()
```

```python
# df = df[df.DAM_perc_dmg != 0] a
# df
```

```python
# The Old and New set of bins
# bins2= [0, 1, 60, 100]
bins2 = [0, 0.00009, 1, 10, 50, 100]
samples_per_bin2, binsP2 = np.histogram(df["DAM_perc_dmg"], bins=bins2)
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

```python
print(samples_per_bin2)
print(binsP2)
```

```python
bin_index2 = np.digitize(df["DAM_perc_dmg"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
# Dropping highly correlated features (correlation value > 0.99) from X data.
features = [
    "HAZ_rainfall_Total",
    "HAZ_rainfall_max_6h",
    "HAZ_rainfall_max_24h",
    "HAZ_v_max",
    "HAZ_v_max_3",
    "HAZ_dis_track_min",
    "GEN_landslide_per",
    "GEN_stormsurge_per",
    #'GEN_Bu_p_inSSA',
    #'GEN_Bu_p_LS',
    "GEN_Red_per_LSbldg",
    "GEN_Or_per_LSblg",
    "GEN_Yel_per_LSSAb",
    #'GEN_RED_per_SSAbldg',
    "GEN_OR_per_SSAbldg",
    "GEN_Yellow_per_LSbl",
    "TOP_mean_slope",
    "TOP_mean_elevation_m",
    "TOP_ruggedness_stdev",
    #'TOP_mean_ruggedness',
    #'TOP_slope_stdev',
    "VUL_poverty_perc",
    "GEN_with_coast",
    "GEN_coast_length",
    "VUL_Housing_Units",
    "VUL_StrongRoof_StrongWall",
    "VUL_StrongRoof_LightWall",
    "VUL_StrongRoof_SalvageWall",
    "VUL_LightRoof_StrongWall",
    "VUL_LightRoof_LightWall",
    "VUL_LightRoof_SalvageWall",
    "VUL_SalvagedRoof_StrongWall",
    "VUL_SalvagedRoof_LightWall",
    "VUL_SalvagedRoof_SalvageWall",
    "VUL_vulnerable_groups",
    "VUL_pantawid_pamilya_beneficiary",
]

X = df[features]
# display(X.columns)
y = df["DAM_perc_dmg"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["DAM_perc_dmg"], stratify=y_input_strat, test_size=0.2
)


# print(X_train.shape, y_train.shape)
# print(X_test.shape, y_test.shape)
```

```python
# create an RandomForest regression model

rf = RandomForestRegressor(
    max_depth=None, n_estimators=100, min_samples_split=8, min_samples_leaf=5
)

rf_model = rf.fit(X_train, y_train)
```

```python
X_train4shapely = pd.DataFrame(data=X_train, columns=features)
```

```python
# compute SHAP values
explainer_rf = shap.Explainer(rf_model, X_train4shapely)
shap_values_rf = explainer_rf(X_train4shapely, check_additivity=False)
```

```python
# Showing Barplot
# shap.plots.heatmap(shap_values_rf[:1000])

shap.plots.bar(shap_values_rf, max_display=20)
```

```python
# Showing Beeswarm Plot
# shap.plots.beeswarm(shap_values_rf, max_display=20, order=shap_values_rf.abs.max(0)#, color="shap_red")

shap.plots.beeswarm(
    shap_values_rf,
    max_display=20,  # , color="shap_red",
)
```

```python
# Showing Heatmap
shap.plots.heatmap(shap_values_rf[:1000])
```

```python
# Random Forest Built-in Feature Importance

plt.rcParams.update({"figure.figsize": (12.0, 8.0)})
plt.rcParams.update({"font.size": 10})

rf.feature_importances_

sorted_idx = rf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], rf.feature_importances_[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
```

```python
rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

importances = rf.feature_importances_
sorted_idx = rf.feature_importances_.argsort()
forest_importances = pd.Series(importances, index=X.columns)


fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
# forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
```

```python
"""
Showing X data illustrates that 5 highly correlated features with the value higher than 0.99 were removed 
before applying feature importance methods on input data.
"""
X
```

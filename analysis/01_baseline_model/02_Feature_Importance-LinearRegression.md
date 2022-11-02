The estimation of Feature Importance based on a Linear Regression model.

Analyzing the most important features in the input dataframe with respect to a simple linear regression model. 
For this reason, first all the highly correlated features were dropped according to result of VIF and 
then regression model was applied to estimate the importance of each remaining features.


```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

from utils import get_clean_dataset
```

```python
df = get_clean_dataset()
```

```python
# Show histogram of damage
df.hist(column="DAM_perc_dmg")
```

```python
# A bin's set defined to categorize target values in different groups(bins)
# The chosen bins
bins2 = [0, 1, 60, 100]
samples_per_bin2, binsP2 = np.histogram(df["DAM_perc_dmg"], bins=bins2)
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

```python
# Display bins
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
# TODO: what is this list?
features = [
    "HAZ_rainfall_Total",
    "HAZ_v_max",
    "GEN_landslide_per",
    "GEN_stormsurge_per",
    "GEN_Red_per_LSbldg",
    "GEN_Or_per_LSblg",
    "GEN_OR_per_SSAbldg",
    "GEN_Yellow_per_LSbl",
    "TOP_mean_slope",
    "GEN_with_coast",
    "GEN_coast_length",
    "VUL_Housing_Units",
    "VUL_StrongRoof_StrongWall",
    "VUL_StrongRoof_SalvageWall",
    "VUL_LightRoof_StrongWall",
    "VUL_LightRoof_LightWall",
    "VUL_LightRoof_SalvageWall",
    "VUL_SalvagedRoof_StrongWall",
    "VUL_SalvagedRoof_LightWall",
    "VUL_SalvagedRoof_SalvageWall",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["DAM_perc_dmg"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)


# Split dataset into training set and test set and applying data stratification according to the defined bins
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["DAM_perc_dmg"], stratify=y_input_strat, test_size=0.2
)
```

```python
regressor = LinearRegression()
model_regr = regressor.fit(X_train, y_train)
```

```python
feature = []
values = []
importance = regressor.coef_
# print(importance)
for i, j in enumerate(importance):
    print("Feature %0d " % (i) + X.columns[i] + ":, Score: %.5f" % (j))

    feature.append(X.columns[i])
    values.append(j)
```

```python
# Creating a dataframe of features and their coefficient values

df_coef = pd.DataFrame(columns=["feature", "coef_value"])
df_coef["feature"] = feature
df_coef["coef_value"] = values

df_coef
```

```python
# Sorting the dataframe of coefficient values in a descending order

final_sorted_df = df_coef.sort_values(by=["coef_value"], ascending=False)
final_sorted_df = final_sorted_df.reset_index(drop=True)
final_sorted_df
```

```python
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Coefficients_Values", fontsize=20)
ax = fig.add_axes([0, 0, 1, 1])
# ax.bar(features,values)
ax.bar(final_sorted_df["feature"], final_sorted_df["coef_value"])
np.rot90(plt.xticks(rotation=90, fontsize=27))
plt.show()
```

```python
X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2)
est2 = est.fit()
print(est2.summary())
```

```python
fea = X.columns
fea.tolist()
fea_1 = pd.DataFrame(columns=["names"])
fea_1["names"] = fea.tolist()
fea_1
```

<!-- #region -->


During the feature selection part of the model, we figured out that very different sets of features were chosen 
in different runs. Hence, a decision was made to search for highly correlated features among all the features 
in the dataset.

The following code looks for highly correlated features in the model's input data.

<!-- #endregion -->

```python
%load_ext jupyter_black
```

```python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

```python
# Whether or not to save the figures
save_fig = True
```

```python
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "input/01_baseline_model"
output_dir = Path(os.getenv("STORM_DATA_DIR")) / "output/01_baseline_model"
```

## Data cleaning

TODO: If this is needed in every notebook, should 
refactor it to a first notebook and save the output,
or make a helper function that does it

```python
# Read in input data
combined_input_data = pd.read_csv(input_dir / "combined_input_data.csv")
combined_input_data
```

```python
# TODO: This seems to have no effect. Remove?
typhoons_with_impact_data = [
    "bopha2012",
    "conson2010",
    "durian2006",
    "fengshen2008",
    "fung-wong2014",
    "goni2015",
    "goni2020",
    "hagupit2014",
    "haima2016",
    "haiyan2013",
    "jangmi2014",
    "kalmaegi2014",
    "kammuri2019",
    "ketsana2009",
    "koppu2015",
    "krosa2013",
    "linfa2015",
    "lingling2014",
    "mangkhut2018",
    "mekkhala2015",
    "melor2015",
    "meranti2016",
    "molave2020",
    "mujigae2015",
    "nakri2019",
    "nari2013",
    "nesat2011",
    "nock-ten2016",
    "noul2015",
    "phanfone2019",
    "rammasun2014",
    "sarika2016",
    "saudel2020",
    "tokage2016",
    "trami2013",
    "usagi2013",
    "utor2013",
    "vamco2020",
    "vongfong2020",
    "yutu2018",
]
combined_input_data = combined_input_data[
    combined_input_data.typhoon.isin(typhoons_with_impact_data)
]
combined_input_data
```

```python
# Set some values to 0
# TODO: Check this equation
def set_zeros(x):
    x_max = 25
    y_max = 50

    v_max = x[0]
    rainfall_max = x[1]
    damage = x[2]
    if pd.notnull(damage):
        value = damage
    elif v_max > x_max or rainfall_max > y_max:
        value = damage
    elif v_max < np.sqrt((1 - (rainfall_max**2 / y_max**2)) * x_max**2):
        value = 0
    # elif ((v_max < x_max)  and  (rainfall_max_6h < y_max) ):
    # elif (v_max < x_max ):
    # value = 0
    else:
        value = np.nan

    return value


combined_input_data["DAM_perc_dmg"] = combined_input_data[
    ["HAZ_v_max", "HAZ_rainfall_Total", "DAM_perc_dmg"]
].apply(set_zeros, axis="columns")
```

```python
# TODO: I thought we want to keep NA damage values
# Remove NA values
combined_input_data = combined_input_data[combined_input_data["DAM_perc_dmg"].notnull()]
combined_input_data
```

```python
# Create cubed wind feature
combined_input_data["HAZ_v_max_3"] = combined_input_data["HAZ_v_max"].apply(
    lambda x: x * x * x
)
```

```python
# Drop Mun_Code since it's not a feature
combined_input_data = combined_input_data.drop(columns="Mun_Code")
```

```python
# TODO: seems to have no effect - remove?
df = combined_input_data.dropna()
display(df)
```

## Correlations

```python
# TODO: This should be run separately
# The correlation Matrix is also done for the input data where the damage value is greater than 10.
# df = df[df['DAM_perc_dmg'] > 10]
# df
```

```python
# get the correlation matrix
fig, ax = plt.subplots()

corrMatrix = df.corr()

plt.rcParams["figure.figsize"] = (48, 48)

sn.set(font_scale=2.5)
heatmap = sn.heatmap(
    corrMatrix, annot=True, cbar_kws={"shrink": 0.5}, annot_kws={"size": 16}
)
heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 42}, pad=18)

if save_fig:
    fig.savefig(output_dir / "corr_matrix.png", format="png")

plt.show()
```

```python
# get the correlation matrix (creating a square matrix with dimensions equal to the number of features)
# get the absolute value of correlation

fig, ax = plt.subplots()

corrMatrix_abs = df.corr().abs()

plt.rcParams["figure.figsize"] = (48, 48)

sn.set(font_scale=2.5)
heatmap = sn.heatmap(
    corrMatrix_abs, annot=True, cbar_kws={"shrink": 0.5}, annot_kws={"size": 16}
)
heatmap.set_title("Correlation Heatmap (abs)", fontdict={"fontsize": 42}, pad=18)

if save_fig:
    fig.savefig(output_dir / "corr_matrix_abs.png", format="png")

plt.show()
```

```python
pair = (
    corrMatrix_abs.where(np.triu(np.ones(corrMatrix_abs.shape), k=1).astype(np.bool))
    .stack()
    .sort_values(ascending=True)
)
pairs = pair[pair.gt(0.8)]
print(pairs)
```

```python
# Correlation matrix will be mirror image(all the diagonal elements=1).
# does not matter that we select the upper triangular or
# lower triangular part of the correlation matrix.

upper_tri = corrMatrix_abs.where(
    np.triu(np.ones(corrMatrix_abs.shape), k=1).astype(np.bool)
)
```

```python
drop_value = 0.80
to_drop = [
    column for column in upper_tri.columns if any(upper_tri[column] > drop_value)
]
print(to_drop)
```

```python
df = df.drop(df[to_drop], axis=1)
# print(df.head())

# %%
df = df.drop("typhoon", axis=1)
df
```

```python
df.columns.tolist()
```

```python
fig, ax = plt.subplots()

corrMatrix = df.corr()
# print (corrMatrix)

plt.rcParams["figure.figsize"] = (48, 48)
# plt.figure(figsize=(36,36))

sn.set(font_scale=2.5)
heatmap = sn.heatmap(
    corrMatrix, annot=True, cbar_kws={"shrink": 0.5}, annot_kws={"size": 16}
)
heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 42}, pad=18)

if save_fig:
    fig.savefig(output_dir / "corr_matrix_drop.png", format="png")

plt.show()
```

```python
names = df.columns.tolist()
names

display(df[names])
```

```python
"""
VIF is another method for finding highly correlated features if there is still in existence.
VIF method, picks each feature and regresses it against all of the other features so VIF value for a feature 
demonstrates the correlation of that feature in total with all the other ones, and not only with one specific feature.
Normally if the estimated VIF value for a feature is greater than 7 so it can be considered a highly correlated feature.
"""

# Implementing VIF

# The independent variables set
X = df[names]
# Creating VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns

# Calculating VIF for each feature
vif_data["VIF"] = [
    variance_inflation_factor(X.values, i) for i in range(len(X.columns))
]


print(vif_data)
```

```python
vif_data_sort = vif_data
vif_data_sort = vif_data.sort_values("VIF")
vif_data_sort = vif_data_sort.reset_index(drop=True)
display(vif_data_sort)
```

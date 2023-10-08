# Correlation Matrix

This code is utilized to compute the correlation among various
features present in the input dataset.

```python
%load_ext jupyter_black
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

from utils import get_training_dataset
```

```python
# Read csv file and import to df
df = pd.read_csv("data/updated_corr.csv")
df.head()
```

## You can check the correlation among features before and after

## removing windspeed == 0

```python
# Remove wind_speed where values==0
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df = df.drop(columns=["grid_point_id", "typhoon_year"])
df.head()
```

```python
# Estimate correlation matrix

fig, ax = plt.subplots()

corrMatrix = df.corr()

plt.rcParams["figure.figsize"] = (10, 10)

sn.set(font_scale=1)
heatmap = sn.heatmap(
    corrMatrix,
    annot=True,
    cbar_kws={"shrink": 0.5},
    annot_kws={"size": 7},
    cmap="vlag_r",
    center=0,
)

# sn.diverging_palette(220, 20, as_cmap=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va="top", color="red")

heatmap.set_title("Correlation Heatmap", fontdict={"fontsize": 10}, pad=22)

plt.show()
```

![png](output_6_0.png)

```python
# The absolute value of correlation

fig, ax = plt.subplots()

corrMatrix_abs = df.corr().abs()
# print (corrMatrix)

plt.rcParams["figure.figsize"] = (9, 6)

sn.set(font_scale=1)
heatmap = sn.heatmap(
    corrMatrix_abs,
    annot=True,
    cbar_kws={"shrink": 0.5},
    annot_kws={"size": 8},
)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va="top", color="red")

heatmap.set_title("Correlation Heatmap (abs)", fontdict={"fontsize": 12},
    pad=22)

plt.show()
```

![png](output_7_0.png)

```python
# Print out correlated pairs of features
pair = (
    corrMatrix_abs.where(np.triu(np.ones(corrMatrix_abs.shape), k=1)
    .astype(np.bool))
    .stack()
    .sort_values(ascending=True)
)
pairs = pair[pair.gt(0.8)]

if len(pairs) > 0:
    print(pairs)
else:
    print("No correlated features found!")
```

mean_slope       std_tri             0.819225
mean_tri         std_tri             0.819606
rural            water               0.829211
std_slope        mean_tri            0.852646
mean_slope       std_slope           0.854719
rainfall_max_6h  rainfall_max_24h    0.922418
total_houses     total_pop           0.961985
std_slope        std_tri             0.989076
mean_slope       mean_tri            0.998442
dtype: float64

### Simple Scatterplot to show relation between ‘total_buildings’ and

### ‘relative_wealth_index’, coloured by damage

```python
# For All damaged values
x = df["rwi"]
y = df["total_houses"]
plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 100})
plt.scatter(
    x,
    y,
    c=df["percent_houses_damaged"],
    cmap="viridis_r",
    s=5,
    vmin=0,
    vmax=100,
)

plt.colorbar().set_label("low To high damage", size=12, color="green")
plt.title("RWI vs total_houses")
plt.xlabel("RWI")
plt.ylabel("total_houses")
plt.show()
```

![png](output_10_0.png)

```python
# For damaged values > 10
x = df[df.percent_houses_damaged > 10]["rwi"]
y = df[df.percent_houses_damaged > 10]["total_houses"]
plt.rcParams.update({"figure.figsize": (8, 6), "figure.dpi": 100})
plt.scatter(
    x,
    y,
    c=df[df.percent_houses_damaged > 10]["percent_houses_damaged"],
    cmap="viridis_r",
    s=5,
    vmin=0,
    vmax=100,
)
# plt.scatter(x, y, c='red', cmap=df['percent_buildings_damaged'], s=20)
plt.colorbar().set_label("low To high damage", size=12, color="green")
plt.title("RWI vs total_houses")
plt.xlabel("RWI")
plt.ylabel("total_houses")
plt.show()
```

![png](output_11_0.png)

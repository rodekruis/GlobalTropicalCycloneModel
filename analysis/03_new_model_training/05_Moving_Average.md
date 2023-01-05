# Simple Moving Average(Running Mean)

It is used to observe how the two variables are related to each other.

```python
%load_ext jupyter_black
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import get_training_dataset
```

```python
# Read csv file and import to df
df = get_training_dataset()
```

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df.head()
```

```python
# Define a Function to plot the relation between each features
def ave(ftr_1, ftr_2):
    roll = (
        df.sort_values(ftr_1)
        .reset_index(drop=True)[ftr_2]
        .rolling(window=500)
        .mean()
    )

    re_ftr_1 = df.sort_values(ftr_1).reset_index(drop=True)[ftr_1]
    re_ftr_2 = df.sort_values(ftr_1).reset_index(drop=True)[ftr_2]

    plt.xlabel(ftr_1)
    plt.ylabel(ftr_2)
    plt.title(f"{ftr_1} vs {ftr_2}")
    return plt.plot(re_ftr_1, re_ftr_2, "*b"), plt.plot(re_ftr_1, roll, "r")
```

```python
# List of features while the last feature is the target
features = [
    "wind_speed",
    "track_distance",
    "total_buildings",
    "percent_buildings_damaged",
]
```

```python
# Call the function wrt those features you are interested
# in their running average's plot
ave(features[2], features[0])

# Use the log scale in x or y axis to have a zoom in scatter plot
plt.xscale("log")
```

## How each varriable is relared to target

```python
# wind_speed and percent_buildings_damaged
ave(features[0], features[3])
```

```python
# track_distance and percent_buildings_damaged
ave(features[1], features[3])
```

```python
# total_buildings and percent_buildings_damaged
ave(features[2], features[3])
plt.xscale("log")
```

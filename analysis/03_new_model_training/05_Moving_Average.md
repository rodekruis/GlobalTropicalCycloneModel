# Simple Moving Average(Running Mean)

## NOTE: This notebook uses total buildings

## which is no longer in the feature data set

It is used to observe how the two variables are related to each other.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from utils import get_training_dataset
```

```python
# Read csv file and import to df
df = get_training_dataset()
```

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df
```

```python
# Define a Function to plot the relation between each features
def ave(ftr_1, ftr_2):
    roll = (
        df.sort_values(ftr_1).reset_index(drop=True)[ftr_2].rolling(window=500).mean()
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
    "rainfall_max_6h",
    "rainfall_max_24h",
    "percent_buildings_damaged",
]
```

## Call the function wrt those features you are interested in their

## running average's plot

```python
ave(features[2], features[0])

# Use the log scale in x or y axis to have a zoom in scatter plot
plt.xscale("log")
```

## How each varriable is relared to target

```python
# wind_speed and percent_buildings_damaged
ave(features[0], features[5])
```

```python
# track_distance and percent_buildings_damaged
ave(features[1], features[5])
```

```python
# total_buildings and percent_buildings_damaged
ave(features[2], features[5])
plt.xscale("log")
```

```python
ave(features[3], features[5])
```

```python
# The trendline between rainfall_max_6h and rainfall_max_24h
fig = px.scatter(
    df,
    x="rainfall_max_6h",
    y="rainfall_max_24h",
    trendline="lowess",
    trendline_options=dict(frac=0.01),
    trendline_color_override="red",
    title="rainfall_max_6h vs rainfall_max_24h",
)
fig.update_layout(
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=False),
    plot_bgcolor="rgba(250,250,250,250)",
)

fig.show()
```

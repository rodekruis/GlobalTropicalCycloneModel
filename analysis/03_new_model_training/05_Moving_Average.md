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
```

```python
# wind_speed and total_buildings

rolling_wind = (
    df.sort_values("total_buildings")
    .reset_index(drop=True)["wind_speed"]
    .rolling(window=500)
    .mean()
)

tot_buildings = df.sort_values("total_buildings").reset_index(drop=True)[
    "total_buildings"
]
wind = df.sort_values("total_buildings").reset_index(drop=True)["wind_speed"]

plt.xlabel("total_buildings")
plt.ylabel("wind_speed")
plt.plot(tot_buildings, wind, "*b")
plt.plot(tot_buildings, rolling_wind, "r")
plt.title("total_buildings and wind_speed")

plt.xscale("log")
```

```python
# wind_speed and track_distance

rolling_wind = (
    df.sort_values("wind_speed")
    .reset_index(drop=True)["track_distance"]
    .rolling(window=500)
    .mean()
)

wind = df.sort_values("wind_speed").reset_index(drop=True)["wind_speed"]
track_dis = df.sort_values("wind_speed").reset_index(drop=True)[
    "track_distance"
]

plt.xlabel("wind_speed")
plt.ylabel("track_distance")
plt.plot(wind, track_dis, "*b")
plt.plot(wind, rolling_wind, "r")
plt.title("wind_speed and track_distance")

# plt.yscale("log")
```

```python
# total_buildings and track_distance

rolling_tot_buildings = (
    df.sort_values("total_buildings")
    .reset_index(drop=True)["track_distance"]
    .rolling(window=500)
    .mean()
)

tot_buildings = df.sort_values("total_buildings").reset_index(drop=True)[
    "total_buildings"
]
track_dis = df.sort_values("total_buildings").reset_index(drop=True)[
    "track_distance"
]

plt.xlabel("total_buildings")
plt.ylabel("track_distance")
plt.plot(tot_buildings, track_dis, "*b")
plt.plot(tot_buildings, rolling_tot_buildings, "r")
plt.title("total_buildings and track_distance")

plt.xscale("log")
```

## How each varriable is relared to target

```python
# wind_speed and percent_buildings_damaged

rolling_wind = (
    df.sort_values("wind_speed")
    .reset_index(drop=True)["percent_buildings_damaged"]
    .rolling(window=550)
    .mean()
)

wind = df.sort_values("wind_speed").reset_index(drop=True)["wind_speed"]
dam_buildings = df.sort_values("wind_speed").reset_index(drop=True)[
    "percent_buildings_damaged"
]

plt.xlabel("wind_speed")
plt.ylabel("percent_buildings_damaged")

plt.plot(wind, dam_buildings, "*b")
plt.plot(
    wind,
    rolling_wind,
    "r",
)
plt.title("wind_speed and percent_buildings_damaged")
# plt.yscale("log")
```

```python
# total_buildings and percent_buildings_damaged

rolling_tot_buildings = (
    df.sort_values("total_buildings")
    .reset_index(drop=True)["percent_buildings_damaged"]
    .rolling(window=700)
    .mean()
)

tot_buildings = df.sort_values("total_buildings").reset_index(drop=True)[
    "total_buildings"
]
dam_buildings = df.sort_values("total_buildings").reset_index(drop=True)[
    "percent_buildings_damaged"
]

plt.xlabel("total_buildings")
plt.ylabel("percent_buildings_damaged")

plt.plot(tot_buildings, dam_buildings, "*b")
plt.plot(
    tot_buildings,
    rolling_tot_buildings,
    "r",
)
plt.title("total_buildings and percent_buildings_damaged")
plt.xscale("log")
```

```python
# track_distance and percent_buildings_damaged

rolling_track_distance = (
    df.sort_values("track_distance")
    .reset_index(drop=True)["percent_buildings_damaged"]
    .rolling(window=700)
    .mean()
)

track_dis = df.sort_values("track_distance").reset_index(drop=True)[
    "track_distance"
]
dam_buildings = df.sort_values("track_distance").reset_index(drop=True)[
    "percent_buildings_damaged"
]

plt.xlabel("track_distance")
plt.ylabel("percent_buildings_damaged")

plt.plot(track_dis, dam_buildings, "*b")
plt.plot(
    track_dis,
    rolling_track_distance,
    "r",
)
plt.title("track_distance and percent_buildings_damaged")
# plt.xscale("log")
```

```python
dam_buildings[dam_buildings > 0].shape
```

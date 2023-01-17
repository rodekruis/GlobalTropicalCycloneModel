# Gather the data needed to train the model

In this notebook we combine all of the data from
step 2. The contents of this notebook is mirrored
in `utils.py` so that it can be used in other notebooks.

```python
%load_ext jupyter_black
```

```python
from pathlib import Path
import os

import numpy as np
import pandas as pd
```

```python
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis/02_new_model_input"
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis/03_new_model_training"
)
```

## Read in buliding damage

```python
# Read in the building damage data
filename = input_dir / "02_housing_damage/output/building_damage_bygrid.csv"

df_damage = pd.read_csv(filename)
df_damage.columns
```

```python
# Select and rename columns,
# drop any rows that don't have a typhoon name
columns_to_keep = {
    "id": "grid_point_id",
    "numbuildings_bygrid": "total_buildings",
    "typhoon": "typhoon_name",
    "Year": "typhoon_year",
    "Totally_Damaged_bygrid": "total_buildings_damaged",
}

df_damage = (
    df_damage.dropna(subset="typhoon")
    .loc[:, list(columns_to_keep.keys())]
    .rename(columns=columns_to_keep)
)
df_damage["typhoon_name"] = df_damage["typhoon_name"].str.upper()
for column_name in ["typhoon_year", "grid_point_id"]:
    df_damage[column_name] = df_damage[column_name].astype(int)

df_damage
```

## Read in windfield

```python
# Read in the data file

filename = input_dir / "01_windfield/windfield_data.csv"

df_windfield = pd.read_csv(filename)
df_windfield.columns
```

```python
# Select columns
columns_to_keep = [
    "typhoon_name",
    "typhoon_year",
    "grid_point_id",
    "wind_speed",
    "track_distance",
]
df_windfield = df_windfield.loc[:, columns_to_keep]
df_windfield
```

## Read in rainfall

```python
filename = input_dir / "03_rainfall/output/rainfall_data_mean.csv"
df_rainfall = pd.read_csv(filename)
df_rainfall[["typhoon_name", "typhoon_year"]] = df_rainfall[
    "typhoon"
].str.split("(\d+)", expand=True)[[0, 1]]
df_rainfall["typhoon_name"] = df_rainfall["typhoon_name"].str.upper()
df_rainfall["typhoon_year"] = df_rainfall["typhoon_year"].astype(int)
df_rainfall = df_rainfall.rename(columns={"id": "grid_point_id"}).loc[
    :,
    [
        "typhoon_name",
        "typhoon_year",
        "grid_point_id",
        "rainfall_max_6h",
        "rainfall_max_24h",
    ],
]
df_rainfall
```

## Merge the datasets

```python
index = ["typhoon_name", "typhoon_year", "grid_point_id"]
object_list = [df_damage, df_rainfall]

# First merge all that are not the windfield, since
# windfield has all the gridpoints that we want
df_all = pd.concat(
    objs=[df.set_index(index) for df in object_list], axis=1, join="outer"
)

# For now do a left join to the windfield, since it has the exact points we want
df_all = df_windfield.set_index(index).merge(
    df_all, left_index=True, right_index=True, how="left"
)

df_all
```

```python
# TODO: remove this if the building dataset is fixed
# Get the number of buildings associated with a gridpoint,
# and fill in the missing values
building_number_dict = (
    df_damage.loc[
        :,
        ["grid_point_id", "total_buildings"],
    ]
    .set_index("grid_point_id")["total_buildings"]
    .to_dict()
)

df_all["total_buildings"] = (
    df_all.reset_index()["grid_point_id"].map(building_number_dict).values
)
df_all
```

## Clean the dataset

```python
df = df_all.fillna(0)
```

```python
df_all.columns
```

```python
# Assume that NAs are all 0s
df_all = df_all.fillna(0)
# Drop rows with 0 buildings
df_all = df_all[df_all["total_buildings"] != 0]
```

```python
# TODO: Remove this if it's fixed in the data
# Create percentage damage column
# Check if total damaged buildings is greater than total buildings
too_few_buildings = (
    df_all["total_buildings"] < df_all["total_buildings_damaged"]
)
sum(too_few_buildings)
```

```python
# TODO: Remove this if it's fixed in the data
# At the moment some cells have more damaged buildings than buildings,
# so adjust the maximum
df_all.loc[too_few_buildings, "total_buildings"] = df_all.loc[
    too_few_buildings, "total_buildings_damaged"
]
```

```python
# Calculate percentage
df_all["percent_buildings_damaged"] = (
    df_all["total_buildings_damaged"] / df_all["total_buildings"] * 100
)
df_all = df_all.drop(columns="total_buildings_damaged")
```

```python
df_all
```

## Write out dataset

```python
df_all.reset_index().to_csv(
    output_dir / "new_model_training_dataset.csv", index=False
)
```

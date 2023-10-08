# Create a historical variable

In Historical Risk Model, the dataset is being updated with
the addition of a new variable called "percent_buildings_damaged_5years".

This variable will be incorporated into the existing dataset.
For every data point or location, an average of damages caused
by typhoons over the past 5 years will be calculated and recorded
as the value for this new variable.

```python
%load_ext jupyter_black
```

```python
import statistics
import numpy as np
import pandas as pd
from pathlib import Path
import os
from utils import get_training_dataset
```

```python
output_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis/03_new_model_training"
)
```

```python
# Read csv file and import to df
df = get_training_dataset()
df.head()
```

```python
# Groupby df based on "typhoon_year" and "grid_point_id" and "percent_houses_damaged"
df_avgDmgCell_and_Year = df.groupby(
    ["typhoon_year", "grid_point_id"], as_index=False
)["percent_houses_damaged"].mean()
df_avgDmgCell_and_Year
```

```python
# Calculate the average damaged of past 5 years for each point
df_res2 = (
    df_avgDmgCell_and_Year.groupby("grid_point_id")
    .rolling(5, min_periods=1)
    .agg({"percent_houses_damaged": "mean", "typhoon_year": "max"})
)

df_res2 = df_res2.rename(
    columns={"percent_houses_damaged": "percent_houses_damaged_5years"}
)
```

```python
df_res2["typhoon_year"] = df_res2["typhoon_year"] + 1
df_res2
```

```python
# Join this new variable to the main df wrt "typhoon_year" and "grid_point_id"
df2 = df.merge(df_res2, on=["typhoon_year", "grid_point_id"], how="left")
df2["percent_houses_damaged_5years"] = df2[
    "percent_houses_damaged_5years"
].fillna(0)

df2
```

```python
# Save this df to a CSV file
df2.to_csv(output_dir / "new_model_training_dataset.csv", index=False)
```

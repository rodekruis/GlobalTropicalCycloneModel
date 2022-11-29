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

import pandas as pd
```

```python
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis/02_new_model_input"
```

```python
# Read in the building damage data. Drop any
# rows that have NA for the typhoon name.
filename = (
    input_dir
    / "02_housing_damage/output/percentage_building_damage_bygrid.csv"
)
df_damage = pd.read_csv(filename).dropna(subset="typhoon")
```

```python
df_damage.columns
```

```python

```

```python
columns_to_keep = [
    "id",
    "pcode",
    "typhoon",
    "Year",
    "Totally",
    "Totally_Damaged_bygrid",
]
df_damage2 = df_damage.loc[:, columns_to_keep]
df_damage2
```

```python

```

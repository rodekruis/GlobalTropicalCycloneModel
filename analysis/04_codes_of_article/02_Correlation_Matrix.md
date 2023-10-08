# Main correlation matrix

During the feature selection part of the model,
we figured out that very different sets of features were chosen
in different runs. Hence, a decision was made to search
or highly correlated features among all the features
in the dataset.

The following code looks for highly correlated features in the
model's input data.

```python
%load_ext jupyter_black
```

```python
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
```

```python
df = pd.read_csv("data/510_data.csv", index_col=0)
df
```

```python
sn.set_theme(style="white")

corrMatrix = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corrMatrix, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sn.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sn.heatmap(
    corrMatrix,
    mask=mask,
    cmap=cmap,
    vmax=1,
    vmin=-1,
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.5},
)
plt.savefig("corr_matrix3.pdf", bbox_inches="tight")
```

![png](output_4_0.png)

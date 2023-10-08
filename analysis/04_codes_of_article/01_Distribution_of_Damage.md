# Intersection of two datasets grid and municipality

```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statistics
import warnings

from matplotlib.ticker import StrMethodFormatter
from sty import fg, rs
from matplotlib import cm
```

```python
df_mun_merged = pd.read_csv("data/df_merged_2.csv")

# Remove the duplicated rows and reset the indices
df_mun_merged.drop_duplicates(keep="first", inplace=True)
df_mun_merged = df_mun_merged.reset_index(drop=True)
df_mun_merged
```

```python
# The converted grid to mun dataset
final_df_new = pd.read_csv("final_df_new.csv")
final_df_new
```

```python
# Combine typhoon name and year columns

for i in range(len(final_df_new)):
    final_df_new.at[i, "typhoon_name"] = final_df_new.loc[i, "typhoon_name"] +
    str(
        final_df_new.loc[i, "typhoon_year"]
    )

final_df_new.drop(["typhoon_year"], axis=1, inplace=True)
final_df_new
```

```python
# Rename the columns
final_df_new = final_df_new.rename(
    columns={"ADM3_PCODE": "Mun_Code", "typhoon_name": "typhoon"}
)
final_df_new.head()
```

## Joined two datasets together (grid and municipality)

```python
# Merge DataFrames based on 'typhoon_name' and 'Mun_Code'
merged_df = pd.merge(
    final_df_new, df_mun_merged, on=["Mun_Code", "typhoon"], how="inner"
)
merged_df
```

```python
# makes the data
y1 = merged_df["y_norm_x"]
y2 = final_df_new["y_norm"]
colors = ["b", "r"]

# plots the histogram
fig, ax1 = plt.subplots()
ax1.hist(
    [y1, y2],
    color=colors,
    log=True,
    bins=20,
    label=["Original ground truth", "Grid to Municipality transformation"],
    width=2.3,
)
# ax1.set_xlim(-2, 100)
ax1.set_ylabel("Frequency", labelpad=20, size=12)
ax1.set_xlabel("Damage Percentage", labelpad=20, size=12)
plt.tight_layout()

plt.legend(loc="upper right")

# plt.savefig("figures/y_norm_histplot.pdf")
# Displaying the plot
plt.show()
```

![png](output_9_0.png)

```python
# Define the bins
bins = [0, 0.00009, 1, 10, 50, 100]

# Plotting the bar plots side by side
check_df1 = pd.cut(merged_df["y_norm_x"], bins=bins, include_lowest=True)
df1_counts = check_df1.value_counts(sort=False)

check_df2 = pd.cut(final_df_new["y_norm"], bins=bins, include_lowest=True)
df2_counts = check_df2.value_counts(sort=False)


# Calculate the x-axis positions for the bars
x = np.arange(len(df1_counts))
width = 0.35

# Plotting df1 bars
fig, ax = plt.subplots()
rects1 = ax.bar(
    x - width / 2,
    df1_counts,
    width,
    color="b",
    # alpha=0.5,
    log=True,
    label="Original ground truth",
)

# Plotting df2 bars
rects2 = ax.bar(
    x + width / 2,
    df2_counts,
    width,
    color="r",
    # alpha=0.5,
    log=True,
    label="Grid to Municipality transformation",
)

# Adding counts for df1
for rect in rects1:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        height + 0.5,
        height,
        ha="center",
        va="bottom",
    )

# Adding counts for df2
for rect in rects2:
    height = rect.get_height()
    ax.text(
        rect.get_x() + rect.get_width() / 2,
        height + 0.5,
        height,
        ha="center",
        va="bottom",
    )

# Adjusting y-axis limits
ax.set_ylim(0, max(max(df1_counts), max(df2_counts)) * 2)


# Adding labels and title to the plot
ax.set_xlabel("Damage Percentage", labelpad=20, size=12)
ax.set_ylabel("Frequency", labelpad=20, size=12)
ax.set_title("")
ax.set_xticks(x)
ax.set_xticklabels(df1_counts.index)

plt.tight_layout()
ax.legend()

# Displaying the plot
# fig.savefig("figures/y_norm_barplots.pdf")
plt.show()
```

Invalid limit will be ignored.
  ax.set_ylim(0, max(max(df1_counts), max(df2_counts)) * 2)

![png](output_10_1.png)

```python

```

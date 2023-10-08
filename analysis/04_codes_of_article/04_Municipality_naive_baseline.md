# Naive baseline

## municipality-based dataset

```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import statsmodels.api as sm
import xgboost as xgb
import pandas as pd
import numpy as np
import statistics
import os

from math import sqrt
from collections import defaultdict
from sklearn import preprocessing
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
```

```python
# Import the CSV file to a dataframe
df = pd.read_csv("data/df_merged_2.csv")
df
```

```python
# Hist plot after data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["y_norm"], bins=bins2)
```

```python
print(samples_per_bin2)
print(binsP2)
```

[3606 2874 1141  388   64]
[0.00e+00 9.00e-05 1.00e+00 1.00e+01 5.00e+01 1.01e+02]

```python
bin_index2 = np.digitize(df["y_norm"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
# Split X and y from dataframe features

X = pd.Series([0] * 8073)
y = df["y_norm"]
```

```python
# Defin two lists to save RMSE and Average Error

RMSE = defaultdict(list)
AVE = defaultdict(list)
```

```python
for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        df["y_norm"],
        stratify=y_input_strat,
        test_size=0.2,
    )

    # create a dummy regressor
    dummy_reg = DummyRegressor(strategy="mean")

    # fit it on the training set
    dummy_reg.fit(X_train, y_train)

    # make predictions on the test set
    y_pred = dummy_reg.predict(X_test)
    y_pred_clipped = y_pred.clip(0, 100)

    # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based
    rmse = sqrt(mean_squared_error(y_test, y_pred_clipped))
    ave = (y_pred_clipped - y_test).sum() / len(y_test)

    print(f"RMSE for mun_based model: {rmse:.2f}")
    print(f"Average Error for mun_based model: {ave:.2f}")

    RMSE["all"].append(rmse)
    AVE["all"].append(ave)

    bin_index = np.digitize(y_test, bins=binsP2)

    for bin_num in range(1, 6):

        mse_idx = mean_squared_error(
            y_test[bin_index == bin_num],
            y_pred_clipped[bin_index == bin_num],
        )
        rmse = np.sqrt(mse_idx)

        ave = (
            y_pred_clipped[bin_index == bin_num] - y_test[bin_index == bin_num]
        ).sum() / len(y_test[bin_index == bin_num])

        RMSE[bin_num].append(rmse)
        AVE[bin_num].append(ave)
```

RMSE for mun_based model: 8.30
Average Error for mun_based model: -0.12
RMSE for mun_based model: 7.48
Average Error for mun_based model: 0.14
RMSE for mun_based model: 7.96
Average Error for mun_based model: -0.08
RMSE for mun_based model: 7.64
Average Error for mun_based model: 0.01
RMSE for mun_based model: 7.90
Average Error for mun_based model: -0.01
RMSE for mun_based model: 8.38
Average Error for mun_based model: -0.05
RMSE for mun_based model: 8.51
Average Error for mun_based model: -0.09
RMSE for mun_based model: 7.91
Average Error for mun_based model: -0.08
RMSE for mun_based model: 7.86
Average Error for mun_based model: 0.04
RMSE for mun_based model: 8.01
Average Error for mun_based model: -0.03
RMSE for mun_based model: 8.18
Average Error for mun_based model: -0.08
RMSE for mun_based model: 8.24
Average Error for mun_based model: -0.05
RMSE for mun_based model: 7.85
Average Error for mun_based model: -0.00
RMSE for mun_based model: 8.15
Average Error for mun_based model: -0.12
RMSE for mun_based model: 8.01
Average Error for mun_based model: -0.08
RMSE for mun_based model: 8.23
Average Error for mun_based model: -0.11
RMSE for mun_based model: 7.95
Average Error for mun_based model: 0.10
RMSE for mun_based model: 7.94
Average Error for mun_based model: 0.01
RMSE for mun_based model: 8.01
Average Error for mun_based model: 0.01
RMSE for mun_based model: 8.01
Average Error for mun_based model: -0.01

```python
# Define a function to plot RMSEs
def rmse_ave_mean(rmse, ave):

    # Mean of RMSE and Standard deviation
    m_rmse = statistics.mean(rmse)
    sd_rmse = statistics.stdev(rmse)

    m_ave = statistics.mean(ave)
    sd_ave = statistics.stdev(ave)

    print(f"mean_RMSE: {m_rmse:.2f}")
    print(f"stdev_RMSE: {sd_rmse:.2f}")

    print(f"mean_average_error: {m_ave:.2f}")
    print(f"stdev_average_error: {sd_ave:.2f}")
```

```python
print("RMSE and Average Error in total", "\n")
rmse_ave_mean(RMSE["all"], AVE["all"])
```

RMSE and Average Error in total

mean_RMSE: 8.03
stdev_RMSE: 0.24
mean_average_error: -0.03
stdev_average_error: 0.07

```python
for bin_num in range(1, 6):

    print(f"\n RMSE and Average Error per bin {bin_num}\n")
    rmse_ave_mean(RMSE[bin_num], AVE[bin_num])
```

  RMSE and Average Error per bin 1

mean_RMSE: 2.22
stdev_RMSE: 0.01
mean_average_error: 2.22
stdev_average_error: 0.01

  RMSE and Average Error per bin 2

mean_RMSE: 2.03
stdev_RMSE: 0.02
mean_average_error: 2.01
stdev_average_error: 0.02

  RMSE and Average Error per bin 3

mean_RMSE: 2.82
stdev_RMSE: 0.16
mean_average_error: -1.42
stdev_average_error: 0.16

  RMSE and Average Error per bin 4

mean_RMSE: 24.81
stdev_RMSE: 1.15
mean_average_error: -21.77
stdev_average_error: 1.18

  RMSE and Average Error per bin 5

mean_RMSE: 60.88
stdev_RMSE: 3.39
mean_average_error: -59.97
stdev_average_error: 3.03

```python

```

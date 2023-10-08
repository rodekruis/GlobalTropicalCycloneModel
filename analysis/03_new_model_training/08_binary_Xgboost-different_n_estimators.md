# XGBoost (different n_estimators)

We plot and compare the f1_score macro average VS different n_estimators

```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import imblearn

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from collections import Counter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

from utils import get_training_dataset
```

```python
# Read csv file and import to df
df = get_training_dataset()
df.head()
```

```python
# Fill NaNs with average estimated value of 'rwi'
df["rwi"].fillna(df["rwi"].mean(), inplace=True)

# Set any values >100% to 100%,
for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] > 100:
        df.at[i, "percent_houses_damaged"] = float(100)
```

```python
# define a threshold to separate target into damaged and not_damaged
thres = 10.0

for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] >= thres:
        df.at[i, "binary_damage"] = 1
    else:
        df.at[i, "binary_damage"] = 0

df["binary_damage"] = df["binary_damage"].astype("int")

# Remove previous target 'percent_buildings_damaged' from the dataframe
df = df.drop(["percent_houses_damaged"], axis=1)
df
```

```python
plt.figure(figsize=(4, 3))
sns.countplot(x="binary_damage", data=df, palette="hls")
plt.title("bar_plot (counts of observations)")
plt.show()
```

![png](output_6_0.png)

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df = df.drop(columns=["grid_point_id", "typhoon_year"])
df.head()
```

```python
# Show histogram of damage
df.hist(column="binary_damage", figsize=(4, 3))
```

array([[<AxesSubplot:title={'center':'binary_damage'}>]], dtype=object)

![png](output_8_1.png)

```python
# Define bins for data stratification
bins2 = [0, 0.1, 1]
samples_per_bin2, binsP2 = np.histogram(df["binary_damage"], bins=bins2)
```

```python
# Check the bins' intervalls (first bin means all zeros,
# second bin means 0 < values <= 1)
df["binary_damage"].value_counts(bins=binsP2)
```

(-0.001, 0.1]    48685
(0.1, 1.0]        1069
Name: binary_damage, dtype: int64

```python
print(samples_per_bin2)
print(binsP2)
```

[48685  1069]
[0.  0.1 1. ]

```python
bin_index2 = np.digitize(df["binary_damage"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    "rainfall_max_24h",
    "rwi",
    "mean_slope",
    "std_slope",
    "mean_tri",
    "std_tri",
    "mean_elev",
    "coast_length",
    "with_coast",
    "urban",
    "rural",
    "water",
    "total_pop",
    "percent_houses_damaged_5years",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["binary_damage"]
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rainfall_max_24h', 'rwi', 'mean_slope', 'std_slope', 'mean_tri',
        'std_tri', 'mean_elev', 'coast_length', 'with_coast', 'urban', 'rural',
        'water', 'total_pop', 'percent_houses_damaged_5years'],
      dtype='object')

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, df["binary_damage"], stratify=y_input_strat, test_size=0.2
)
```

```python
# Check train data before resampling
print(Counter(y_train))
```

Counter({0: 38948, 1: 855})

```python
# Ask the user whether to perform oversampling or undersampling
sampling_type = int(
    input(
        "Enter 1 for oversampling, " +
        "2 for undersampling or 3 for combination of both: "
    )
)
```

Enter 1 for oversampling, 2 for undersampling or 3 for combination of both: 2

```python
if sampling_type == 1:
    # Define oversampling strategy
    over = RandomOverSampler(sampling_strategy=0.1)
    # Fit and apply the transform
    X_train, y_train = over.fit_resample(X_train, y_train)

elif sampling_type == 2:
    under = RandomUnderSampler(sampling_strategy=0.7)
    X_train, y_train = under.fit_resample(X_train, y_train)

elif sampling_type == 3:
    over = RandomOverSampler(sampling_strategy=0.1)
    X_train, y_train = over.fit_resample(X_train, y_train)

    under = RandomUnderSampler(sampling_strategy=0.7)
    X_train, y_train = under.fit_resample(X_train, y_train)


else:
    print("Invalid input. Please enter 1, 2 or 3.")
```

```python
# Check train data after resampling
print(Counter(y_train))
```

Counter({0: 1221, 1: 855})

```python
# Define an empty list to keep f1_score of each n_estimator
f1_lst = []

# List of 10 different n_estimator from 10 to 100
n_estimator_lst = [12, 22, 32, 42, 52, 62, 72, 82, 92, 102]

for i in range(len(n_estimator_lst)):
    # Use XGBClassifier as a Machine Learning model to fit the data
    xgb_model = XGBClassifier(n_estimators=n_estimator_lst[i])

    eval_set = [(X_train, y_train), (X_test, y_test)]
    # eval_set = [(X_test, y_test)]
    xgb_model.fit(
        X_train,
        y_train,
        eval_metric=["error", "logloss"],
        eval_set=eval_set,
        verbose=False,
    )

    y_pred = xgb_model.predict(X_test)
    f1_lst.append(f1_score(y_test, y_pred, average="macro"))
```

```python
# Display the f1_score list obtained from xgboost model in the loop
display(f1_lst)
display(n_estimator_lst)
```

[0.6676821547919375,
0.6703459732248127,
0.674495899785816,
0.6730279686949603,
0.6767305863765603,
0.6746622499771243,
0.6776427826590434,
0.6767841121357867,
0.6769982423456928,
0.6772676461108323]

[12, 22, 32, 42, 52, 62, 72, 82, 92, 102]

```python
# Create a plot to compare n_estimator vs F1_score
x = n_estimator_lst
y = f1_lst
plt.rcParams.update({"figure.figsize": (6, 4), "figure.dpi": 100})
plt.plot(n_estimator_lst, f1_lst, marker="*", markeredgecolor="red", color="blue")
plt.title("F1-macro average vs n_estimator")
plt.xlabel("n_estimator")
plt.ylabel("F1-macro average")
plt.xticks([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.xlim(0, 110)
plt.show()
```

![png](output_22_0.png)

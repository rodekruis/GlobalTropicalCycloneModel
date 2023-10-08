# Binary Model - Random Forest

Based on the target values and using the test_trial method,
we have set a threshold of 10.0 to convert the
continuous target into a binary target.
Subsequently, we employ the Random Forest Classification algorithm
on the input dataset with the binary target.
We evaluate the model performance using the Confusion Matrix
and Classification Report.

```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import imblearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]
df = df.drop(columns=["grid_point_id", "typhoon_year"])
df.head()
```

```python
plt.figure(figsize=(4, 3))
sns.countplot(x="binary_damage", data=df, palette="hls")
plt.show()
```

![png](output_7_0.png)

```python
# Show histogram of damage
df.hist(column="binary_damage", figsize=(4, 3))
```

array([[<AxesSubplot:title={'center':'binary_damage'}>]], dtype=object)

![png](output_8_1.png)

```python
# Hist plot after data stratification
bins2 = [0, 0.1, 1]
samples_per_bin2, binsP2 = np.histogram(df["binary_damage"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

[<matplotlib.lines.Line2D at 0x7fbb99605bb0>]

![png](output_9_1.png)

```python
# Check the bins' intervalls (first bin means all zeros,
# second bin means 0 < values <= 1)
df["binary_damage"].value_counts(bins=binsP2)
```

(-0.001, 0.1]    48685
(0.1, 1.0]        1069
Name: binary_damage, dtype: int64

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
        "Enter 1 for oversampling," +
        " 2 for undersampling or 3 for combination of both: "
    )
)
```

Enter 1 for oversampling, 2 for undersampling or 3 for combination of both: 1

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
# Summarize class distribution of oversampling
print(Counter(y_train))
```

Counter({0: 38948, 1: 3894})

```python
# Define a list to keep f1_score of each n_estimator
f1_lst = []
n_estimator_lst = []
```

```python
# Random Forest regression model
model_clf = RandomForestClassifier()
model_clf.fit(X_train, y_train)

# Make predictions for the test set
y_pred = model_clf.predict(X_test)
```

```python
# View confusion matrix for test data and predictions
matrix = confusion_matrix(y_test, y_pred)
matrix
```

array([[9691,   46],
        [  87,  127]])

```python
# View the classification report for test data and predictions
print(classification_report(y_test, y_pred))
```

precision    recall  f1-score   support

0       0.99      1.00      0.99      9737
1       0.73      0.59      0.66       214

accuracy                           0.99      9951
macro avg       0.86      0.79      0.82      9951
weighted avg       0.99      0.99      0.99      9951

```python
# Binary absolute and relative with colorbar
fig, ax = plot_confusion_matrix(
    conf_mat=matrix,
    show_absolute=True,
    show_normed=True,
    colorbar=True,
    cmap=plt.cm.Greens,
)

ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted 0s", "Predicted 1s"))
ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual 0s", "Actual 1s"))
plt.title("Confusion Matrix for Random Forest Model")
plt.show()
```

![png](output_23_0.png)

## Feature Importance

```python
# Random Forest Built-in Feature Importance

plt.rcParams.update({"figure.figsize": (6.0, 4.0)})
plt.rcParams.update({"font.size": 10})

model_clf.feature_importances_

sorted_idx = model_clf.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], model_clf.feature_importances_[sorted_idx])
plt.xlabel("Built_in Feature Importance")
plt.title("Random Forest")
```

Text(0.5, 1.0, 'Random Forest')

![png](output_25_1.png)

```python

```

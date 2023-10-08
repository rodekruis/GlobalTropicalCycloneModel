# Binary Model - XGBoost

Based on the target values and using the test_trial method, we have set a
threshold of 10.0 to convert the continuous target into a binary target.
Subsequently, we employ the XGBoost Classification algorithm on the input
dataset with the binary target. We evaluate the model performance using the
Confusion Matrix and Classification Report, Log loss and Classification error
plots.

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
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from matplotlib import cm
from mlxtend.plotting import plot_confusion_matrix
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
# Hist plot after data stratification
bins2 = [0, 0.1, 1]
samples_per_bin2, binsP2 = np.histogram(df["binary_damage"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
plt.title("plot after data stratification")
```

Text(0.5, 1.0, 'plot after data stratification')

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

Enter 1 for oversampling, 2 for undersampling or 3 for combination of both: 3

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
# Check train data After resampling
print(Counter(y_train))
```

Counter({0: 5562, 1: 3894})

```python
# Define a list to keep f1_score of each n_estimator
f1_lst = []
n_estimator_lst = []
```

```python
# Use XGBClassifier as a Machine Learning model to fit the data
xgb_model = XGBClassifier()

eval_set = [(X_train, y_train), (X_test, y_test)]
# eval_set = [(X_test, y_test)]
xgb_model.fit(
    X_train,
    y_train,
    eval_metric=["error", "logloss"],
    eval_set=eval_set,
    verbose=False,
)
```

```python
y_pred = xgb_model.predict(X_test)
```

```python
# y_pred = xgb_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
```

array([[9461,  276],
        [  34,  180]])

```python
# Classification Report
print(metrics.classification_report(y_test, y_pred))
print(metrics.confusion_matrix(y_test, y_pred))
```

precision    recall  f1-score   support

0       1.00      0.97      0.98      9737
1       0.39      0.84      0.54       214

accuracy                           0.97      9951
macro avg       0.70      0.91      0.76      9951
weighted avg       0.98      0.97      0.97      9951

[[9461  276]
[  34  180]]

```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

Accuracy: 96.88%

```python
# Plot Confusion Matrix
fig, ax = plot_confusion_matrix(
    conf_mat=cm,
    show_absolute=True,
    show_normed=True,
    colorbar=True,
    cmap=plt.cm.Greens,
)

ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted 0s", "Predicted 1s"))
ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual 0s", "Actual 1s"))
plt.title("Confusion Matrix for XGBoost Model")
plt.show()
```

![png](output_26_0.png)

## Plot Log Loss and Classification Error

```python
results = xgb_model.evals_result()
epochs = len(results["validation_0"]["error"])
x_axis = range(0, epochs)
```

```python
f1_score(y_test, y_pred, average="macro")
```

0.7605972322248988

```python
# plot log loss
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x_axis, results["validation_0"]["logloss"], label="Train")
ax.plot(x_axis, results["validation_1"]["logloss"], label="Test")
ax.legend()

plt.xlabel("Predicted Probability")
plt.ylabel("Log Loss")
plt.title("XGBoost Log Loss")
plt.show()
```

![png](output_30_0.png)

```python
# plot classification error
fig, ax = plt.subplots(figsize=(4, 4))
ax.plot(x_axis, results["validation_0"]["error"], label="Train")
ax.plot(x_axis, results["validation_1"]["error"], label="Test")
ax.legend()

plt.xlabel("n_estimators")
plt.ylabel("Classification Error")
plt.title("XGBoost Classification Error")
plt.show()
```

![png](output_31_0.png)

## Feature Importance

```python
# Xgboost Built-in Feature Importance

# xgb_model.feature_importances_.argsort()[::-1]
plt.rcParams.update({"figure.figsize": (6.0, 4.0)})
plt.rcParams.update({"font.size": 10})

sorted_idx = xgb_model.feature_importances_.argsort()
plt.barh(X.columns[sorted_idx], xgb_model.feature_importances_[sorted_idx])
plt.xlabel("Built_in Feature Importance")
plt.title("Xgboost")
plt.show()
```

![png](output_33_0.png)

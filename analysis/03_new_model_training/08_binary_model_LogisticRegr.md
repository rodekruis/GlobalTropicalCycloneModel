# Binary Logistic Regression Model

Based on the target values and using the test_trial method,
we have set a threshold of 10.0 to convert the continuous target
into a binary target.
Subsequently, we employ the Logistic Regression algorithm on
the input dataset with the binary target.
We evaluate the model performance using the Confusion Matrix and
Classification Report.

```python
%load_ext jupyter_black
```

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import imblearn
import shap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from matplotlib import cm
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
# Define a threshold to separate target into damaged and not_damaged
thres = 10.0

for i in range(len(df)):
    if df.loc[i, "percent_houses_damaged"] >= thres:
        df.at[i, "binary_damage"] = 1
    else:
        df.at[i, "binary_damage"] = 0

df["binary_damage"] = df["binary_damage"].astype("int")

# Remove previous target 'percent_buildings_damaged' from the dataframe
df = df.drop(["percent_houses_damaged"], axis=1)

# Remove one of the rainfall features
df = df.drop(["rainfall_max_24h"], axis=1)
df
```

```python
# Show bar plot
plt.figure(figsize=(4, 3))
sns.countplot(x="binary_damage", data=df, palette="hls")
plt.title("bar_plot (counts of observations)")
plt.show()
```

![png](output_6_0.png)

```python
# Remove zeros from wind_speed
df = (df[(df[["wind_speed"]] != 0).any(axis=1)]).reset_index(drop=True)
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
# Define bins and data stratification
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
# For this model we need to remove highly correlated features obtained from
# correlation_matrix code
features = [
    "wind_speed",
    "track_distance",
    "total_houses",
    "rainfall_max_6h",
    # "rainfall_max_24h",
    "rwi",
    "mean_slope",
    # "std_slope",
    # "mean_tri",
    # "std_tri",
    # "mean_elev",
    "coast_length",
    # "with_coast",
    "urban",
    "rural",
    # "water",
    # "total_pop",
    "percent_houses_damaged_5years",
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["binary_damage"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```

Index(['wind_speed', 'track_distance', 'total_houses', 'rainfall_max_6h',
        'rwi', 'mean_slope', 'coast_length', 'urban', 'rural',
        'percent_houses_damaged_5years'],
      dtype='object')

```python
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, df["binary_damage"], stratify=y_input_strat, test_size=0.2
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
# Check train data After resampling
print(Counter(y_train))
```

Counter({0: 1221, 1: 855})

```python
# Define binary logistic regression model

model = LogisticRegression(solver="liblinear", random_state=0)
model.fit(X_train, y_train)

LogisticRegression(
    C=1.0,
    class_weight=None,
    dual=False,
    fit_intercept=True,
    intercept_scaling=1,
    l1_ratio=None,
    # max_iter=100,
    multi_class="warn",
    n_jobs=None,
    penalty="l2",
    random_state=0,
    solver="liblinear",
    tol=0.0001,
    verbose=0,
)
```

```python
y_pred = model.predict(X_test)
```

```python
# Check the coefficient values of features
model.coef_
```

array([[ 1.75954572, -0.35680141, -0.70304726,  0.48033979, -0.26778132,
        -0.2103076 ,  0.14023605, -0.21488527, -0.27081205,  0.18078408]])

```python
model.score(X_train, y_train)
```

0.9190751445086706

```python
# Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
```

array([[8932,  805],
        [  13,  201]])

```python
# Classification Report
print(classification_report(y_test, y_pred))
```

precision    recall  f1-score   support

0       1.00      0.92      0.96      9737
1       0.20      0.94      0.33       214

accuracy                           0.92      9951
macro avg       0.60      0.93      0.64      9951
weighted avg       0.98      0.92      0.94      9951

```python
# Plot Confusion Matrix (Binary absolute and relative with colorbar)
fig, ax = plot_confusion_matrix(
    confusion_matrix(y_test, y_pred),
    show_absolute=True,
    show_normed=True,
    colorbar=True,
)

ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=("Predicted 0s", "Predicted 1s"))
ax.yaxis.set(ticks=(0, 1), ticklabels=("Actual 0s", "Actual 1s"))
plt.title("Confusion_matrix")
plt.show()
```

![png](output_26_0.png)

## Feature Importance

```python
# insert coefficient values in a list
coef_lst = model.coef_[0].tolist()

# insert coefficient values in a df
df_coef = pd.DataFrame(columns=["feature", "coef_value"])
for i in range(len(features)):
    df_coef.at[i, "feature"] = features[i]
    df_coef.at[i, "coef_value"] = coef_lst[i]

df_coef
```

```python
# Sorting the dataframe of coefficient values in a descending order

final_sorted_df = df_coef.sort_values(by=["coef_value"], ascending=False)
final_sorted_df = final_sorted_df.reset_index(drop=True)
final_sorted_df
```

```python
fig = plt.figure(figsize=(5, 4))
ax = fig.add_axes([0, 0, 1, 1])
# ax.bar(features,values)
ax.bar(final_sorted_df["feature"], final_sorted_df["coef_value"])
np.rot90(plt.xticks(rotation=90, fontsize=10))
ax.set_title("Coefficients_Values of Binary Logistic Regression Model",
    fontsize=12)
plt.show()
```

![png](output_30_0.png)

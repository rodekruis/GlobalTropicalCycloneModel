# ROC CURVE for Binary Models

## (Binary Logistic Regression, Random Forest, and XGBoost)

We used the ROC curve to have a graphical representation of
the performance of the three binary classification models.
The curve can be interpreted as the tradeoff
between sensitivity and specificity.

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
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
bins2 = [0, 0.00009, 1]
samples_per_bin2, binsP2 = np.histogram(df["binary_damage"], bins=bins2)
```

```python
# Check the bins' intervalls (first bin means all zeros,
# second bin means 0 < values <= 1)
df["binary_damage"].value_counts(bins=binsP2)
```

(-0.001, 9e-05]    48685
(9e-05, 1.0]        1069
Name: binary_damage, dtype: int64

```python
print(samples_per_bin2)
print(binsP2)
```

[48685  1069]
[0.e+00 9.e-05 1.e+00]

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
# Logistic Regression
model_lr = LogisticRegression().fit(X_train, y_train)
probs_lr = model_lr.predict_proba(X_test)[:, 1]

# Random Forest
model_rf = RandomForestClassifier().fit(X_train, y_train)
probs_rf = model_rf.predict_proba(X_test)[:, 1]

# XGBoost
model_xg = XGBClassifier().fit(X_train, y_train)
probs_xg = model_xg.predict_proba(X_test)[:, 1]
```

```python
y_test_int = y_test.replace({"damaged": 1, "not_damaged": 0})

auc_lr = roc_auc_score(y_test_int, probs_lr)
fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test_int, probs_lr)

auc_rf = roc_auc_score(y_test_int, probs_rf)
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test_int, probs_rf)

auc_xg = roc_auc_score(y_test_int, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test_int, probs_xg)
```

```python
# Roc Curve plots for all the three binary models in one show
plt.figure(figsize=(7, 5))
plt.plot(fpr_lr, tpr_lr, label=f"AUC (Logistic Regression) = {auc_lr:.2f}")
plt.plot(fpr_rf, tpr_rf, label=f"AUC (Random Forests) = {auc_rf:.2f}")
plt.plot(fpr_xg, tpr_xg, label=f"AUC (XGBoost) = {auc_xg:.2f}")
plt.plot([0, 1], [0, 1], color="blue", linestyle="--", label="Baseline")
plt.title("ROC Curve", size=20)
plt.xlabel("False Positive Rate", size=12)
plt.ylabel("True Positive Rate", size=12)
plt.legend();
```

![png](output_22_0.png)

```python
y_test_int = y_test.replace({"damaged": 1, "not_damaged": 0})

auc_xg = roc_auc_score(y_test_int, probs_xg)
fpr_xg, tpr_xg, thresholds_xg = roc_curve(y_test_int, probs_xg)
```

```python
# ROC Curve plot only for XGBoost
plt.plot(fpr_xg, tpr_xg, label=f"AUC (XGBoost) = {auc_xg:.2f}")
plt.plot([0, 1], [0, 1], color="blue", linestyle="--", label="Baseline")
plt.title("ROC Curve", size=20)
plt.xlabel("False Positive Rate", size=10)
plt.ylabel("True Positive Rate", size=10)
plt.legend();
```

![png](output_24_0.png)

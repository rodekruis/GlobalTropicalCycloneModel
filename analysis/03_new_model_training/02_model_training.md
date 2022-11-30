# Model training

The input data is prepared by joining the calculated windfield with damaged values.
Subsampling is done by dropping those rows where the windspeed is 0, the the data stratification is done on damaged values.
The XGBoost Reduced Over fitting model, was trained on this prepared input data with gridcells.
The RMSE calculated in total and per each bin.

```python
%load_ext jupyter_black
```

```python
from utils import get_training_dataset
```

```python
# df = get_training_dataset()
# df
```

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
```

```python
# Read csv file and import to df

df = pd.read_csv("new_model_training_dataset-2.csv")
df
```

```python
# Show histogram of damage
df.hist(column="percent_buildings_damaged", figsize=(4, 3))
```

```python
# Hist plot after data stratification
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_buildings_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

```python
# Remove zeros from wind_speed
df = df[(df[["wind_speed"]] != 0).any(axis=1)]

df = df.drop(columns=["grid_point_id", "typhoon_year"])
df
```

```python
# Hist plot after removing rows where windspeed is 0
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_buildings_damaged"], bins=bins2)
plt.figure(figsize=(4, 3))
plt.xlabel("Damage Values")
plt.ylabel("Frequency")
plt.plot(binsP2[1:], samples_per_bin2)
```

```python
print(samples_per_bin2)
print(binsP2)
```

```python
# Check the bins' intervalls
# df["percent_buildings_damaged"].value_counts(bins=binsP2)
```

```python
bin_index2 = np.digitize(df["percent_buildings_damaged"], bins=binsP2)
```

```python
y_input_strat = bin_index2
```

```python
# Defin two lists to save total RMSE of test and train data
test_RMSE_list = []
train_RMSE_list = []

# Define empty lists for bins results(test and train)
test_RMSE_list_bin1 = []
test_RMSE_list_bin2 = []
test_RMSE_list_bin3 = []
test_RMSE_list_bin4 = []
test_RMSE_list_bin5 = []

train_RMSE_list_bin1 = []
train_RMSE_list_bin2 = []
train_RMSE_list_bin3 = []
train_RMSE_list_bin4 = []
train_RMSE_list_bin5 = []
```

```python
features = [
    "wind_speed",
    "total_buildings",
    # "total_buildings_damaged"
]

# Split X and y from dataframe features
X = df[features]
display(X.columns)
y = df["percent_buildings_damaged"]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
```

```python
# Run XGBoost Reduced Overfitting in for loop to estimate RMSE per bins

for i in range(20):
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, df["percent_buildings_damaged"], stratify=y_input_strat, test_size=0.2
    )

    # XGBoost Reduced Overfitting
    xgb = XGBRegressor(
        base_score=0.5,
        booster="gbtree",
        colsample_bylevel=0.8,
        colsample_bynode=0.8,
        colsample_bytree=0.8,
        gamma=3,
        eta=0.01,
        importance_type="gain",
        learning_rate=0.1,
        max_delta_step=0,
        max_depth=4,
        min_child_weight=1,
        missing=1,
        n_estimators=100,
        early_stopping_rounds=10,
        n_jobs=1,
        nthread=None,
        objective="reg:squarederror",
        reg_alpha=0,
        reg_lambda=1,
        scale_pos_weight=1,
        seed=None,
        silent=None,
        subsample=0.8,
        verbosity=1,
        eval_metric=["rmse", "logloss"],
        random_state=0,
    )

    eval_set = [(X_test, y_test)]
    xgb_model = xgb.fit(
        X_train,
        y_train,
        eval_set=eval_set,
        verbose=False,
        # sample_weight=pow(y_train, 2),
    )

    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2)
    est2 = est.fit()
    print(est2.summary())

    # Calculate RMSE in total

    y_pred_train = xgb.predict(X_train)
    mse_train_idx = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train_idx)

    y_pred = xgb.predict(X_test)
    mse_idx = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse_idx)

    print("----- Training ------")
    print(f"Root mean squared error: {rmse_train:.2f}")

    print("----- Test ------")
    print(f"Root mean squared error: {rmse:.2f}")

    test_RMSE_list.append(rmse)
    train_RMSE_list.append(rmse_train)

    # Calculate RMSE per bins

    bin_index_test = np.digitize(y_test, bins=binsP2)
    bin_index_train = np.digitize(y_train, bins=binsP2)

    # Estimation of RMSE for train data
    y_pred_train = xgb.predict(X_train)

    mse_train_idx1 = mean_squared_error(
        y_train[bin_index_train == 1], y_pred_train[bin_index_train == 1]
    )
    rmse_train_1 = np.sqrt(mse_train_idx1)
    mse_train_idx2 = mean_squared_error(
        y_train[bin_index_train == 2], y_pred_train[bin_index_train == 2]
    )
    rmse_train_2 = np.sqrt(mse_train_idx2)
    mse_train_idx3 = mean_squared_error(
        y_train[bin_index_train == 3], y_pred_train[bin_index_train == 3]
    )
    rmse_train_3 = np.sqrt(mse_train_idx3)
    mse_train_idx4 = mean_squared_error(
        y_train[bin_index_train == 4], y_pred_train[bin_index_train == 4]
    )
    rmse_train_4 = np.sqrt(mse_train_idx4)
    mse_train_idx5 = mean_squared_error(
        y_train[bin_index_train == 5], y_pred_train[bin_index_train == 5]
    )
    rmse_train_5 = np.sqrt(mse_train_idx5)

    # Estimation of RMSE for test data
    y_pred = xgb.predict(X_test)

    mse_idx1 = mean_squared_error(
        y_test[bin_index_test == 1], y_pred[bin_index_test == 1]
    )
    rmse_1 = np.sqrt(mse_idx1)
    mse_idx2 = mean_squared_error(
        y_test[bin_index_test == 2], y_pred[bin_index_test == 2]
    )
    rmse_2 = np.sqrt(mse_idx2)
    mse_idx3 = mean_squared_error(
        y_test[bin_index_test == 3], y_pred[bin_index_test == 3]
    )
    rmse_3 = np.sqrt(mse_idx3)
    mse_idx4 = mean_squared_error(
        y_test[bin_index_test == 4], y_pred[bin_index_test == 4]
    )
    rmse_4 = np.sqrt(mse_idx4)
    mse_idx5 = mean_squared_error(
        y_test[bin_index_test == 5], y_pred[bin_index_test == 5]
    )
    rmse_5 = np.sqrt(mse_idx5)

    train_RMSE_list_bin1.append(rmse_train_1)
    train_RMSE_list_bin2.append(rmse_train_2)
    train_RMSE_list_bin3.append(rmse_train_3)
    train_RMSE_list_bin4.append(rmse_train_4)
    train_RMSE_list_bin5.append(rmse_train_5)

    test_RMSE_list_bin1.append(rmse_1)
    test_RMSE_list_bin2.append(rmse_2)
    test_RMSE_list_bin3.append(rmse_3)
    test_RMSE_list_bin4.append(rmse_4)
    test_RMSE_list_bin5.append(rmse_5)
```

# Plot RMSE in total

```python
# RMSE in total

m_test_rmse = statistics.mean(test_RMSE_list)
plt.figure(figsize=(4, 3))
plt.axvline(m_test_rmse, color="red", linestyle="dashed")
plt.hist(
    test_RMSE_list,
    bins=np.arange(17.0, 19.0, 0.12),
    edgecolor="k",
    histtype="bar",
    density=True,
)
sd_test_rmse = statistics.stdev(test_RMSE_list)


m_train_rmse = statistics.mean(train_RMSE_list)
plt.axvline(m_train_rmse, color="b", linestyle="dashed")
plt.hist(
    train_RMSE_list,
    bins=np.arange(17.0, 19.0, 0.12),
    color="orange",
    edgecolor="k",
    histtype="bar",
    density=True,
    alpha=0.7,
)
sd_train_rmse = statistics.stdev(train_RMSE_list)

print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

print(f"mean_RMSE_test: {m_test_rmse:.2f}")
print(f"mean_RMSE_train: {m_train_rmse:.2f}")

# create legend
labels = ["Mean_test", "Mean_train", "test", "train"]
plt.legend(labels)

plt.xlabel("The RMSE error")
plt.ylabel("Frequency")
plt.title("histogram of the RMSE distribution")
plt.show()
```

# Plot RMSE per bins

```python
# RMSE of bins_1

m_test_rmse = statistics.mean(test_RMSE_list_bin1)
plt.figure(figsize=(4, 3))
plt.axvline(m_test_rmse, color="red", linestyle="dashed")
plt.hist(
    test_RMSE_list_bin1,
    bins=np.arange(6.0, 6.7, 0.05),
    edgecolor="k",
    histtype="bar",
    density=True,
)
sd_test_rmse = statistics.stdev(test_RMSE_list_bin1)


m_train_rmse = statistics.mean(train_RMSE_list_bin1)
plt.axvline(m_train_rmse, color="b", linestyle="dashed")
plt.hist(
    train_RMSE_list_bin1,
    bins=np.arange(6.0, 6.7, 0.05),
    color="orange",
    edgecolor="k",
    histtype="bar",
    density=True,
    alpha=0.7,
)
sd_train_rmse = statistics.stdev(train_RMSE_list_bin1)

print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

print(f"mean_RMSE_test: {m_test_rmse:.2f}")
print(f"mean_RMSE_train: {m_train_rmse:.2f}")

# create legend
labels = ["Mean_test", "Mean_train", "test", "train"]
plt.legend(labels)

plt.xlabel("The RMSE error")
plt.ylabel("Frequency")
plt.title("histogram of the RMSE distribution (bin1)")
plt.show()
```

```python
# RMSE of bins_2

m_test_rmse = statistics.mean(test_RMSE_list_bin2)
plt.figure(figsize=(4, 3))
plt.axvline(m_test_rmse, color="red", linestyle="dashed")
plt.hist(
    test_RMSE_list_bin2,
    bins=np.arange(10.0, 11.0, 0.06),
    edgecolor="k",
    histtype="bar",
    density=True,
)
sd_test_rmse = statistics.stdev(test_RMSE_list_bin2)


m_train_rmse = statistics.mean(train_RMSE_list_bin2)
plt.axvline(m_train_rmse, color="b", linestyle="dashed")
plt.hist(
    train_RMSE_list_bin2,
    bins=np.arange(10.0, 11.0, 0.06),
    color="orange",
    edgecolor="k",
    histtype="bar",
    density=True,
    alpha=0.7,
)
sd_train_rmse = statistics.stdev(train_RMSE_list_bin2)

print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

print(f"mean_RMSE_test: {m_test_rmse:.2f}")
print(f"mean_RMSE_train: {m_train_rmse:.2f}")

# create legend
labels = ["Mean_test", "Mean_train", "test", "train"]
plt.legend(labels)

plt.xlabel("The RMSE error")
plt.ylabel("Frequency")
plt.title("histogram of the RMSE distribution (bin2)")
plt.show()
```

```python
# RMSE of bins_3

m_test_rmse = statistics.mean(test_RMSE_list_bin3)
plt.figure(figsize=(4, 3))
plt.axvline(m_test_rmse, color="red", linestyle="dashed")
plt.hist(
    test_RMSE_list_bin3,
    bins=np.arange(13.0, 15.0, 0.13),
    edgecolor="k",
    histtype="bar",
    density=True,
)
sd_test_rmse = statistics.stdev(test_RMSE_list_bin3)


m_train_rmse = statistics.mean(train_RMSE_list_bin3)
plt.axvline(m_train_rmse, color="b", linestyle="dashed")
plt.hist(
    train_RMSE_list_bin3,
    bins=np.arange(13.0, 15.0, 0.13),
    color="orange",
    edgecolor="k",
    histtype="bar",
    density=True,
    alpha=0.7,
)
sd_train_rmse = statistics.stdev(train_RMSE_list_bin3)

print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

print(f"mean_RMSE_test: {m_test_rmse:.2f}")
print(f"mean_RMSE_train: {m_train_rmse:.2f}")

# create legend
labels = ["Mean_test", "Mean_train", "test", "train"]
plt.legend(labels)

plt.xlabel("The RMSE error")
plt.ylabel("Frequency")
plt.title("histogram of the RMSE distribution (bin3)")
plt.show()
```

```python
# RMSE of bins_4

m_test_rmse = statistics.mean(test_RMSE_list_bin4)
plt.figure(figsize=(4, 3))
plt.axvline(m_test_rmse, color="red", linestyle="dashed")
plt.hist(
    test_RMSE_list_bin4,
    bins=np.arange(15.5, 16.5, 0.07),
    edgecolor="k",
    histtype="bar",
    density=True,
)
sd_test_rmse = statistics.stdev(test_RMSE_list_bin4)


m_train_rmse = statistics.mean(train_RMSE_list_bin4)
plt.axvline(m_train_rmse, color="b", linestyle="dashed")
plt.hist(
    train_RMSE_list_bin4,
    bins=np.arange(15.5, 16.5, 0.07),
    color="orange",
    edgecolor="k",
    histtype="bar",
    density=True,
    alpha=0.7,
)
sd_train_rmse = statistics.stdev(train_RMSE_list_bin4)

print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

print(f"mean_RMSE_test: {m_test_rmse:.2f}")
print(f"mean_RMSE_train: {m_train_rmse:.2f}")

# create legend
labels = ["Mean_test", "Mean_train", "test", "train"]
plt.legend(labels)

plt.xlabel("The RMSE error")
plt.ylabel("Frequency")
plt.title("histogram of the RMSE distribution (bin4)")
plt.show()
```

```python
# RMSE of bins_5

m_test_rmse = statistics.mean(test_RMSE_list_bin5)
plt.figure(figsize=(4, 3))
plt.axvline(m_test_rmse, color="red", linestyle="dashed")
plt.hist(
    test_RMSE_list_bin5,
    bins=np.arange(67.0, 72.0, 0.3),
    edgecolor="k",
    histtype="bar",
    density=True,
)
sd_test_rmse = statistics.stdev(test_RMSE_list_bin5)


m_train_rmse = statistics.mean(train_RMSE_list_bin5)
plt.axvline(m_train_rmse, color="b", linestyle="dashed")
plt.hist(
    train_RMSE_list_bin5,
    bins=np.arange(67.0, 72.0, 0.3),
    color="orange",
    edgecolor="k",
    histtype="bar",
    density=True,
    alpha=0.7,
)
sd_train_rmse = statistics.stdev(train_RMSE_list_bin5)

print(f"stdev_RMSE_test: {sd_test_rmse:.2f}")
print(f"stdev_RMSE_train: {sd_train_rmse:.2f}")

print(f"mean_RMSE_test: {m_test_rmse:.2f}")
print(f"mean_RMSE_train: {m_train_rmse:.2f}")

# create legend
labels = ["Mean_test", "Mean_train", "test", "train"]
plt.legend(labels)

plt.xlabel("The RMSE error")
plt.ylabel("Frequency")
plt.title("histogram of the RMSE distribution (bin5)")
plt.show()
```

```python

```

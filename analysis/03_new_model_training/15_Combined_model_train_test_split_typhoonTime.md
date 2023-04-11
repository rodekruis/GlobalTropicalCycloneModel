# Combined  Model (XGBoost Undersampling + XGBoost Regression)

We developed a hybrid model using both xgboost regression and xgboost classification(while undersampling technique was implemented to enhance its performance). Subsequently, we evaluated the performance of this combined model on the test dataset while train_test_split is done based on typhoons' time and compared it with the result of the simple xgboost regression model. 


```python
%load_ext jupyter_black
```



<script type="application/javascript" id="jupyter_black">
(function() {
    if (window.IPython === undefined) {
        return
    }
    var msg = "WARNING: it looks like you might have loaded " +
        "jupyter_black in a non-lab notebook with " +
        "`is_lab=True`. Please double check, and if " +
        "loading with `%load_ext` please review the README!"
    console.log(msg)
    alert(msg)
})()
</script>




```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import shap
import imblearn
import statsmodels.api as sm
import statistics
import warnings

from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import RobustScaler
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from xgboost.sklearn import XGBRegressor
from sklearn.dummy import DummyRegressor
from xgboost import XGBClassifier
from sty import fg, rs

from sklearn.metrics import confusion_matrix
from matplotlib import cm
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

from utils import get_training_dataset
```

    pandas.Int64Index is deprecated and will be removed from pandas in a future version. Use pandas.Index with the appropriate dtype instead.



```python
# Hide all warnings
warnings.filterwarnings("ignore")
```


```python
# Read CSV file and import to a df
df = get_training_dataset()
```


```python
# Replace empty cells of RWI with mean value
df["rwi"].fillna(df["rwi"].mean(), inplace=True)

# Set any values >100% to 100%,
for r in range(len(df)):
    if df.loc[r, "percent_houses_damaged"] > 100:
        df.at[r, "percent_houses_damaged"] = float(100)
```


```python
df = (df[(df[["wind_speed"]] != 0).any(axis=1)]).reset_index(drop=True)
df = df.drop(columns=["grid_point_id", "typhoon_year"])
```


```python
# Specify features
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
```


```python
# List of typhoons
typhoons = [
    "DURIAN",
    "FENGSHEN",
    "KETSANA",
    "CONSON",
    "NESAT",
    "BOPHA",
    "NARI",
    "KROSA",
    "HAIYAN",
    "USAGI",
    "UTOR",
    "JANGMI",
    "KALMAEGI",
    "RAMMASUN",
    "HAGUPIT",
    "FUNG-WONG",
    "LINGLING",
    "MUJIGAE",
    "MELOR",
    "NOUL",
    "GONI",
    "LINFA",
    "KOPPU",
    "MEKKHALA",
    "HAIMA",
    "TOKAGE",
    "MERANTI",
    "NOCK-TEN",
    "SARIKA",
    "MANGKHUT",
    "YUTU",
    "KAMMURI",
    "NAKRI",
    "PHANFONE",
    "SAUDEL",
    "VAMCO",
    "VONGFONG",
    "MOLAVE",
]
```


```python
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
bins_eval = [0, 1, 10, 20, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"], bins=bins2)
```


```python
# Define range of for loop
# Latest typhoons in terms of time
tyohoons_for_test = [
    "MERANTI",
    "NOCK-TEN",
    "SARIKA",
    "MANGKHUT",
    "YUTU",
    "KAMMURI",
    "NAKRI",
    "PHANFONE",
    "SAUDEL",
    "VAMCO",
    "VONGFONG",
    "MOLAVE",
]
num_exp = len(tyohoons_for_test)

# Define number of bins
num_bins = len(bins_eval)
```


```python
# Define empty list to save RMSE in combined model

test_RMSE_lst = []
test_RMSE_bin = defaultdict(list)

# Define empty list to save RMSE in model1

test_RMSE_lst_M1 = []
test_RMSE_bin_M1 = defaultdict(list)
```


```python
for run_ix in range(num_exp):

    # Oldest typhoons in the training set
    typhoons_train_lst = typhoons[run_ix : run_ix + 29]

    bin_index2 = np.digitize(df["percent_houses_damaged"], bins=binsP2)
    y_input_strat = bin_index2

    # Split X and y from dataframe features
    X = df[features]
    y = df["percent_houses_damaged"]

    # Split df to train and test (one typhoon for test and the rest of typhoons for train)
    df_test = df[df["typhoon_name"] == tyohoons_for_test[run_ix]]

    df_train = pd.DataFrame()
    for run_ix_train in range(len(typhoons_train_lst)):
        df_train = df_train.append(
            df[df["typhoon_name"] == typhoons_train_lst[run_ix_train]]
        )

    # Split X and y from dataframe features
    X_test = df_test[features]
    X_train = df_train[features]

    y_train = df_train["percent_houses_damaged"]
    y_test = df_test["percent_houses_damaged"]

    print(df_test["typhoon_name"].unique())

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
        verbosity=0,
        eval_metric=["rmse", "logloss"],
        random_state=0,
    )

    eval_set = [(X_test, y_test)]
    xgb_model = xgb.fit(X_train, y_train, eval_set=eval_set, verbose=False)

    # Make prediction on train and test data
    y_pred_train = xgb.predict(X_train)
    y_pred = xgb.predict(X_test)

    # Calculate RMSE in total
    mse_train_idx = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train_idx)

    mse_idx = mean_squared_error(y_test, y_pred)
    rmseM1 = np.sqrt(mse_idx)

    # Add total RMSE of Model1 to the list
    test_RMSE_lst_M1.insert(run_ix, rmseM1)
    print("RMSE in total and per bin (M1 model)")
    print(f"total: {rmseM1:.2f}")

    # Calculate RMSE per bins
    bin_index_test = np.digitize(y_test, bins=bins_eval)
    RSME_test_model1 = np.zeros(num_bins - 1)

    for bin_num in range(1, num_bins):

        # Estimation of RMSE for test data
        if (
            len(y_test[bin_index_test == bin_num]) != 0
            and len(y_pred[bin_index_test == bin_num]) != 0
        ):
            mse_idx = mean_squared_error(
                y_test[bin_index_test == bin_num], y_pred[bin_index_test == bin_num]
            )
            RSME_test_model1[bin_num - 1] = np.sqrt(mse_idx)

            # Add RMSE of Model1 to the list of each bin
            test_RMSE_bin_M1[bin_num].append(RSME_test_model1[bin_num - 1])
            print(f"bin{[bin_num]}:{RSME_test_model1[bin_num-1]}")
    # else:
    #    test_RMSE_bin_M1[bin_num].insert(run_ix, "No exist")

    # Define a threshold to separate target into damaged and not_damaged
    thres = 10.0
    y_test_bool = y_test >= thres
    y_train_bool = y_train >= thres
    y_test_bin = (y_test_bool) * 1
    y_train_bin = (y_train_bool) * 1

    sum(y_train_bin)

    # Define undersampling strategy
    under = RandomUnderSampler(sampling_strategy=0.1)
    # Fit and apply the transform
    X_train_us, y_train_us = under.fit_resample(X_train, y_train_bin)

    # Use XGBClassifier as a Machine Learning model to fit the data
    xgb_model = XGBClassifier(eval_metric=["error", "logloss"])

    eval_set = [(X_test, y_test_bin)]
    xgb_model.fit(
        X_train_us,
        y_train_us,
        eval_set=eval_set,
        verbose=False,
    )

    # Make prediction on test data and print Confusion Matrix
    y_pred_test = xgb_model.predict(X_test)
    cm = confusion_matrix(y_test_bin, y_pred_test)

    # Make prediction on train data and print Confusion Matrix
    y_pred_train = xgb_model.predict(X_train)
    cm = confusion_matrix(y_train_bin, y_pred_train)

    reduced_df = X_train.copy()

    reduced_df["percent_houses_damaged"] = y_train.values
    reduced_df["predicted_value"] = y_pred_train

    fliterd_df = reduced_df[reduced_df.predicted_value == 1]

    ### Third step is to train XGBoost regression model for this reduced train data (including damg>10.0%)
    bin_index2 = np.digitize(fliterd_df["percent_houses_damaged"], bins=binsP2)
    y_input_strat = bin_index2

    # Split X and y from dataframe features
    X_r = fliterd_df[features]
    y_r = fliterd_df["percent_houses_damaged"]

    # XGBoost Reduced Overfitting
    xgbR = XGBRegressor(
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
        verbosity=0,
        eval_metric=["rmse", "logloss"],
        random_state=0,
    )

    eval_set = [(X_r, y_r)]
    xgbR_model = xgbR.fit(X_r, y_r, eval_set=eval_set, verbose=False)

    # Make prediction on train and global test data
    y_pred_r = xgbR.predict(X_r)
    y_pred_test_total = xgbR.predict(X_test)

    # Calculate RMSE in total

    mse_train_idxR = mean_squared_error(y_r, y_pred_r)
    rmse_trainR = np.sqrt(mse_train_idxR)

    mse_idxR = mean_squared_error(y_test, y_pred_test_total)
    rmseR = np.sqrt(mse_idxR)

    ## Last step is to add model combination (model M1 with model MR)
    # Check the result of classifier for test set
    reduced_test_df = X_test.copy()

    # joined X_test with countinous target and binary predicted values
    reduced_test_df["percent_houses_damaged"] = y_test.values
    reduced_test_df["predicted_value"] = y_pred_test

    # damaged prediction
    fliterd_test_df1 = reduced_test_df[reduced_test_df.predicted_value == 1]

    # not damaged prediction
    fliterd_test_df0 = reduced_test_df[reduced_test_df.predicted_value == 0]

    # Use X0 and X1 for the M1 and MR models' predictions
    X1 = fliterd_test_df1[features]
    X0 = fliterd_test_df0[features]

    # For the output equal to 1 apply MR to evaluate the performance
    y1_pred = xgbR.predict(X1)
    y1 = fliterd_test_df1["percent_houses_damaged"]

    # For the output equal to 0 apply M1 to evaluate the performance
    y0_pred = xgb.predict(X0)
    y0 = fliterd_test_df0["percent_houses_damaged"]

    fliterd_test_df0["predicted_percent_damage"] = y0_pred
    fliterd_test_df1["predicted_percent_damage"] = y1_pred

    # Join two dataframes together
    join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])

    # Calculate RMSE in total
    mse_combined_model = mean_squared_error(
        join_test_dfs["percent_houses_damaged"],
        join_test_dfs["predicted_percent_damage"],
    )

    rmse_combined_model = np.sqrt(mse_combined_model)
    test_RMSE_lst.insert(run_ix, rmse_combined_model)
    print("RMSE in total and per bin (Combined model)")
    print(f"total: {rmse_combined_model:.2f}")

    # Calculate RMSE per bin
    y_join = join_test_dfs["percent_houses_damaged"]
    y_pred_join = join_test_dfs["predicted_percent_damage"]

    bin_index_test = np.digitize(y_join, bins=bins_eval)
    RSME_combined_model = np.zeros(num_bins - 1)

    for bin_num in range(1, num_bins):
        if (
            len(y_join[bin_index_test == bin_num]) != 0
            and len(y_pred_join[bin_index_test == bin_num]) != 0
        ):
            mse_combined_model_bin = mean_squared_error(
                y_join[bin_index_test == bin_num],
                y_pred_join[bin_index_test == bin_num],
            )
            RSME_combined_model[bin_num - 1] = np.sqrt(mse_combined_model_bin)
            test_RMSE_bin[bin_num].append(RSME_combined_model[bin_num - 1])
            print(f"bin{[bin_num]}: {RSME_combined_model[bin_num - 1]}")

        # else:
        #   test_RMSE_bin[bin_num].insert(run_ix, "No exist")
```

    ['MERANTI']
    RMSE in total and per bin (M1 model)
    total: 3.51
    bin[1]:0.2584208213372432
    bin[2]:13.765338343947667
    bin[4]:9.418041509158368
    RMSE in total and per bin (Combined model)
    total: 3.01
    bin[1]: 0.2584208213372432
    bin[2]: 12.676755517750486
    bin[4]: 5.703384546150215
    ['NOCK-TEN']
    RMSE in total and per bin (M1 model)
    total: 3.22
    bin[1]:1.5572452722149688
    bin[2]:8.46199719727252
    bin[3]:8.69934029551616
    bin[4]:9.457486894559104
    bin[5]:22.382500390707676
    RMSE in total and per bin (Combined model)
    total: 2.83
    bin[1]: 1.5674872803897795
    bin[2]: 7.551371941664953
    bin[3]: 7.214958548289578
    bin[4]: 6.710698588550311
    bin[5]: 16.16566138328957
    ['SARIKA']
    RMSE in total and per bin (M1 model)
    total: 0.41
    bin[1]:0.4112726454809718
    RMSE in total and per bin (Combined model)
    total: 0.41
    bin[1]: 0.4112726454809718
    ['MANGKHUT']
    RMSE in total and per bin (M1 model)
    total: 2.73
    bin[1]:0.7691440934749746
    bin[2]:4.910144576796866
    bin[3]:5.901761350205051
    RMSE in total and per bin (Combined model)
    total: 2.82
    bin[1]: 0.9261380526421844
    bin[2]: 5.208437009160083
    bin[3]: 4.951898716788552
    ['YUTU']
    RMSE in total and per bin (M1 model)
    total: 0.57
    bin[1]:0.3830001802394938
    bin[2]:1.6763729335530966
    RMSE in total and per bin (Combined model)
    total: 0.70
    bin[1]: 0.5134494902257389
    bin[2]: 1.9483831982456257
    ['KAMMURI']
    RMSE in total and per bin (M1 model)
    total: 2.80
    bin[1]:1.3983647581468903
    bin[2]:2.9980570659075942
    bin[3]:8.617071470559416
    bin[4]:22.503810077504244
    RMSE in total and per bin (Combined model)
    total: 2.21
    bin[1]: 1.5302101213209016
    bin[2]: 3.4368593330919843
    bin[3]: 3.22176575303716
    bin[4]: 12.4823480542959
    ['NAKRI']
    RMSE in total and per bin (M1 model)
    total: 0.07
    bin[1]:0.0672568641861282
    RMSE in total and per bin (Combined model)
    total: 0.07
    bin[1]: 0.0672568641861282
    ['PHANFONE']
    RMSE in total and per bin (M1 model)
    total: 2.88
    bin[1]:0.6523290788712757
    bin[2]:2.655222023620572
    bin[3]:10.72813475579603
    bin[4]:20.478630422152598
    RMSE in total and per bin (Combined model)
    total: 1.60
    bin[1]: 0.7592312754620436
    bin[2]: 3.183079940564389
    bin[3]: 3.8671873184574417
    bin[4]: 9.222273660795235
    ['SAUDEL']
    RMSE in total and per bin (M1 model)
    total: 0.27
    bin[1]:0.2666832006317645
    RMSE in total and per bin (Combined model)
    total: 0.27
    bin[1]: 0.2666832006317645
    ['VAMCO']
    RMSE in total and per bin (M1 model)
    total: 1.10
    bin[1]:0.3248407961546835
    bin[2]:2.6239881788500696
    bin[3]:7.881335435497877
    RMSE in total and per bin (Combined model)
    total: 1.04
    bin[1]: 0.36005005607108875
    bin[2]: 2.98983803940602
    bin[3]: 1.193736734422409
    ['VONGFONG']
    RMSE in total and per bin (M1 model)
    total: 1.28
    bin[1]:0.335149791555512
    bin[2]:2.6264807684346536
    bin[3]:9.410221241921445
    bin[4]:15.249584280272789
    bin[5]:40.6698072539491
    RMSE in total and per bin (Combined model)
    total: 0.88
    bin[1]: 0.48179766014723596
    bin[2]: 2.696839299343334
    bin[3]: 3.7628482794636797
    bin[4]: 9.104537382016685
    bin[5]: 23.669994174115118
    ['MOLAVE']
    RMSE in total and per bin (M1 model)
    total: 0.51
    bin[1]:0.3589135292063722
    bin[2]:2.003225687379784
    RMSE in total and per bin (Combined model)
    total: 0.51
    bin[1]: 0.3589135292063722
    bin[2]: 2.003225687379784



```python
# Compare total RMSEs of Combined model with M1 model
combined_test_rmse = statistics.mean(test_RMSE_lst)
m1_test_rmse = statistics.mean(test_RMSE_lst_M1)

print(f"mean_RMSE_test_M1_model: {m1_test_rmse:.2f}")
print(f"mean_RMSE_test_Combined_model: {combined_test_rmse:.2f}")
```

    mean_RMSE_test_M1_model: 1.61
    mean_RMSE_test_Combined_model: 1.36



```python
# Compare RMSE per bin between Combined model with M1 model

for bin_num in range(1, 6):
    combined_test_rmse_bin = statistics.mean(test_RMSE_bin[bin_num])
    m1_test_rmse_bin = statistics.mean(test_RMSE_bin_M1[bin_num])

    print(f"RMSE per bin {bin_num}")
    print(f"mean_RMSE_test_Combined_model: {combined_test_rmse_bin:.2f}")
    print(f"mean_RMSE_test_M1_model: {m1_test_rmse_bin:.2f}\n")
```

    RMSE per bin 1
    mean_RMSE_test_Combined_model: 0.63
    mean_RMSE_test_M1_model: 0.57
    
    RMSE per bin 2
    mean_RMSE_test_Combined_model: 4.63
    mean_RMSE_test_M1_model: 4.64
    
    RMSE per bin 3
    mean_RMSE_test_Combined_model: 4.04
    mean_RMSE_test_M1_model: 8.54
    
    RMSE per bin 4
    mean_RMSE_test_Combined_model: 8.64
    mean_RMSE_test_M1_model: 15.42
    
    RMSE per bin 5
    mean_RMSE_test_Combined_model: 19.92
    mean_RMSE_test_M1_model: 31.53
    



```python
# Define a function to plot RMSEs
def rmse_bin_plot(M1_rmse, combined_rmse, min_rg, max_rg, step):

    m1_test_rmse = statistics.mean(M1_rmse)
    plt.figure(figsize=(4, 3))
    plt.axvline(m1_test_rmse, color="red", linestyle="dashed")
    plt.hist(
        M1_rmse,
        bins=np.arange(min_rg, max_rg, step),
        edgecolor="k",
        histtype="bar",
        density=True,
    )
    m1_sd_test_rmse = statistics.stdev(M1_rmse)

    combined_test_rmse = statistics.mean(combined_rmse)
    plt.axvline(combined_test_rmse, color="b", linestyle="dashed")
    plt.hist(
        combined_rmse,
        bins=np.arange(min_rg, max_rg, step),
        color="orange",
        edgecolor="k",
        histtype="bar",
        density=True,
        alpha=0.7,
    )
    combined_sd_test_rmse = statistics.stdev(combined_rmse)

    print(f"mean_RMSE_M1: {m1_test_rmse:.2f}(±{m1_sd_test_rmse:.2f})")
    print(f"mean_RMSE_Combined: {combined_test_rmse:.2f}(±{combined_sd_test_rmse:.2f})")

    # create legend
    labels = ["Mean_M1", "Mean_combined", "M1", "Combined"]
    plt.legend(labels)

    plt.xlabel("The RMSE error")
    plt.ylabel("Frequency")
    plt.title("histogram of the RMSE distribution")
    plt.show()
```


```python
print("RMSE and Stdev in total")
rmse_bin_plot(test_RMSE_lst_M1, test_RMSE_lst, 1.0, 2.0, 0.07)
```

    RMSE and Stdev in total
    mean_RMSE_M1: 1.61(±1.31)
    mean_RMSE_Combined: 1.36(±1.09)



    
![png](output_16_1.png)
    



```python
bin_params = {
    1: (0.5, 2.0, 0.09),
    2: (2.5, 5.5, 0.18),
    3: (4.0, 9.0, 0.27),
    4: (7.5, 17.5, 0.9),
    5: (19.5, 34.5, 1.2),
}


for bin_num in range(1, 6):

    print(f"RMSE and Stdev per bin {bin_num}")
    rmse_bin_plot(
        test_RMSE_bin_M1[bin_num], test_RMSE_bin[bin_num], *bin_params[bin_num]
    )
```

    RMSE and Stdev per bin 1
    mean_RMSE_M1: 0.57(±0.46)
    mean_RMSE_Combined: 0.63(±0.49)



    
![png](output_17_1.png)
    


    RMSE and Stdev per bin 2
    mean_RMSE_M1: 4.64(±4.01)
    mean_RMSE_Combined: 4.63(±3.49)



    
![png](output_17_3.png)
    


    RMSE and Stdev per bin 3
    mean_RMSE_M1: 8.54(±1.61)
    mean_RMSE_Combined: 4.04(±1.99)



    
![png](output_17_5.png)
    


    RMSE and Stdev per bin 4
    mean_RMSE_M1: 15.42(±6.07)
    mean_RMSE_Combined: 8.64(±2.63)



    
![png](output_17_7.png)
    


    RMSE and Stdev per bin 5
    mean_RMSE_M1: 31.53(±12.93)
    mean_RMSE_Combined: 19.92(±5.31)



    
![png](output_17_9.png)
    



```python

```

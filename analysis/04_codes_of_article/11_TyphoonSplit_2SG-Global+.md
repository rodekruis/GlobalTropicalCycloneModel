# Typhoon's time split

## We split based on typhoons' time, the training list includes the

## oldest 70% of typhoons

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
import statsmodels.api as sm
import statistics
import warnings

from math import sqrt
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
from collections import Counter
from sty import fg, rs
from itertools import chain

from sklearn.metrics import confusion_matrix
from matplotlib import cm
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from utils import get_training_dataset, weight_file
```

```python
# Hide all warnings
warnings.filterwarnings("ignore")
```

```python
# Read CSV file and import to a df
df = get_training_dataset()
```

```python
for i in range(len(df)):
    df.at[i, "typhoon_year"] = str(df.loc[i]["typhoon_year"])
```

```python
df["typhoon_name"] = df["typhoon_name"] + df["typhoon_year"]
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
df_data = df.drop(columns=["grid_point_id", "typhoon_year"])
```

```python
display(df.head())
display(df_data.head())
```

```python
# Read municipality dataset which already merged with y_norm converted ground truth
df_mun_merged = pd.read_csv("data/df_merged_2.csv")

# Remove the duplicated rows
df_mun_merged.drop_duplicates(keep="first", inplace=True)
df_mun_merged = df_mun_merged.reset_index(drop=True)

# Make the name of typhoons to uppercase
df_mun_merged["typhoon"] = df_mun_merged["typhoon"].str.upper()

# Rename y_norm column
df_mun_merged = df_mun_merged.rename(columns={"y_norm": "y_norm_mun"})

df_mun_merged
```

```python
# Function to transform name of typhoons to lowercase and remove begining of years
def transform_strings(strings):
    transformed_strings = []
    for string in strings:
        transformed_string = string[0].upper() + string[1:-4].lower() + string[-2:]
        transformed_strings.append(transformed_string)
    return transformed_strings
```

```python
# List of typhoons
typhoons = [
    "DURIAN2006",
    "FENGSHEN2008",
    "KETSANA2009",
    "CONSON2010",
    "NESAT2011",
    "BOPHA2012",
    "NARI2013",
    "KROSA2013",
    "HAIYAN2013",
    "USAGI2013",
    "UTOR2013",
    "JANGMI2014",
    "KALMAEGI2014",
    "RAMMASUN2014",
    "HAGUPIT2014",
    "FUNG-WONG2014",
    "LINGLING2014",
    "MUJIGAE2015",
    "MELOR2015",
    "NOUL2015",
    "GONI2015",
    "LINFA2015",
    "KOPPU2015",
    "MEKKHALA2015",
    "HAIMA2016",
    "TOKAGE2016",
    "MERANTI2016",
    "NOCK-TEN2016",
    "SARIKA2016",
    "MANGKHUT2018",
    "YUTU2018",
    "KAMMURI2019",
    "NAKRI2019",
    "PHANFONE2019",
    "SAUDEL2020",
    "GONI2020",
    "VAMCO2020",
    "VONGFONG2020",
    "MOLAVE2020",
]
```

```python
# Read the new weight CSV file and import to a df
df_weight = weight_file("/ggl_grid_to_mun_weights.csv")
df_weight.head()
```

```python
# Change name of column ['id'] to ['grid_point_id'] the same name as in input df
df_weight.rename(columns={"id": "grid_point_id"}, inplace=True)
df_weight.head()
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
    # "strong_roof_strong_wall",
    # "strong_roof_light_wall",
    # "strong_roof_salvage_wall",
    # "light_roof_strong_wall",
    # "light_roof_light_wall",
    # "light_roof_salvage_wall",
    # "salvaged_roof_strong_wall",
    # "salvaged_roof_light_wall",
    # "salvaged_roof_salvage_wall",
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
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df_data["percent_houses_damaged"], bins=bins2)
```

```python
# Define range of for loops
num_exp_main = 20

# Latest typhoons in terms of time
num_exp = 12
typhoons_for_test = typhoons[-num_exp:]

# LOOCV
#num_exp = len(typhoons)

# Define number of bins
num_bins = len(bins2)
```

```python
# Define empty list to save RMSE in each iteration
main_RMSE_lst = []
main_RMSE_bin = defaultdict(list)

main_AVE_lst = []
main_AVE_bin = defaultdict(list)
```

```python
for run_exm in range(num_exp_main):

    # Define empty list to save RMSE in combined model
    test_RMSE_lst = []
    test_RMSE_bin = defaultdict(list)

    # Define empty list to save RMSE in model1
    test_RMSE_lst_M1 = []
    test_RMSE_bin_M1 = defaultdict(list)

    # Defin two lists to save RMSE and Average Error
    RMSE = defaultdict(list)
    AVE = defaultdict(list)

    #for run_ix in range(27, num_exp):
    for run_ix in range(num_exp):

        # WITHOUT removing old typhoons from training set
        typhoons_train_lst = typhoons[0 : run_ix + 27]

        # WITH removing old typhoons from training set
        # typhoons_train_lst = typhoons[run_ix : run_ix + 27]

        # In each run Keep one typhoon for the test list while the
        # rest of the typhoons in the training set
        #typhoons_for_test = typhoons[run_ix]
        #typhoons_train_lst = typhoons[:run_ix] + typhoons[run_ix + 1 :]

        # Random split
        # typhoons_for_test = test_list[run_ix]
        # typhoons_train_lst = train_list

        # print(typhoons_train_lst)

        bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
        y_input_strat = bin_index2

        # Split X and y from dataframe features
        X = df_data[features]
        y = df_data["percent_houses_damaged"]

        # Split df to train and test (one typhoon for test and the rest of
        # typhoons for train)
        # For when we train over all typhoon this df_test is required
        #df_test = df_data[df_data["typhoon_name"] == typhoons_for_test]

        df_test = df[df["typhoon_name"] == typhoons_for_test[run_ix]]

        df_train = pd.DataFrame()
        for run_ix_train in range(len(typhoons_train_lst)):
            df_train = df_train.append(
                df_data[df_data["typhoon_name"] == typhoons_train_lst[run_ix_train]]
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

        eval_set = [(X_train, y_train)]
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
        # print("RMSE in total and per bin (M1 model)")
        # print(f"total: {rmseM1:.2f}")

        # Calculate RMSE per bins
        bin_index_test = np.digitize(y_test, bins=binsP2)
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
                # print(f"bin{[bin_num]}:{RSME_test_model1[bin_num-1]}")
        # else:
        #    test_RMSE_bin_M1[bin_num].insert(run_ix, "No exist")

        # Define a threshold to separate target into damaged and not_damaged
        thres = 10.0
        y_test_bool = y_test >= thres
        y_train_bool = y_train >= thres
        y_test_bin = (y_test_bool) * 1
        y_train_bin = (y_train_bool) * 1

        sum(y_train_bin)

        ## Define undersampling strategy
        under = RandomUnderSampler(sampling_strategy=0.1)

        # Fit and apply the transform
        X_train_us, y_train_us = under.fit_resample(X_train, y_train_bin)

        # Use XGBClassifier as a Machine Learning model to fit the data
        xgb_model = XGBClassifier(eval_metric=["error", "logloss"])

        eval_set = [(X_train, y_train_bin)]
        xgb_model.fit(
            X_train_us,
            y_train_us,
            eval_set=eval_set,
            verbose=False,
        )

        # Make prediction on test data and print Confusion Matrix
        y_pred_test = xgb_model.predict(X_test)
        cm_test = confusion_matrix(y_test_bin, y_pred_test)
        # print(cm_test)

        # Make prediction on train data and print Confusion Matrix
        y_pred_train = xgb_model.predict(X_train)
        cm_train = confusion_matrix(y_train_bin, y_pred_train)
        # print(cm_train)

        y_pred_train_us = xgb_model.predict(X_train_us)
        cm_train_us = confusion_matrix(y_train_us, y_pred_train_us)
        # print(cm_train_us)

        reduced_df = X_train.copy()

        reduced_df["percent_houses_damaged"] = y_train.values
        reduced_df["predicted_value"] = y_pred_train

        fliterd_df = reduced_df[reduced_df.predicted_value == 1]

        ### Third step is to train XGBoost regression model for this
        # reduced train data (including damg>10.0%)
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
        if len(y1_pred) > 0:
            fliterd_test_df1["predicted_percent_damage"] = y1_pred

        # Join two dataframes together
        join_test_dfs = pd.concat([fliterd_test_df0, fliterd_test_df1])

        y_join = join_test_dfs["percent_houses_damaged"]
        y_pred_join = join_test_dfs["predicted_percent_damage"]

        pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])

        pred_df["y_all"] = y_join
        pred_df["y_pred_all"] = y_pred_join

        # Filter damages greater than 10 to estimate RMSE for these values
        # pred_df["y_all"] = y_join[y_join > 10]
        # pred_df["y_pred_all"] = y_pred_join[y_join > 10]


        # Join data with y_all and y_all_pred
        df_data_w_pred = pd.merge(pred_df, df_data, left_index=True, right_index=True)
        # Join data with grid_point_id typhoon_year
        df_data_w_pred_grid = pd.merge(
            df[["grid_point_id", "typhoon_year"]],
            df_data_w_pred,
            left_index=True,
            right_index=True,
        )
        df_data_w_pred_grid.sort_values("y_pred_all", ascending=False)

        # join with weights df
        join_df = df_data_w_pred_grid.merge(df_weight, on="grid_point_id", how="left")

        # Indicate where values are valid and not missing
        join_df = join_df.loc[join_df["weight"].notna()]

        # Multiply weight by y_all and y_pred_all
        join_df["weight*y_pred*houses"] = (
            join_df["y_pred_all"] * join_df["weight"] * join_df["total_houses"]
                / 100
        )
        join_df["weight*y*houses"] = (
            join_df["y_all"] * join_df["weight"] * join_df["total_houses"] / 100
        )
        join_df["weight*houses"] = join_df["weight"] * join_df["total_houses"]

        join_df.sort_values("y_pred_all", ascending=False)

        # Groupby by municipality and typhoon_name with sum as the aggregation function
        agg_df = join_df.groupby(["ADM3_PCODE", "typhoon_name", "typhoon_year"]).agg(
            "sum"
        )

        # Normalize by the sum of the weights
        agg_df["y_pred_norm"] = (
            agg_df["weight*y_pred*houses"] / agg_df["weight*houses"] * 100
        )
        agg_df["y_norm"] = agg_df["weight*y*houses"] / agg_df["weight*houses"]
            * 100

        # Drop not required column y and y_pred before multiplying by weight
        agg_df.drop("y_all", axis=1, inplace=True)
        agg_df.drop("y_pred_all", axis=1, inplace=True)

        # Remove rows with NaN after normalization
        final_df = agg_df.dropna()
        final_df=final_df.reset_index()
        print(len(final_df))

        # Intersection of two datasets grid and municipality
        # Rename a column
        final_df = final_df.rename(columns={"ADM3_PCODE": "Mun_Code",
            "typhoon_name": "typhoon"})

        # Merge DataFrames based on 'typhoon_name' and 'Mun_Code'
        merged_df = pd.merge(final_df, df_mun_merged, on=["Mun_Code", "typhoon"],
            how="inner")
        print(len(merged_df))

        # Calculate RMSE & Average Error in total for converted grid_based model
        # to Mun_based
        if len(merged_df["y_norm"]) > 0:
            rmse = sqrt(mean_squared_error(merged_df["y_norm"], merged_df["y_pred_norm"]))
            ave = (merged_df["y_pred_norm"] - merged_df["y_norm"]).sum() / len(
                merged_df["y_norm"]
            )

            print(f"RMSE for grid_based model: {rmse:.2f}")
            print(f"Average Error for grid_based model: {ave:.2f}")

            RMSE["all"].append(rmse)
            AVE["all"].append(ave)

        bin_index = np.digitize(merged_df["y_norm"], bins=binsP2)

        for bin_num in range(1, 6):
            if len(merged_df["y_norm"][bin_index == bin_num]) > 0:

                mse_idx = mean_squared_error(
                    merged_df["y_norm"][bin_index == bin_num],
                    merged_df["y_pred_norm"][bin_index == bin_num],
                )
                rmse = np.sqrt(mse_idx)

                ave = (
                    merged_df["y_pred_norm"][bin_index == bin_num]
                    - merged_df["y_norm"][bin_index == bin_num]
                ).sum() / len(merged_df["y_norm"][bin_index == bin_num])

                RMSE[bin_num].append(rmse)
                AVE[bin_num].append(ave)
    print(RMSE["all"])
    # Save total RMSE in each iteration

    main_RMSE_lst.append(RMSE["all"])
    main_AVE_lst.append(AVE["all"])

    for bin_num in range(1, 6):

        # Save RMSE per bin in each iteration
        main_RMSE_bin[bin_num].append(RMSE[bin_num])
        main_AVE_bin[bin_num].append(AVE[bin_num])
```

['NOCK-TEN2016']
801
432
RMSE for grid_based model: 5.94
Average Error for grid_based model: 1.46
['SARIKA2016']
830
66
RMSE for grid_based model: 0.47
Average Error for grid_based model: -0.14
['MANGKHUT2018']
345
280
RMSE for grid_based model: 4.64
Average Error for grid_based model: 1.41
['YUTU2018']
562
173
RMSE for grid_based model: 0.77
Average Error for grid_based model: -0.05
['KAMMURI2019']
821
328
RMSE for grid_based model: 4.55
Average Error for grid_based model: 0.42
['NAKRI2019']
6
2
RMSE for grid_based model: 0.02
Average Error for grid_based model: 0.02
['PHANFONE2019']
922
218
RMSE for grid_based model: 4.51
Average Error for grid_based model: -2.02
['SAUDEL2020']
711
2
RMSE for grid_based model: 0.17
Average Error for grid_based model: 0.17
['GONI2020']
826
229
RMSE for grid_based model: 2.86
Average Error for grid_based model: -0.61
['VAMCO2020']
751
296
RMSE for grid_based model: 1.38
Average Error for grid_based model: -0.28
['VONGFONG2020']
1172
313
RMSE for grid_based model: 4.00
Average Error for grid_based model: -0.40
['MOLAVE2020']
737
125
RMSE for grid_based model: 1.22
Average Error for grid_based model: 0.44
[5.9434499223548185, 0.46952847372691847, 4.638136248973591,
0.7730202256359381, 4.548609216541342, 0.023273978195560485,
4.510291416370362, 0.17413787423671478, 2.8644752480357645,
1.3793069634013775, 3.9957745550200165, 1.2213774750867892]
['NOCK-TEN2016']

```python
# Estimate total RMSE

rmse = statistics.mean(list(chain.from_iterable(main_RMSE_lst)))
print(f"RMSE: {rmse:.2f}")

sd_rmse = statistics.stdev(list(chain.from_iterable(main_RMSE_lst)))
print(f"stdev: {sd_rmse:.2f}")

ave = statistics.mean(list(chain.from_iterable(main_AVE_lst)))
print(f"Average Error: {ave:.2f}")

sd_ave = statistics.stdev(list(chain.from_iterable(main_AVE_lst)))
print(f"Stdev of Average Error: {sd_ave:.2f}")
```

RMSE: 2.55
stdev: 1.98
Average Error: 0.03
Stdev of Average Error: 0.87

```python
# Estimate RMSE per bin

for bin_num in range(1, 6):

    rmse_bin = statistics.mean(list(chain.from_iterable(main_RMSE_bin[bin_num])))
    sd_rmse_bin = statistics.stdev(list(chain.from_iterable(main_RMSE_bin[bin_num])))
    ave_bin = statistics.mean(list(chain.from_iterable(main_AVE_bin[bin_num])))
    sd_ave_bin = statistics.stdev(list(chain.from_iterable(main_AVE_bin[bin_num])))

    print(f"\nRMSE & STDEV & Average Error per bin {bin_num}")

    print(f"RMSE: {rmse_bin:.2f}")
    print(f"STDEV: {sd_rmse_bin:.2f}")
    print(f"Average_Error: {ave_bin:.2f}")
    print(f"Stdev of Average_Error: {sd_ave_bin:.2f}")
```

RMSE & STDEV & Average Error per bin 1
RMSE: 0.10
STDEV: 0.07
Average_Error: 0.02
Stdev of Average_Error: 0.05

RMSE & STDEV & Average Error per bin 2
RMSE: 1.32
STDEV: 1.84
Average_Error: 0.51
Stdev of Average_Error: 1.03

RMSE & STDEV & Average Error per bin 3
RMSE: 4.82
STDEV: 3.42
Average_Error: 0.69
Stdev of Average_Error: 3.17

RMSE & STDEV & Average Error per bin 4
RMSE: 12.89
STDEV: 3.08
Average_Error: -2.89
Stdev of Average_Error: 9.83

RMSE & STDEV & Average Error per bin 5
RMSE: 31.05
STDEV: 21.92
Average_Error: -31.05
Stdev of Average_Error: 21.92

```python

```

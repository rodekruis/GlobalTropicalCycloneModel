# Combined  Model (XGBoost Undersampling + XGBoost Regression)

We developed a hybrid model using both xgboost regression and
xgboost classification(while undersampling technique was implemented
to enhance its performance). Subsequently, we evaluated the performance
 of this combined model on the test dataset while train_test_split is
 done based on different typhoons and compared it with the result of the
 simple xgboost regression model.

## The whole code is in a loop with the length of number of typhoons to

## estimate the average of RMSE in total

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
samples_per_bin2, binsP2 = np.histogram(df["percent_houses_damaged"],
    bins=bins2)
```

```python
# Define range of for loop
num_exp = len(typhoons)

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
for run_ix in range(len(typhoons)):

    bin_index2 = np.digitize(df["percent_houses_damaged"], bins=binsP2)
    y_input_strat = bin_index2

    # Split X and y from dataframe features
    X = df[features]
    y = df["percent_houses_damaged"]

    # Split df to train and test (one typhoon for test and the rest of
    # typhoons for train)
    df_test = df[df["typhoon_name"] == typhoons[run_ix]]
    df_train = df[df["typhoon_name"] != typhoons[run_ix]]

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
                y_test[bin_index_test == bin_num],
                y_pred[bin_index_test == bin_num]
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

['DURIAN']
RMSE in total and per bin (M1 model)
total: 6.42
bin[1]:2.7555309052548185
bin[2]:10.687236820349543
bin[3]:10.675959475002982
bin[4]:19.47978752387769
bin[5]:38.50531886707366
RMSE in total and per bin (Combined model)
total: 6.74
bin[1]: 2.970590870733237
bin[2]: 12.129689888333417
bin[3]: 13.115864834068649
bin[4]: 19.534511202137615
bin[5]: 35.30194780327537
['FENGSHEN']
RMSE in total and per bin (M1 model)
total: 4.32
bin[1]:1.3009252479796967
bin[2]:4.808301305798591
bin[3]:8.405258583345162
bin[4]:17.998772971635603
bin[5]:88.26337237706686
RMSE in total and per bin (Combined model)
total: 4.67
bin[1]: 2.085114904391558
bin[2]: 6.192520176266688
bin[3]: 8.092139804539427
bin[4]: 17.83636772119137
bin[5]: 86.87423088578325
['KETSANA']
RMSE in total and per bin (M1 model)
total: 1.79
bin[1]:0.3559350996404284
bin[2]:2.5383638524532763
bin[3]:11.550493749703222
bin[5]:57.26652545106064
RMSE in total and per bin (Combined model)
total: 1.79
bin[1]: 0.3559350996404284
bin[2]: 2.5383638524532763
bin[3]: 11.550493749703222
bin[5]: 57.26652545106064
['CONSON']
RMSE in total and per bin (M1 model)
total: 0.53
bin[1]:0.5224157577621258
bin[2]:1.3065582455332485
RMSE in total and per bin (Combined model)
total: 1.55
bin[1]: 1.550260871351224
bin[2]: 1.3065582455332485
['NESAT']
RMSE in total and per bin (M1 model)
total: 0.99
bin[1]:0.8927562010059683
bin[2]:2.0010700877315366
RMSE in total and per bin (Combined model)
total: 1.38
bin[1]: 1.0767431918086978
bin[2]: 3.7035960816082802
['BOPHA']
RMSE in total and per bin (M1 model)
total: 7.36
bin[1]:1.3769067629131395
bin[2]:3.8093895170416454
bin[3]:9.796020696369512
bin[4]:27.836825031500346
bin[5]:55.86464882543052
RMSE in total and per bin (Combined model)
total: 7.25
bin[1]: 1.8192547290193708
bin[2]: 4.144846798553565
bin[3]: 9.24422293064059
bin[4]: 26.524138083976215
bin[5]: 54.45589525211374
['NARI']
RMSE in total and per bin (M1 model)
total: 1.26
bin[1]:1.2068470842263668
bin[2]:2.536235505644099
bin[3]:9.26546968130164
RMSE in total and per bin (Combined model)
total: 1.58
bin[1]: 1.5379380603224495
bin[2]: 2.536235505644099
bin[3]: 9.26546968130164
['KROSA']
RMSE in total and per bin (M1 model)
total: 1.35
bin[1]:1.0108819861164995
bin[2]:3.543792122615195
RMSE in total and per bin (Combined model)
total: 2.02
bin[1]: 1.4736110836556768
bin[2]: 5.506472351335019
['HAIYAN']
RMSE in total and per bin (M1 model)
total: 12.29
bin[1]:2.9433914731048816
bin[2]:4.973433296765921
bin[3]:11.483977702263703
bin[4]:22.27992372621274
bin[5]:44.783330137572825
RMSE in total and per bin (Combined model)
total: 11.85
bin[1]: 3.2796159192057246
bin[2]: 5.728490585835882
bin[3]: 9.507238690006632
bin[4]: 21.671860371111922
bin[5]: 42.44473260367167
['USAGI']
RMSE in total and per bin (M1 model)
total: 3.07
bin[1]:0.6593863146149653
bin[2]:7.46704301413372
RMSE in total and per bin (Combined model)
total: 3.36
bin[1]: 0.6593863146149653
bin[2]: 8.200598188678743
['UTOR']
RMSE in total and per bin (M1 model)
total: 1.98
bin[1]:1.563581049978049
bin[2]:6.555694140480516
bin[3]:4.79647090097904
bin[4]:9.834727454482689
RMSE in total and per bin (Combined model)
total: 2.61
bin[1]: 2.092956498470969
bin[2]: 9.14531695069401
bin[3]: 7.161074132998829
bin[4]: 8.343963226369453
['JANGMI']
RMSE in total and per bin (M1 model)
total: 0.44
bin[1]:0.43665328866468817
bin[2]:0.3022261030332076
RMSE in total and per bin (Combined model)
total: 0.44
bin[1]: 0.43665328866468817
bin[2]: 0.3022261030332076
['KALMAEGI']
RMSE in total and per bin (M1 model)
total: 0.30
bin[1]:0.29598669033671277
RMSE in total and per bin (Combined model)
total: 0.30
bin[1]: 0.29598669033671277
['RAMMASUN']
RMSE in total and per bin (M1 model)
total: 2.73
bin[1]:1.422642641928698
bin[2]:3.406671364657686
bin[3]:7.399696357359305
bin[4]:24.32486715658272
RMSE in total and per bin (Combined model)
total: 3.39
bin[1]: 2.1391720848994735
bin[2]: 4.947631414345778
bin[3]: 8.593745981092422
bin[4]: 23.363370440665047
['HAGUPIT']
RMSE in total and per bin (M1 model)
total: 3.53
bin[1]:0.5692660978603241
bin[2]:3.5178389962285217
bin[3]:11.31625930537002
bin[4]:23.213185749990956
RMSE in total and per bin (Combined model)
total: 3.25
bin[1]: 1.2300345586837953
bin[2]: 4.897725816219348
bin[3]: 9.462041410224007
bin[4]: 18.971632548959377
['FUNG-WONG']
RMSE in total and per bin (M1 model)
total: 0.69
bin[1]:0.31325303908511654
bin[2]:3.001597443910638
bin[3]:11.45143763432142
RMSE in total and per bin (Combined model)
total: 0.69
bin[1]: 0.31325303908511654
bin[2]: 3.001597443910638
bin[3]: 11.45143763432142
['LINGLING']
RMSE in total and per bin (M1 model)
total: 0.92
bin[1]:0.27583423875462926
bin[2]:3.6123084676006436
bin[3]:10.952357840484215
RMSE in total and per bin (Combined model)
total: 0.92
bin[1]: 0.27583423875462926
bin[2]: 3.6123084676006436
bin[3]: 10.952357840484215
['MUJIGAE']
RMSE in total and per bin (M1 model)
total: 0.19
bin[1]:0.1949605562070014
RMSE in total and per bin (Combined model)
total: 0.19
bin[1]: 0.1949605562070014
['MELOR']
RMSE in total and per bin (M1 model)
total: 6.12
bin[1]:1.254288571416619
bin[2]:6.6565729814580195
bin[3]:8.035241690168128
bin[4]:24.301272887708855
bin[5]:52.53950422155242
RMSE in total and per bin (Combined model)
total: 5.78
bin[1]: 1.7319401177112923
bin[2]: 7.930439565154086
bin[3]: 6.158453005128111
bin[4]: 19.58480850439755
bin[5]: 53.51792489314388
['NOUL']
RMSE in total and per bin (M1 model)
total: 2.96
bin[1]:2.955244092100129
RMSE in total and per bin (Combined model)
total: 3.39
bin[1]: 3.388459431034357
['GONI']
RMSE in total and per bin (M1 model)
total: 2.48
bin[1]:1.4000456309808338
bin[2]:4.787295128760476
bin[3]:10.719694208903999
bin[4]:17.076455423484614
RMSE in total and per bin (Combined model)
total: 2.48
bin[1]: 1.3894450843639998
bin[2]: 5.411326401258921
bin[3]: 10.940766481299319
bin[4]: 14.30680147206411
['LINFA']
RMSE in total and per bin (M1 model)
total: 0.32
bin[1]:0.31824953888985646
RMSE in total and per bin (Combined model)
total: 0.32
bin[1]: 0.31824953888985646
['KOPPU']
RMSE in total and per bin (M1 model)
total: 2.59
bin[1]:1.7630080692958645
bin[2]:3.632373032205826
bin[3]:11.645100169350028
bin[4]:21.04681261445919
RMSE in total and per bin (Combined model)
total: 2.71
bin[1]: 1.7908202759312841
bin[2]: 4.557901890763043
bin[3]: 11.531889893502559
bin[4]: 21.04681261445919
['MEKKHALA']
RMSE in total and per bin (M1 model)
total: 3.75
bin[1]:1.3124830552918259
bin[2]:6.22561301993063
bin[3]:12.346311502026555
bin[4]:29.166005808602655
bin[5]:70.0305318917253
RMSE in total and per bin (Combined model)
total: 3.97
bin[1]: 1.9237539322405048
bin[2]: 6.22561301993063
bin[3]: 12.346311502026555
bin[4]: 28.651205617833288
bin[5]: 70.0305318917253
['HAIMA']
RMSE in total and per bin (M1 model)
total: 4.31
bin[1]:0.905332010406325
bin[2]:2.8004406431209063
bin[3]:12.066332856990297
bin[4]:31.338508935599677
RMSE in total and per bin (Combined model)
total: 4.36
bin[1]: 1.1874664586005974
bin[2]: 2.9752338173097046
bin[3]: 11.758739653898951
bin[4]: 31.338508935599677
['TOKAGE']
RMSE in total and per bin (M1 model)
total: 0.66
bin[1]:0.6570247319468342
RMSE in total and per bin (Combined model)
total: 0.66
bin[1]: 0.6570247319468342
['MERANTI']
RMSE in total and per bin (M1 model)
total: 4.54
bin[1]:0.14336361688855698
bin[2]:17.58335309451672
bin[4]:12.815788921692782
RMSE in total and per bin (Combined model)
total: 6.60
bin[1]: 0.14336361688855698
bin[2]: 23.477083519670014
bin[4]: 22.392713761508883
['NOCK-TEN']
RMSE in total and per bin (M1 model)
total: 5.12
bin[1]:2.3098872247934668
bin[2]:13.49197170812092
bin[3]:16.29788361966499
bin[4]:13.35080147002792
bin[5]:17.458055975208392
RMSE in total and per bin (Combined model)
total: 5.71
bin[1]: 3.0730303691990706
bin[2]: 14.98119080551286
bin[3]: 16.884206116000563
bin[4]: 11.53298270479596
bin[5]: 18.446077573046036
['SARIKA']
RMSE in total and per bin (M1 model)
total: 0.58
bin[1]:0.5788402284812953
RMSE in total and per bin (Combined model)
total: 0.71
bin[1]: 0.7051917577896595
['MANGKHUT']
RMSE in total and per bin (M1 model)
total: 6.18
bin[1]:0.8981317071758602
bin[2]:10.4446478847608
bin[3]:17.723775737295313
RMSE in total and per bin (Combined model)
total: 7.72
bin[1]: 1.6009442361022697
bin[2]: 13.319101096350199
bin[3]: 20.490048281474415
['YUTU']
RMSE in total and per bin (M1 model)
total: 0.52
bin[1]:0.4170516515109994
bin[2]:1.2808360444833093
RMSE in total and per bin (Combined model)
total: 1.20
bin[1]: 0.8360725097332669
bin[2]: 3.4864033503507827
['KAMMURI']
RMSE in total and per bin (M1 model)
total: 3.21
bin[1]:1.6703517382154325
bin[2]:3.9992476130723813
bin[3]:9.156166468035282
bin[4]:23.520264891591584
RMSE in total and per bin (Combined model)
total: 4.01
bin[1]: 2.4242489318295837
bin[2]: 6.037636670800101
bin[3]: 8.939413379248792
bin[4]: 20.567206718320076
['NAKRI']
RMSE in total and per bin (M1 model)
total: 0.04
bin[1]:0.04029471936999575
RMSE in total and per bin (Combined model)
total: 0.04
bin[1]: 0.04029471936999575
['PHANFONE']
RMSE in total and per bin (M1 model)
total: 3.56
bin[1]:0.5616522913610306
bin[2]:4.924413076772502
bin[3]:14.064913346372805
bin[4]:23.079913967400348
RMSE in total and per bin (Combined model)
total: 3.54
bin[1]: 0.8511571494379467
bin[2]: 5.168342717766189
bin[3]: 14.049464298339927
bin[4]: 21.86681322344152
['SAUDEL']
RMSE in total and per bin (M1 model)
total: 0.27
bin[1]:0.26937735748452235
RMSE in total and per bin (Combined model)
total: 0.27
bin[1]: 0.26937735748452235
['VAMCO']
RMSE in total and per bin (M1 model)
total: 1.20
bin[1]:0.32626990931830097
bin[2]:2.82384532729526
bin[3]:8.992751908217574
RMSE in total and per bin (Combined model)
total: 1.23
bin[1]: 0.32626990931830097
bin[2]: 2.9463081045836668
bin[3]: 8.992751908217574
['VONGFONG']
RMSE in total and per bin (M1 model)
total: 1.81
bin[1]:0.3946449775281797
bin[2]:2.2262606885801017
bin[3]:12.00865005200716
bin[4]:22.609718139796954
bin[5]:59.78735810386365
RMSE in total and per bin (Combined model)
total: 1.60
bin[1]: 0.3946449775281797
bin[2]: 5.730998075020828
bin[3]: 12.00865005200716
bin[4]: 19.127578629714915
bin[5]: 45.20627003776258
['MOLAVE']
RMSE in total and per bin (M1 model)
total: 0.73
bin[1]:0.652054652504046
bin[2]:1.863067732886607
RMSE in total and per bin (Combined model)
total: 0.73
bin[1]: 0.652054652504046
bin[2]: 1.863067732886607

```python
# test_RMSE_bin[5] = [value for value in test_RMSE_bin[5]
# if isinstance(value, int)]
```

```python
# Compare total RMSEs of Combined model with M1 model
combined_test_rmse = statistics.mean(test_RMSE_lst)
m1_test_rmse = statistics.mean(test_RMSE_lst_M1)

print(f"mean_RMSE_test_M1_model: {m1_test_rmse:.2f}")
print(f"mean_RMSE_test_Combined_model: {combined_test_rmse:.2f}")
```

mean_RMSE_test_M1_model: 2.66
mean_RMSE_test_Combined_model: 2.92

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
mean_RMSE_test_Combined_model: 1.25
mean_RMSE_test_M1_model: 0.97

RMSE per bin 2
mean_RMSE_test_Combined_model: 6.07
mean_RMSE_test_M1_model: 4.89

RMSE per bin 3
mean_RMSE_test_Combined_model: 11.02
mean_RMSE_test_M1_model: 10.92

RMSE per bin 4
mean_RMSE_test_Combined_model: 20.39
mean_RMSE_test_M1_model: 21.37

RMSE per bin 5
mean_RMSE_test_Combined_model: 51.50
mean_RMSE_test_M1_model: 53.83

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
    print(f"mean_RMSE_Combined: {combined_test_rmse:.2f}
        (±{combined_sd_test_rmse:.2f})")

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
rmse_bin_plot(test_RMSE_lst_M1, test_RMSE_lst, 2.0, 3.5, 0.09)
```

RMSE and Stdev in total
mean_RMSE_M1: 2.66(±2.57)
mean_RMSE_Combined: 2.92(±2.63)

![png](output_17_1.png)

```python
bin_params = {
    1: (0.5, 2.0, 0.09),
    2: (4.5, 6.5, 0.12),
    3: (9.5, 11.5, 0.1),
    4: (18.0, 22.0, 0.23),
    5: (35.0, 55.0, 1.2),
}


for bin_num in range(1, 6):

    print(f"RMSE and Stdev per bin {bin_num}")
    rmse_bin_plot(
        test_RMSE_bin_M1[bin_num], test_RMSE_bin[bin_num], *bin_params[bin_num]
    )
```

RMSE and Stdev per bin 1
mean_RMSE_M1: 0.97(±0.78)
mean_RMSE_Combined: 1.25(±0.95)

![png](output_18_1.png)

RMSE and Stdev per bin 2
mean_RMSE_M1: 4.89(±3.80)
mean_RMSE_Combined: 6.07(±4.72)

![png](output_18_3.png)

RMSE and Stdev per bin 3
mean_RMSE_M1: 10.92(±2.82)
mean_RMSE_Combined: 11.02(±3.19)

![png](output_18_5.png)

RMSE and Stdev per bin 4
mean_RMSE_M1: 21.37(±5.79)
mean_RMSE_Combined: 20.39(±5.67)

![png](output_18_7.png)

RMSE and Stdev per bin 5
mean_RMSE_M1: 53.83(±19.81)
mean_RMSE_Combined: 51.50(±19.74)

![png](output_18_9.png)

```python

```

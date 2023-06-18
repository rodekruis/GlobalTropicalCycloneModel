# Typhoon's time split

#### We split based on typhoons' time, the training list includes the oldest 70% of typhoons.


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
```


```python
# Hide all warnings
warnings.filterwarnings("ignore")
```


```python
# Read CSV file and import to a df
df = pd.read_csv("data/df_merged_2.csv")
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Mun_Code</th>
      <th>typhoon</th>
      <th>HAZ_rainfall_Total</th>
      <th>HAZ_rainfall_max_6h</th>
      <th>HAZ_rainfall_max_24h</th>
      <th>HAZ_v_max</th>
      <th>HAZ_dis_track_min</th>
      <th>GEN_landslide_per</th>
      <th>GEN_stormsurge_per</th>
      <th>GEN_Bu_p_inSSA</th>
      <th>...</th>
      <th>VUL_LightRoof_LightWall</th>
      <th>VUL_LightRoof_SalvageWall</th>
      <th>VUL_SalvagedRoof_StrongWall</th>
      <th>VUL_SalvagedRoof_LightWall</th>
      <th>VUL_SalvagedRoof_SalvageWall</th>
      <th>VUL_vulnerable_groups</th>
      <th>VUL_pantawid_pamilya_beneficiary</th>
      <th>DAM_perc_dmg</th>
      <th>HAZ_v_max_3</th>
      <th>y_norm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH175101000</td>
      <td>DURIAN2006</td>
      <td>185.828571</td>
      <td>14.716071</td>
      <td>7.381696</td>
      <td>55.032241</td>
      <td>2.478142</td>
      <td>2.64</td>
      <td>6.18</td>
      <td>6.18</td>
      <td>...</td>
      <td>41.892832</td>
      <td>1.002088</td>
      <td>0.000000</td>
      <td>0.027836</td>
      <td>0.083507</td>
      <td>2.951511</td>
      <td>46.931106</td>
      <td>3.632568</td>
      <td>166667.757548</td>
      <td>3.34975</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH083701000</td>
      <td>DURIAN2006</td>
      <td>8.818750</td>
      <td>0.455208</td>
      <td>0.255319</td>
      <td>8.728380</td>
      <td>288.358553</td>
      <td>0.06</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>13.645253</td>
      <td>0.549120</td>
      <td>0.030089</td>
      <td>0.090266</td>
      <td>0.112833</td>
      <td>3.338873</td>
      <td>25.989168</td>
      <td>0.000000</td>
      <td>664.968323</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH015501000</td>
      <td>DURIAN2006</td>
      <td>24.175000</td>
      <td>2.408333</td>
      <td>0.957639</td>
      <td>10.945624</td>
      <td>274.953818</td>
      <td>1.52</td>
      <td>1.28</td>
      <td>1.28</td>
      <td>...</td>
      <td>15.592295</td>
      <td>0.075838</td>
      <td>0.000000</td>
      <td>0.015168</td>
      <td>0.075838</td>
      <td>2.131755</td>
      <td>32.185651</td>
      <td>0.000000</td>
      <td>1311.358762</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH015502000</td>
      <td>DURIAN2006</td>
      <td>14.930000</td>
      <td>1.650000</td>
      <td>0.586250</td>
      <td>12.108701</td>
      <td>252.828578</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>7.100454</td>
      <td>0.023280</td>
      <td>0.011640</td>
      <td>0.000000</td>
      <td>0.128041</td>
      <td>1.589369</td>
      <td>29.612385</td>
      <td>0.000000</td>
      <td>1775.385328</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH175302000</td>
      <td>DURIAN2006</td>
      <td>13.550000</td>
      <td>1.054167</td>
      <td>0.528125</td>
      <td>10.660943</td>
      <td>258.194381</td>
      <td>5.52</td>
      <td>0.36</td>
      <td>0.36</td>
      <td>...</td>
      <td>30.354796</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.032852</td>
      <td>0.000000</td>
      <td>1.387007</td>
      <td>35.052562</td>
      <td>0.000000</td>
      <td>1211.676901</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>8068</th>
      <td>PH084823000</td>
      <td>NOUL2015</td>
      <td>9.700000</td>
      <td>0.408333</td>
      <td>0.216146</td>
      <td>8.136932</td>
      <td>277.107823</td>
      <td>1.80</td>
      <td>6.25</td>
      <td>6.25</td>
      <td>...</td>
      <td>32.492212</td>
      <td>0.311526</td>
      <td>0.031153</td>
      <td>0.155763</td>
      <td>0.031153</td>
      <td>2.827833</td>
      <td>31.308411</td>
      <td>0.000000</td>
      <td>538.743551</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8069</th>
      <td>PH015547000</td>
      <td>NOUL2015</td>
      <td>17.587500</td>
      <td>1.414583</td>
      <td>0.386458</td>
      <td>9.818999</td>
      <td>305.789817</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>4.703833</td>
      <td>0.027875</td>
      <td>0.000000</td>
      <td>0.034843</td>
      <td>0.097561</td>
      <td>1.073268</td>
      <td>12.766551</td>
      <td>0.000000</td>
      <td>946.676507</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8070</th>
      <td>PH025014000</td>
      <td>NOUL2015</td>
      <td>11.487500</td>
      <td>0.614583</td>
      <td>0.230319</td>
      <td>15.791907</td>
      <td>210.313249</td>
      <td>0.06</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>...</td>
      <td>3.063753</td>
      <td>0.022528</td>
      <td>0.000000</td>
      <td>0.067583</td>
      <td>0.022528</td>
      <td>1.140109</td>
      <td>9.348952</td>
      <td>0.000000</td>
      <td>3938.254316</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8071</th>
      <td>PH140127000</td>
      <td>NOUL2015</td>
      <td>11.600000</td>
      <td>1.400000</td>
      <td>0.412766</td>
      <td>13.867145</td>
      <td>218.189328</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>...</td>
      <td>3.119093</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.837537</td>
      <td>21.928166</td>
      <td>0.000000</td>
      <td>2666.620370</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>8072</th>
      <td>PH051612000</td>
      <td>NOUL2015</td>
      <td>32.305556</td>
      <td>1.744444</td>
      <td>1.210417</td>
      <td>15.647639</td>
      <td>219.542224</td>
      <td>4.15</td>
      <td>3.05</td>
      <td>3.05</td>
      <td>...</td>
      <td>36.191860</td>
      <td>0.280316</td>
      <td>0.010382</td>
      <td>0.031146</td>
      <td>0.103821</td>
      <td>2.518110</td>
      <td>31.634136</td>
      <td>0.000000</td>
      <td>3831.302757</td>
      <td>0.00000</td>
    </tr>
  </tbody>
</table>
<p>8073 rows × 40 columns</p>
</div>




```python
df["typhoon"] = df["typhoon"].str.upper()
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
# Specify features after removing highly correlated ones (510 model)
features = [
    "HAZ_rainfall_Total",
    "HAZ_rainfall_max_6h",
    "HAZ_rainfall_max_24h",
    "HAZ_v_max",
    "HAZ_v_max_3",
    "HAZ_dis_track_min",
    "GEN_landslide_per",
    "GEN_stormsurge_per",
    #'GEN_Bu_p_inSSA',
    #'GEN_Bu_p_LS',
    "GEN_Red_per_LSbldg",
    "GEN_Or_per_LSblg",
    "GEN_Yel_per_LSSAb",
    "GEN_RED_per_SSAbldg",
    #'GEN_OR_per_SSAbldg',
    "GEN_Yellow_per_LSbl",
    "TOP_mean_slope",
    "TOP_mean_elevation_m",
    "TOP_ruggedness_stdev",
    #'TOP_mean_ruggedness',
    #'TOP_slope_stdev',
    "VUL_poverty_perc",
    "GEN_with_coast",
    "GEN_coast_length",
    "VUL_Housing_Units",
    "VUL_StrongRoof_StrongWall",
    "VUL_StrongRoof_LightWall",
    "VUL_StrongRoof_SalvageWall",
    "VUL_LightRoof_StrongWall",
    "VUL_LightRoof_LightWall",
    "VUL_LightRoof_SalvageWall",
    "VUL_SalvagedRoof_StrongWall",
    "VUL_SalvagedRoof_LightWall",
    "VUL_SalvagedRoof_SalvageWall",
    "VUL_vulnerable_groups",
    "VUL_pantawid_pamilya_beneficiary",
]
```


```python
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df["y_norm"], bins=bins2)
```


```python
# Define range of for loops
num_exp_main = 20

# num_exp = 12
# typhoons_for_test = typhoons[-num_exp:]

# Define range of for loop (LOO)
num_exp = len(typhoons)

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

    test_RMSE_lst_M1 = []
    test_RMSE_bin_M1 = defaultdict(list)

    # Defin two lists to save RMSE and Average Error
    RMSE = defaultdict(list)
    AVE = defaultdict(list)

    for run_ix in range(27, num_exp):

        # WITHOUT removing old typhoons from training set
        # typhoons_train_lst = typhoons[0 : run_ix + 27]

        # In each run Keep one typhoon for the test list while the rest of the typhoons in the training set
        typhoons_for_test = typhoons[run_ix]
        typhoons_train_lst = typhoons[:run_ix] + typhoons[run_ix + 1 :]

        bin_index2 = np.digitize(df["y_norm"], bins=binsP2)
        y_input_strat = bin_index2

        # Split X and y from dataframe features
        X = df[features]
        y = df["y_norm"]

        # Split df to train and test (one typhoon for test and the rest of typhoons for train)
        df_test = df[df["typhoon"] == typhoons_for_test]

        # df_test = df[df["typhoon"] == typhoons_for_test[run_ix]]

        df_train = pd.DataFrame()
        for run_ix_train in range(len(typhoons_train_lst)):
            df_train = df_train.append(
                df[df["typhoon"] == typhoons_train_lst[run_ix_train]]
            )

        # Split X and y from dataframe features
        X_test = df_test[features]
        X_train = df_train[features]

        y_train = df_train["y_norm"]
        y_test = df_test["y_norm"]

        print(df_test["typhoon"].unique())

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
        y_pred = xgb.predict(X_test)

        # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        ave = (y_pred - y_test).sum() / len(y_test)

        print(f"RMSE for grid_based model: {rmse:.2f}")
        print(f"Average Error for grid_based model: {ave:.2f}")

        RMSE["all"].append(rmse)
        AVE["all"].append(ave)

        bin_index = np.digitize(y_test, bins=binsP2)

        for bin_num in range(1, 6):
            if len(y_test[bin_index == bin_num]) > 0:

                mse_idx = mean_squared_error(
                    y_test[bin_index == bin_num], y_pred[bin_index == bin_num]
                )
                rmse = np.sqrt(mse_idx)

                ave = (
                    y_pred[bin_index == bin_num] - y_test[bin_index == bin_num]
                ).sum() / len(y_test[bin_index == bin_num])

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
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.13
    Average Error for grid_based model: 0.71
    ['SARIKA2016']
    RMSE for grid_based model: 0.74
    Average Error for grid_based model: 0.10
    ['MANGKHUT2018']
    RMSE for grid_based model: 4.21
    Average Error for grid_based model: 1.06
    ['YUTU2018']
    RMSE for grid_based model: 1.09
    Average Error for grid_based model: 0.29
    ['KAMMURI2019']
    RMSE for grid_based model: 5.33
    Average Error for grid_based model: 0.92
    ['NAKRI2019']
    RMSE for grid_based model: 0.01
    Average Error for grid_based model: -0.00
    ['PHANFONE2019']
    RMSE for grid_based model: 3.96
    Average Error for grid_based model: -0.52
    ['SAUDEL2020']
    RMSE for grid_based model: 0.61
    Average Error for grid_based model: 0.59
    ['GONI2020']
    RMSE for grid_based model: 3.59
    Average Error for grid_based model: -0.53
    ['VAMCO2020']
    RMSE for grid_based model: 1.94
    Average Error for grid_based model: 0.28
    ['VONGFONG2020']
    RMSE for grid_based model: 3.58
    Average Error for grid_based model: -0.02
    ['MOLAVE2020']
    RMSE for grid_based model: 2.49
    Average Error for grid_based model: 1.36
    [4.1348099463386605, 0.7378040992410009, 4.213895233370351, 1.0906709033799908, 5.32707165548746, 0.008159077816887742, 3.9557687532058745, 0.6094510118572806, 3.5896507283958603, 1.9438042382746203, 3.5835932069188177, 2.485737804638981]



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

    RMSE: 2.64
    stdev: 1.66
    Average Error: 0.35
    Stdev of Average Error: 0.57



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
    RMSE: 0.25
    STDEV: 0.17
    Average_Error: 0.10
    Stdev of Average_Error: 0.09
    
    RMSE & STDEV & Average Error per bin 2
    RMSE: 1.82
    STDEV: 1.41
    Average_Error: 1.10
    Stdev of Average_Error: 1.17
    
    RMSE & STDEV & Average Error per bin 3
    RMSE: 5.16
    STDEV: 2.32
    Average_Error: 1.84
    Stdev of Average_Error: 2.26
    
    RMSE & STDEV & Average Error per bin 4
    RMSE: 10.41
    STDEV: 4.58
    Average_Error: -4.16
    Stdev of Average_Error: 6.95
    
    RMSE & STDEV & Average Error per bin 5
    RMSE: 33.24
    STDEV: 11.17
    Average_Error: -33.24
    Stdev of Average_Error: 11.17



```python
# WF-increasing (without removing old typhoons in the training set)

# Calculate the averages of each position wrt typhoons in test set (we do this average estimation to use it for plotting)
Mun_rmse_wf = [sum(values) / len(values) for values in zip(*main_RMSE_lst)]
Mun_ave_wf = [sum(values) / len(values) for values in zip(*main_AVE_lst)]

df_wf = pd.DataFrame(
    columns=[
        "Mun_rmse_wf",
        "Mun_ave_wf",
    ]
)

df_wf["Mun_rmse_wf"] = Mun_rmse_wf
df_wf["Mun_ave_wf"] = Mun_ave_wf
df_wf.to_csv("df_wf.csv", index=False)
```


```python
# WF-increasing (without removing old typhoons in the training set)

# Calculate the averages of each position wrt typhoons in test set (we do this average estimation to use it for plotting)
Mun_rmse_loo = [sum(values) / len(values) for values in zip(*main_RMSE_lst)]
Mun_ave_loo = [sum(values) / len(values) for values in zip(*main_AVE_lst)]

df_loo = pd.DataFrame(
    columns=[
        "Mun_rmse_loo",
        "Mun_ave_loo",
    ]
)

df_loo["Mun_rmse_loo"] = Mun_rmse_loo
df_loo["Mun_ave_loo"] = Mun_ave_loo
df_loo.to_csv("df_loo.csv", index=False)
```


```python

```

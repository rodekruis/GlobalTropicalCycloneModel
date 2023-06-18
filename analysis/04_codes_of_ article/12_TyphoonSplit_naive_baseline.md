# Typhoon split Naive baseline

#### We split based on typhoons


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

from math import sqrt
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.dummy import DummyRegressor
from sty import fg, rs
from itertools import chain

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
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>...</th>
      <th>std_tri</th>
      <th>mean_elev</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged</th>
      <th>percent_houses_damaged_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DURIAN2006</td>
      <td>2006</td>
      <td>8284</td>
      <td>12.460039</td>
      <td>275.018491</td>
      <td>0.670833</td>
      <td>0.313021</td>
      <td>0.479848</td>
      <td>-0.213039</td>
      <td>31.336503</td>
      <td>...</td>
      <td>34.629550</td>
      <td>42.218750</td>
      <td>5303.659490</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN2006</td>
      <td>2006</td>
      <td>8286</td>
      <td>11.428974</td>
      <td>297.027578</td>
      <td>0.929167</td>
      <td>0.343229</td>
      <td>55.649739</td>
      <td>0.206000</td>
      <td>23.447758</td>
      <td>...</td>
      <td>25.475388</td>
      <td>72.283154</td>
      <td>61015.543599</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.14</td>
      <td>0.86</td>
      <td>276.871504</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN2006</td>
      <td>2006</td>
      <td>8450</td>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>0.716667</td>
      <td>0.424479</td>
      <td>8.157414</td>
      <td>-0.636000</td>
      <td>31.336503</td>
      <td>...</td>
      <td>54.353996</td>
      <td>102.215198</td>
      <td>66707.438070</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>0.89</td>
      <td>448.539453</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN2006</td>
      <td>2006</td>
      <td>8451</td>
      <td>12.511864</td>
      <td>273.639330</td>
      <td>0.568750</td>
      <td>0.336979</td>
      <td>88.292015</td>
      <td>-0.227500</td>
      <td>31.336503</td>
      <td>...</td>
      <td>31.814048</td>
      <td>58.988877</td>
      <td>53841.050168</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>2101.708435</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN2006</td>
      <td>2006</td>
      <td>8452</td>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>0.589583</td>
      <td>0.290625</td>
      <td>962.766739</td>
      <td>-0.299667</td>
      <td>23.546053</td>
      <td>...</td>
      <td>25.976413</td>
      <td>111.386527</td>
      <td>87378.257957</td>
      <td>1</td>
      <td>0.07</td>
      <td>0.46</td>
      <td>0.47</td>
      <td>11632.726327</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



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
      <th>typhoon_name</th>
      <th>wind_speed</th>
      <th>track_distance</th>
      <th>rainfall_max_6h</th>
      <th>rainfall_max_24h</th>
      <th>total_houses</th>
      <th>rwi</th>
      <th>strong_roof_strong_wall</th>
      <th>strong_roof_light_wall</th>
      <th>strong_roof_salvage_wall</th>
      <th>...</th>
      <th>std_tri</th>
      <th>mean_elev</th>
      <th>coast_length</th>
      <th>with_coast</th>
      <th>urban</th>
      <th>rural</th>
      <th>water</th>
      <th>total_pop</th>
      <th>percent_houses_damaged</th>
      <th>percent_houses_damaged_5years</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DURIAN2006</td>
      <td>12.460039</td>
      <td>275.018491</td>
      <td>0.670833</td>
      <td>0.313021</td>
      <td>0.479848</td>
      <td>-0.213039</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>...</td>
      <td>34.629550</td>
      <td>42.218750</td>
      <td>5303.659490</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DURIAN2006</td>
      <td>11.428974</td>
      <td>297.027578</td>
      <td>0.929167</td>
      <td>0.343229</td>
      <td>55.649739</td>
      <td>0.206000</td>
      <td>23.447758</td>
      <td>23.591571</td>
      <td>0.037516</td>
      <td>...</td>
      <td>25.475388</td>
      <td>72.283154</td>
      <td>61015.543599</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.14</td>
      <td>0.86</td>
      <td>276.871504</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>DURIAN2006</td>
      <td>13.077471</td>
      <td>262.598363</td>
      <td>0.716667</td>
      <td>0.424479</td>
      <td>8.157414</td>
      <td>-0.636000</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>...</td>
      <td>54.353996</td>
      <td>102.215198</td>
      <td>66707.438070</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.11</td>
      <td>0.89</td>
      <td>448.539453</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DURIAN2006</td>
      <td>12.511864</td>
      <td>273.639330</td>
      <td>0.568750</td>
      <td>0.336979</td>
      <td>88.292015</td>
      <td>-0.227500</td>
      <td>31.336503</td>
      <td>29.117802</td>
      <td>0.042261</td>
      <td>...</td>
      <td>31.814048</td>
      <td>58.988877</td>
      <td>53841.050168</td>
      <td>1</td>
      <td>0.00</td>
      <td>0.12</td>
      <td>0.88</td>
      <td>2101.708435</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>DURIAN2006</td>
      <td>11.977511</td>
      <td>284.680297</td>
      <td>0.589583</td>
      <td>0.290625</td>
      <td>962.766739</td>
      <td>-0.299667</td>
      <td>23.546053</td>
      <td>23.660429</td>
      <td>0.037576</td>
      <td>...</td>
      <td>25.976413</td>
      <td>111.386527</td>
      <td>87378.257957</td>
      <td>1</td>
      <td>0.07</td>
      <td>0.46</td>
      <td>0.47</td>
      <td>11632.726327</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 29 columns</p>
</div>



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
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>id</th>
      <th>numbuildings</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012801000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>1052</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.586399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012810000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>0</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012815000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>742</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.413601</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012801000</td>
      <td>11050.0</td>
      <td>120.9E_18.4N</td>
      <td>193</td>
      <td>11050</td>
      <td>196</td>
      <td>0.984694</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012810000</td>
      <td>11050.0</td>
      <td>120.9E_18.4N</td>
      <td>0</td>
      <td>11050</td>
      <td>196</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Change name of column ['id'] to ['grid_point_id'] the same name as in input df
df_weight.rename(columns={"id": "grid_point_id"}, inplace=True)
df_weight.head()
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
      <th>ADM3_PCODE</th>
      <th>id_x</th>
      <th>Centroid</th>
      <th>numbuildings_x</th>
      <th>grid_point_id</th>
      <th>numbuildings</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012801000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>1052</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.586399</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012810000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>0</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012815000</td>
      <td>11049.0</td>
      <td>120.9E_18.5N</td>
      <td>742</td>
      <td>11049</td>
      <td>1794</td>
      <td>0.413601</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012801000</td>
      <td>11050.0</td>
      <td>120.9E_18.4N</td>
      <td>193</td>
      <td>11050</td>
      <td>196</td>
      <td>0.984694</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012810000</td>
      <td>11050.0</td>
      <td>120.9E_18.4N</td>
      <td>0</td>
      <td>11050</td>
      <td>196</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Define bins
bins2 = [0, 0.00009, 1, 10, 50, 101]
samples_per_bin2, binsP2 = np.histogram(df_data["percent_houses_damaged"], bins=bins2)
```


```python
# Define range of for loops
num_exp_main = 20

# Walk-forward
# num_exp = 12
# typhoons_for_test = typhoons[-num_exp:]

# LOOCV
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

    RMSE = defaultdict(list)
    AVE = defaultdict(list)
        
    #for run_ix in range(num_exp):

        # WITHOUT removing old typhoons from training set
        # typhoons_train_lst = typhoons[0 : run_ix + 27]
        
    for run_ix in range(27, num_exp):

        # In each run Keep one typhoon for the test list while the rest of the typhoons in the training set
        typhoons_for_test = typhoons[run_ix]
        typhoons_train_lst = typhoons[:run_ix] + typhoons[run_ix + 1 :]


        bin_index2 = np.digitize(df_data["percent_houses_damaged"], bins=binsP2)
        y_input_strat = bin_index2

        # Split X and y from dataframe features
        X = pd.Series([0] * 49754)
        y = df_data["percent_houses_damaged"]
        
        # For Walk-forward
        # df_test = df[df["typhoon_name"] == typhoons_for_test[run_ix]]

        # For LOOCV
        df_test = df_data[df_data["typhoon_name"] == typhoons_for_test]


        df_train = pd.DataFrame()
        for run_ix_train in range(len(typhoons_train_lst)):
            df_train = df_train.append(
                df_data[df_data["typhoon_name"] == typhoons_train_lst[run_ix_train]]
            )

        # Split X and y from dataframe features
        # X_test = df_test["percent_houses_damaged"]
        # X_train = df_train["percent_houses_damaged"]

        X_test = pd.Series([0] * len(df_test))
        X_train = pd.Series([0] * len(df_train))

        y_train = df_train["percent_houses_damaged"]
        y_test = df_test["percent_houses_damaged"]

        print(df_test["typhoon_name"].unique())

        # create a dummy regressor
        dummy_reg = DummyRegressor(strategy="mean")

        # fit it on the training set
        dummy_reg.fit(X_train, y_train)

        # make predictions on the test set
        y_pred = dummy_reg.predict(X_test)

        # Define an empty data frame to insert real and prediction values
        pred_df = pd.DataFrame(columns=["y_all", "y_pred_all"])

        # Normal prediction of test data (in all the ranges values of target)
        pred_df["y_all"] = y_test
        pred_df["y_pred_all"] = y_pred

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
            join_df["y_pred_all"] * join_df["weight"] * join_df["total_houses"] / 100
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
        agg_df["y_norm"] = agg_df["weight*y*houses"] / agg_df["weight*houses"] * 100

        # Drop not required column y and y_pred before multiplying by weight
        agg_df.drop("y_all", axis=1, inplace=True)
        agg_df.drop("y_pred_all", axis=1, inplace=True)

        # Remove rows with NaN after normalization
        final_df = agg_df.dropna()

        # Calculate RMSE & Average Error in total for converted grid_based model to Mun_based

        if (len(final_df["y_norm"])) != 0:
            rmse = sqrt(mean_squared_error(final_df["y_norm"], final_df["y_pred_norm"]))
            ave = (final_df["y_pred_norm"] - final_df["y_norm"]).sum() / len(
                final_df["y_norm"]
            )

            print(f"RMSE for grid_based model: {rmse:.2f}")
            print(f"Average Error for grid_based model: {ave:.2f}")

            RMSE["all"].append(rmse)
            AVE["all"].append(ave)

        bin_index = np.digitize(final_df["y_norm"], bins=binsP2)

        for bin_num in range(1, 6):
            if len(final_df["y_norm"][bin_index == bin_num]) > 0:

                mse_idx = mean_squared_error(
                    final_df["y_norm"][bin_index == bin_num],
                    final_df["y_pred_norm"][bin_index == bin_num],
                )
                rmse = np.sqrt(mse_idx)

                ave = (
                    final_df["y_pred_norm"][bin_index == bin_num]
                    - final_df["y_norm"][bin_index == bin_num]
                ).sum() / len(final_df["y_norm"][bin_index == bin_num])

                RMSE[bin_num].append(rmse)
                AVE[bin_num].append(ave)

        # Save total RMSE in each iteration

    main_RMSE_lst.append(RMSE["all"])
    main_AVE_lst.append(AVE["all"])

    for bin_num in range(1, 6):

        # Save RMSE per bin in each iteration
        main_RMSE_bin[bin_num].append(RMSE[bin_num])
        main_AVE_bin[bin_num].append(AVE[bin_num])
```

    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77
    ['NOCK-TEN2016']
    RMSE for grid_based model: 4.44
    Average Error for grid_based model: -0.23
    ['SARIKA2016']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['MANGKHUT2018']
    RMSE for grid_based model: 2.04
    Average Error for grid_based model: -0.08
    ['YUTU2018']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.72
    ['KAMMURI2019']
    RMSE for grid_based model: 2.90
    Average Error for grid_based model: -0.23
    ['NAKRI2019']
    RMSE for grid_based model: 0.83
    Average Error for grid_based model: 0.83
    ['PHANFONE2019']
    RMSE for grid_based model: 2.58
    Average Error for grid_based model: 0.12
    ['SAUDEL2020']
    RMSE for grid_based model: 0.85
    Average Error for grid_based model: 0.85
    ['GONI2020']
    RMSE for grid_based model: 3.02
    Average Error for grid_based model: 0.15
    ['VAMCO2020']
    RMSE for grid_based model: 1.17
    Average Error for grid_based model: 0.54
    ['VONGFONG2020']
    RMSE for grid_based model: 2.60
    Average Error for grid_based model: 0.66
    ['MOLAVE2020']
    RMSE for grid_based model: 0.88
    Average Error for grid_based model: 0.77



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

    RMSE: 1.92
    stdev: 1.14
    Average Error: 0.41
    Stdev of Average Error: 0.42



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
    RMSE: 0.84
    STDEV: 0.01
    Average_Error: 0.84
    Stdev of Average_Error: 0.01
    
    RMSE & STDEV & Average Error per bin 2
    RMSE: 0.75
    STDEV: 0.06
    Average_Error: 0.72
    Stdev of Average_Error: 0.08
    
    RMSE & STDEV & Average Error per bin 3
    RMSE: 3.38
    STDEV: 1.13
    Average_Error: -2.66
    Stdev of Average_Error: 0.99
    
    RMSE & STDEV & Average Error per bin 4
    RMSE: 16.37
    STDEV: 3.90
    Average_Error: -15.32
    Stdev of Average_Error: 3.15
    
    RMSE & STDEV & Average Error per bin 5
    RMSE: 57.02
    STDEV: 6.19
    Average_Error: -57.02
    Stdev of Average_Error: 6.19



```python

```

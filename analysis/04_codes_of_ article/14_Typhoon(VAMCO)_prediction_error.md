```python
import numpy as np
import pandas as pd
```


```python
df=pd.read_csv("VAMCO2020.csv")
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
      <th>grid_point_id</th>
      <th>real_dmg</th>
      <th>pred_dmg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9233</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9234</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9235</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9236</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>18795</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>18796</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>18797</td>
      <td>0.0</td>
      <td>0.022712</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>18962</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>18963</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1349 rows × 3 columns</p>
</div>




```python
df['prediction_error'] = df['pred_dmg'] - df['real_dmg']
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
      <th>grid_point_id</th>
      <th>real_dmg</th>
      <th>pred_dmg</th>
      <th>prediction_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9233</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9234</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9235</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9236</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9237</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1344</th>
      <td>18795</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1345</th>
      <td>18796</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1346</th>
      <td>18797</td>
      <td>0.0</td>
      <td>0.022712</td>
      <td>0.022712</td>
    </tr>
    <tr>
      <th>1347</th>
      <td>18962</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1348</th>
      <td>18963</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>1349 rows × 4 columns</p>
</div>




```python
# Count the number of positive and negative errors
positive_errors = sum(df['prediction_error'] > 0)
negative_errors = sum(df['prediction_error'] < 0)
```


```python
print(f"positive_errors rate is: {positive_errors}")
print(f"negative_errors rate is: {negative_errors}")
```

    positive_errors rate is: 796
    negative_errors rate is: 346



```python
#df.to_csv('VAMCO2020_updated.csv', index=False)
```

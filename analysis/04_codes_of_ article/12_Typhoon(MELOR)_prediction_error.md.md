### Real Damage, Predicted Damage and The Prediction Error for typhoon Melor


```python
import numpy as np
import pandas as pd
```


```python
df=pd.read_csv("MELOR_gridTomun_fulldata.csv")
df.head()
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
      <th>mun_code</th>
      <th>real_damg</th>
      <th>pred_damg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012901000</td>
      <td>0.0</td>
      <td>0.057195</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012908000</td>
      <td>0.0</td>
      <td>0.060973</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012931000</td>
      <td>0.0</td>
      <td>0.074185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012932000</td>
      <td>0.0</td>
      <td>0.121766</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012933000</td>
      <td>0.0</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['prediction_error'] = df['pred_damg'] - df['real_damg']
df.head()
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
      <th>mun_code</th>
      <th>real_damg</th>
      <th>pred_damg</th>
      <th>prediction_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH012901000</td>
      <td>0.0</td>
      <td>0.057195</td>
      <td>0.057195</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH012908000</td>
      <td>0.0</td>
      <td>0.060973</td>
      <td>0.060973</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH012931000</td>
      <td>0.0</td>
      <td>0.074185</td>
      <td>0.074185</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH012932000</td>
      <td>0.0</td>
      <td>0.121766</td>
      <td>0.121766</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH012933000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df.to_csv('MELOR_gridTomun_fulldata_updated.csv', index=False)
```

### Reduced dataset to the same size of municipality dataset by intersection of the municipalities


```python
df1=pd.read_csv("MELOR_gridTomun_reduceddata.csv")
df1.head()
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
      <th>mun_code</th>
      <th>real_damg</th>
      <th>pred_damg</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH034920000</td>
      <td>0.034281</td>
      <td>0.497263</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH035403000</td>
      <td>0.000000</td>
      <td>0.001555</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH035418000</td>
      <td>0.000000</td>
      <td>0.001456</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH037707000</td>
      <td>0.058259</td>
      <td>0.848913</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH041005000</td>
      <td>0.005456</td>
      <td>1.842850</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1['prediction_error'] = df1['pred_damg'] - df1['real_damg']
df1.head()
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
      <th>mun_code</th>
      <th>real_damg</th>
      <th>pred_damg</th>
      <th>prediction_error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PH034920000</td>
      <td>0.034281</td>
      <td>0.497263</td>
      <td>0.462981</td>
    </tr>
    <tr>
      <th>1</th>
      <td>PH035403000</td>
      <td>0.000000</td>
      <td>0.001555</td>
      <td>0.001555</td>
    </tr>
    <tr>
      <th>2</th>
      <td>PH035418000</td>
      <td>0.000000</td>
      <td>0.001456</td>
      <td>0.001456</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PH037707000</td>
      <td>0.058259</td>
      <td>0.848913</td>
      <td>0.790654</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PH041005000</td>
      <td>0.005456</td>
      <td>1.842850</td>
      <td>1.837394</td>
    </tr>
  </tbody>
</table>
</div>




```python
#df1.to_csv('MELOR_gridTomun_reduceddata_updated.csv', index=False)
```

# Real Damage, Predicted Damage and The Prediction Error for typhoon Melor

```python
import numpy as np
import pandas as pd
```

```python
df=pd.read_csv("MELOR_gridTomun_fulldata.csv")
df.head()
```

```python
df['prediction_error'] = df['pred_damg'] - df['real_damg']
df.head()
```

```python
#df.to_csv('MELOR_gridTomun_fulldata_updated.csv', index=False)
```

## Reduced dataset to the same size of municipality dataset by intersection of

## the municipalities

```python
df1=pd.read_csv("MELOR_gridTomun_reduceddata.csv")
df1.head()
```

```python
df1['prediction_error'] = df1['pred_damg'] - df1['real_damg']
df1.head()
```

```python
#df1.to_csv('MELOR_gridTomun_reduceddata_updated.csv', index=False)
```

# Windfields

This notebook is for downloading typhoon tracks from
IBTrACS and generating the windfields.

TODOs:

- Get the full list of track IDs from IBTraCS (link is below)
- Add interpolation between the timesteps
- Understand if we need to change any of the input parameters
  to the climada methods
- Aggregate up to a 0.1 deg grid and save output


```python
%load_ext jupyter_black
```

    The jupyter_black extension is already loaded. To reload it, use:
      %reload_ext jupyter_black



```python
from pathlib import Path
import os

from climada.hazard import Centroids, TCTracks, TropCyclone
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
```

## Get typhoon data

Typhoon IDs from IBTrACS are taken from 
[here](https://ncics.org/ibtracs/index.php?name=browse-name)


```python
# Import list of typhoons to a dataframe
typhoons_df = pd.read_csv("typhoons.csv")
typhoons_df
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
      <th>typhoon_id</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2008169N08135</td>
      <td>FENGSHEN</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2009268N14128</td>
      <td>KETSANA</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2010191N12138</td>
      <td>CONSON</td>
      <td>2010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011266N13139</td>
      <td>NESAT</td>
      <td>2011</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2012331N03157</td>
      <td>BOPHA</td>
      <td>2012</td>
    </tr>
    <tr>
      <th>6</th>
      <td>2013282N14132</td>
      <td>NARI</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2013301N13142</td>
      <td>KROSA</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>8</th>
      <td>2013306N07162</td>
      <td>HAIYAN</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2013259N17132</td>
      <td>USAGI</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2013220N12137</td>
      <td>UTOR</td>
      <td>2013</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2014362N07130</td>
      <td>JANGMI</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>12</th>
      <td>2014254N10142</td>
      <td>KALMAEGI</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>13</th>
      <td>2014190N08154</td>
      <td>RAMMASUN</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>14</th>
      <td>2014334N02156</td>
      <td>HAGUPIT</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>15</th>
      <td>2014260N13135</td>
      <td>FUNG-WONG</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>16</th>
      <td>2014015N10129</td>
      <td>LINGLING</td>
      <td>2014</td>
    </tr>
    <tr>
      <th>17</th>
      <td>2015273N12130</td>
      <td>MUJIGAE</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>18</th>
      <td>2015344N07145</td>
      <td>MELOR</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>19</th>
      <td>2015122N07144</td>
      <td>NOUL</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>20</th>
      <td>2015226N12151</td>
      <td>GONI2015</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>21</th>
      <td>2015183N13130</td>
      <td>LINFA</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>22</th>
      <td>2015285N14151</td>
      <td>KOPPU</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>23</th>
      <td>2015012N09146</td>
      <td>MEKKHALA</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>24</th>
      <td>2016288N07145</td>
      <td>HAIMA</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2016328N09130</td>
      <td>TOKAGE</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2016253N13144</td>
      <td>MERANTI</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>27</th>
      <td>2016355N07146</td>
      <td>NOCK-TEN</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>28</th>
      <td>2016287N13130</td>
      <td>SARIKA</td>
      <td>2016</td>
    </tr>
    <tr>
      <th>29</th>
      <td>2018250N12170</td>
      <td>MANGKHUT</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>30</th>
      <td>2018294N08161</td>
      <td>YUTU</td>
      <td>2018</td>
    </tr>
    <tr>
      <th>31</th>
      <td>2019329N09160</td>
      <td>KAMMURI</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>32</th>
      <td>2019308N13114</td>
      <td>NAKRI</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>33</th>
      <td>2019354N05151</td>
      <td>PHANFONE</td>
      <td>2019</td>
    </tr>
    <tr>
      <th>34</th>
      <td>2020291N06141</td>
      <td>SAUDEL</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>35</th>
      <td>2020299N11144</td>
      <td>GONI2020</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>36</th>
      <td>2020313N08135</td>
      <td>VAMCO</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>37</th>
      <td>2020129N07134</td>
      <td>VONGFONG</td>
      <td>2020</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Download all tracks from the west pacific basin
sel_ibtracs = TCTracks.from_ibtracs_netcdf(
    provider="usa", year_range=(2006, 2022), basin="WP"
)
sel_ibtracs.size
```

    2022-11-28 21:07:42,917 - climada.hazard.tc_tracks - WARNING - 18 storm events are discarded because no valid wind/pressure values have been found: 2006184N16110, 2007276N20147, 2008223N27151, 2009291N16111, 2009328N06108, ...





    494




```python
# Get_track returns the first matching track based
# on the track ID. Interpolate from 3 hours to
# 30 minute intervals to create a smooth intensity field.
tc_tracks = TCTracks()
for typhoon_id in typhoons_df["typhoon_id"]:
    tc_track = sel_ibtracs.get_track(typhoon_id)
    tc_track = tc_track.interp(
        time=pd.date_range(
            tc_track.time.values[0], tc_track.time.values[-1], freq="30T"
        )
    )
    tc_tracks.append(tc_track)

display(tc_tracks.data[:1])
```


    [<xarray.Dataset>
     Dimensions:                 (time: 733)
     Coordinates:
         lat                     (time) float64 6.1 6.118 6.137 ... 11.66 11.68 11.7
         lon                     (time) float64 149.8 149.7 149.6 ... 81.93 81.8
       * time                    (time) datetime64[ns] 2006-11-24T12:00:00 ... 200...
     Data variables:
         time_step               (time) float64 3.0 3.0 3.0 3.0 ... 3.0 3.0 3.0 3.0
         radius_max_wind         (time) float64 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
         radius_oci              (time) float64 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
         max_sustained_wind      (time) float64 15.0 15.0 15.0 ... 15.0 15.0 15.0
         central_pressure        (time) float64 1.006e+03 1.006e+03 ... 1.006e+03
         environmental_pressure  (time) float64 1.008e+03 1.008e+03 ... 1.006e+03
         basin                   (time) <U2 'WP' 'WP' 'WP' 'WP' ... 'NI' 'NI' 'NI'
     Attributes:
         max_sustained_wind_unit:  kn
         central_pressure_unit:    mb
         name:                     DURIAN
         sid:                      2006329N06150
         orig_event_flag:          True
         data_provider:            ibtracs_usa
         id_no:                    2006329006150.0
         category:                 4]



```python
# Plot the tracks
# Takes a while, especially after the interpolation.
tc_tracks.plot()
```




    <GeoAxesSubplot:>




    
![png](output_7_1.png)
    


## Construct the windfield

The typhoon tracks will be used to construct the wind field.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.


```python
# input_dir = (Path(os.getenv("STORM_DATA_DIR"))/ "analysis/02_new_model_input/input")
# filepath = input_dir / "phl_0.1_degree_grid_centroids.gpkg"
# gdf = gpd.read_file(filepath)

filepath = "input/phl_0.1_degree_grid_centroids.gpkg"
gdf = gpd.read_file(filepath)

gdf
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
      <th>id</th>
      <th>left</th>
      <th>top</th>
      <th>right</th>
      <th>bottom</th>
      <th>Area</th>
      <th>AreainKM</th>
      <th>Len</th>
      <th>Longitude</th>
      <th>Latitude</th>
      <th>Centroid</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>114.25</td>
      <td>21.15</td>
      <td>114.35</td>
      <td>21.05</td>
      <td>1.150374e+08</td>
      <td>115.037</td>
      <td>10.72555</td>
      <td>114.3</td>
      <td>21.1</td>
      <td>114.3E_21.1N</td>
      <td>POINT (114.30000 21.10000)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.0</td>
      <td>114.25</td>
      <td>21.05</td>
      <td>114.35</td>
      <td>20.95</td>
      <td>1.151129e+08</td>
      <td>115.113</td>
      <td>10.72907</td>
      <td>114.3</td>
      <td>21.0</td>
      <td>114.3E_21N</td>
      <td>POINT (114.30000 21.00000)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.0</td>
      <td>114.25</td>
      <td>20.95</td>
      <td>114.35</td>
      <td>20.85</td>
      <td>1.151881e+08</td>
      <td>115.188</td>
      <td>10.73257</td>
      <td>114.3</td>
      <td>20.9</td>
      <td>114.3E_20.9N</td>
      <td>POINT (114.30000 20.90000)</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>114.25</td>
      <td>20.85</td>
      <td>114.35</td>
      <td>20.75</td>
      <td>1.152629e+08</td>
      <td>115.263</td>
      <td>10.73606</td>
      <td>114.3</td>
      <td>20.8</td>
      <td>114.3E_20.8N</td>
      <td>POINT (114.30000 20.80000)</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>114.25</td>
      <td>20.75</td>
      <td>114.35</td>
      <td>20.65</td>
      <td>1.153373e+08</td>
      <td>115.337</td>
      <td>10.73952</td>
      <td>114.3</td>
      <td>20.7</td>
      <td>114.3E_20.7N</td>
      <td>POINT (114.30000 20.70000)</td>
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
    </tr>
    <tr>
      <th>20745</th>
      <td>20746.0</td>
      <td>126.65</td>
      <td>5.05</td>
      <td>126.75</td>
      <td>4.95</td>
      <td>1.226348e+08</td>
      <td>122.635</td>
      <td>11.07406</td>
      <td>126.7</td>
      <td>5.0</td>
      <td>126.7E_5N</td>
      <td>POINT (126.70000 5.00000)</td>
    </tr>
    <tr>
      <th>20746</th>
      <td>20747.0</td>
      <td>126.65</td>
      <td>4.95</td>
      <td>126.75</td>
      <td>4.85</td>
      <td>1.226529e+08</td>
      <td>122.653</td>
      <td>11.07488</td>
      <td>126.7</td>
      <td>4.9</td>
      <td>126.7E_4.9N</td>
      <td>POINT (126.70000 4.90000)</td>
    </tr>
    <tr>
      <th>20747</th>
      <td>20748.0</td>
      <td>126.65</td>
      <td>4.85</td>
      <td>126.75</td>
      <td>4.75</td>
      <td>1.226705e+08</td>
      <td>122.671</td>
      <td>11.07567</td>
      <td>126.7</td>
      <td>4.8</td>
      <td>126.7E_4.8N</td>
      <td>POINT (126.70000 4.80000)</td>
    </tr>
    <tr>
      <th>20748</th>
      <td>20749.0</td>
      <td>126.65</td>
      <td>4.75</td>
      <td>126.75</td>
      <td>4.65</td>
      <td>1.226879e+08</td>
      <td>122.688</td>
      <td>11.07646</td>
      <td>126.7</td>
      <td>4.7</td>
      <td>126.7E_4.7N</td>
      <td>POINT (126.70000 4.70000)</td>
    </tr>
    <tr>
      <th>20749</th>
      <td>20750.0</td>
      <td>126.65</td>
      <td>4.65</td>
      <td>126.75</td>
      <td>4.55</td>
      <td>1.227048e+08</td>
      <td>122.705</td>
      <td>11.07722</td>
      <td>126.7</td>
      <td>4.6</td>
      <td>126.7E_4.6N</td>
      <td>POINT (126.70000 4.60000)</td>
    </tr>
  </tbody>
</table>
<p>20750 rows × 12 columns</p>
</div>




```python
# multipolygon data to centroids

cent = Centroids.from_geodataframe(gpd.read_file(filepath))

cent.check()
cent.plot();
```

    2022-11-28 21:08:13,092 - climada.util.plot - WARNING - Error parsing coordinate system 'epsg:4326'. Using projection PlateCarree in plot.



    
![png](output_10_1.png)
    



```python
# construct tropical cyclones
tc = TropCyclone.from_tracks(tc_tracks, centroids=cent, store_windfields=True)
```


```python
# Let's look at the first typhoon in the dictionary as an example.

for typhoon in typhoons_df:
    id_pplt = typhoons_df.loc[
        typhoons_df["typhoon_name"] == "NARI", "typhoon_id"
    ].iloc[0]

tc.plot_intensity(id_pplt)
```

    2022-11-28 21:09:07,278 - climada.util.plot - WARNING - Error parsing coordinate system 'epsg:4326'. Using projection PlateCarree in plot.





    <GeoAxesSubplot:title={'center':'Event ID 7: 2013282N14132'}>




    
![png](output_12_2.png)
    



```python
# Then calculate windfield
intensity = tc.intensity
```


```python
tc.intensity
```




    <39x20750 sparse matrix of type '<class 'numpy.float64'>'
    	with 132418 stored elements in Compressed Sparse Row format>




```python
windfield_data = intensity.data
display(len(windfield_data))
display(windfield_data)
```


    132418



    array([17.68550419, 18.47764793, 19.46302592, ..., 18.71531643,
           18.38318901, 17.93270828])



```python
import scipy.sparse

scipy.sparse.save_npz("sparse_matrix.npz", tc.intensity, compressed=True)
```


```python
sparse_matrix = scipy.sparse.load_npz("sparse_matrix.npz")
```


```python
wind_speed = sparse_matrix.toarray()
wind_speed
```




    array([[ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [26.11021209, 25.79097958, 25.43851576, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           ...,
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ],
           [ 0.        ,  0.        ,  0.        , ...,  0.        ,
             0.        ,  0.        ]])




```python
# Create a geodataframe
grid_geo = gpd.GeoDataFrame(
    columns=[
        "typhoon_name",
        "typhoon_year",
        "wind_speed",
        "grid_point_id",
        "grid_point_centroid",
    ],
    # geometry="grid_point_centroid",
)
```


```python
# join calculated wind_speed with id and centroid of imported gpd file

for i in range(len(wind_speed)):
    grid_geo.at[i, "typhoon_name"] = typhoons_df.loc[i, "typhoon_name"]
    grid_geo.at[i, "typhoon_year"] = typhoons_df.loc[i, "typhoon_year"]
    grid_geo.at[i, "wind_speed"] = wind_speed[i]
    grid_geo.at[i, "grid_point_id"] = gdf.id.to_numpy()
    grid_geo.at[i, "grid_point_centroid"] = gdf.geometry.to_numpy()
    # geometry_col

grid_geo.head()
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
      <th>wind_speed</th>
      <th>grid_point_id</th>
      <th>grid_point_centroid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>DURIAN</td>
      <td>2006</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, ...</td>
      <td>[POINT (114.3 21.099999999999998), POINT (114....</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FENGSHEN</td>
      <td>2008</td>
      <td>[26.110212094776074, 25.790979582866676, 25.43...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, ...</td>
      <td>[POINT (114.3 21.099999999999998), POINT (114....</td>
    </tr>
    <tr>
      <th>2</th>
      <td>KETSANA</td>
      <td>2009</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, ...</td>
      <td>[POINT (114.3 21.099999999999998), POINT (114....</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CONSON</td>
      <td>2010</td>
      <td>[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, ...</td>
      <td>[POINT (114.3 21.099999999999998), POINT (114....</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NESAT</td>
      <td>2011</td>
      <td>[0.0, 0.0, 0.0, 0.0, 17.88974553461081, 18.607...</td>
      <td>[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, ...</td>
      <td>[POINT (114.3 21.099999999999998), POINT (114....</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Explode each typhoon to multiple rows

typhoonsWindfield_df = typhoons_df.copy()
typhoonsWindfield_df["grid_point_id"] = None
typhoonsWindfield_df["wind_speed"] = None
typhoonsWindfield_df["grid_point_centroid"] = None

for i in range(len(wind_speed)):
    typhoonsWindfield_df.at[i, "wind_speed"] = wind_speed[i]
    typhoonsWindfield_df.at[i, "grid_point_id"] = gdf.id[:]

    typhoonsWindfield_df.at[i, "grid_point_centroid"] = grid_geo[
        "grid_point_centroid"
    ][:][i]
    # typhoonsWindfield_df.at[i, "grid_point_centroid"] = grid_geo["grid_point_centroid"][i][i]
typhoonsWindfield_df_gridcells = typhoonsWindfield_df.explode(
    ["wind_speed", "grid_point_id", "grid_point_centroid"]
)
typhoonsWindfield_df_gridcells
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
      <th>typhoon_id</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>grid_point_centroid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>POINT (114.3 21.099999999999998)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>POINT (114.3 20.999999999999996)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>POINT (114.3 20.9)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>POINT (114.3 20.799999999999994)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>POINT (114.3 20.700000000000003)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20746.0</td>
      <td>0.0</td>
      <td>POINT (126.7 4.999999999999997)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20747.0</td>
      <td>0.0</td>
      <td>POINT (126.7 4.8999999999999995)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20748.0</td>
      <td>0.0</td>
      <td>POINT (126.7 4.799999999999998)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20749.0</td>
      <td>0.0</td>
      <td>POINT (126.7 4.699999999999997)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20750.0</td>
      <td>0.0</td>
      <td>POINT (126.7 4.599999999999999)</td>
    </tr>
  </tbody>
</table>
<p>809250 rows × 6 columns</p>
</div>




```python
# Convert argument to a numeric type and float id to int
typhoonsWindfield_df_gridcells[
    ["grid_point_id", "wind_speed"]
] = typhoonsWindfield_df_gridcells[["grid_point_id", "wind_speed"]].apply(
    pd.to_numeric
)

typhoonsWindfield_df_gridcells[
    "grid_point_id"
] = typhoonsWindfield_df_gridcells["grid_point_id"].astype("int")
typhoonsWindfield_df_gridcells.head()
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
      <th>typhoon_id</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>grid_point_centroid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>1</td>
      <td>0.0</td>
      <td>POINT (114.3 21.099999999999998)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>2</td>
      <td>0.0</td>
      <td>POINT (114.3 20.999999999999996)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>3</td>
      <td>0.0</td>
      <td>POINT (114.3 20.9)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>4</td>
      <td>0.0</td>
      <td>POINT (114.3 20.799999999999994)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>5</td>
      <td>0.0</td>
      <td>POINT (114.3 20.700000000000003)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Remove zeros from prepared df
typhoonsWindfield_df_gridcells = typhoonsWindfield_df_gridcells[
    (typhoonsWindfield_df_gridcells[["wind_speed"]] != 0).any(axis=1)
]
typhoonsWindfield_df_gridcells
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
      <th>typhoon_id</th>
      <th>typhoon_name</th>
      <th>typhoon_year</th>
      <th>grid_point_id</th>
      <th>wind_speed</th>
      <th>grid_point_centroid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>56</td>
      <td>17.685504</td>
      <td>POINT (114.30000000000001 15.599999999999998)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>57</td>
      <td>18.477648</td>
      <td>POINT (114.30000000000001 15.499999999999996)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>58</td>
      <td>19.463026</td>
      <td>POINT (114.30000000000001 15.399999999999999)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>59</td>
      <td>20.542491</td>
      <td>POINT (114.30000000000001 15.299999999999995)</td>
    </tr>
    <tr>
      <th>0</th>
      <td>2006329N06150</td>
      <td>DURIAN</td>
      <td>2006</td>
      <td>60</td>
      <td>21.721394</td>
      <td>POINT (114.30000000000001 15.199999999999998)</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20663</td>
      <td>19.612757</td>
      <td>POINT (126.7 13.299999999999999)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20664</td>
      <td>19.148407</td>
      <td>POINT (126.7 13.199999999999998)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20665</td>
      <td>18.715316</td>
      <td>POINT (126.7 13.1)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20666</td>
      <td>18.383189</td>
      <td>POINT (126.7 13)</td>
    </tr>
    <tr>
      <th>38</th>
      <td>2020296N09137</td>
      <td>MOLAVE</td>
      <td>2020</td>
      <td>20667</td>
      <td>17.932708</td>
      <td>POINT (126.7 12.899999999999999)</td>
    </tr>
  </tbody>
</table>
<p>132418 rows × 6 columns</p>
</div>




```python
# Save df as a csv file
typhoonsWindfield_df_gridcells.to_csv("windfield_data.csv")
```


```python

```

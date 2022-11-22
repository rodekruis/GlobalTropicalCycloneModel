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
# TODO: It would be better to turn this into a Excel or
# csv file with a separate year, name, and typhoon ID,
# then read that in as a Pandas dataframe
new_typhoon_id_dict = {}
# Typhoons' name and Id
typhoon_id_dict = {
    "2006329N06150": "DURIAN",
    "2008169N08135": "FENGSHEN",
    "2009268N14128": "KETSANA",
    "2010191N12138": "CONSON",
    "2011266N13139": "NESAT",
    "2012331N03157": "BOPHA",
    "2013282N14132": "NARI",
    "2013301N13142": "KROSA",
    "2013306N07162": "HAIYAN",
    "2013259N17132": "USAGI",
    "2013220N12137": "UTOR",
    "2014362N07130": "JANGMI",
    "2014254N10142": "KALMAEGI",
    "2014190N08154": "RAMMASUN",
    "2014334N02156": "HAGUPIT",
    "2014260N13135": "FUNG-WONG",
    "2014015N10129": "LINGLING",
    "2015273N12130": "MUJIGAE",
    "2015344N07145": "MELOR",
    "2015122N07144": "NOUL",
    "2015226N12151": "GONI",
    "2015183N13130": "LINFA",
    "2015285N14151": "KOPPU",
    "2015012N09146": "MEKKHALA",
    "2016288N07145": "HAIMA",
    "2016328N09130": "TOKAGE",
    "2016253N13144": "MERANTI",
    "2016355N07146": "NOCK-TEN",
    "2016287N13130": "SARIKA",
    "2018250N12170": "MANGKHUT",
    "2018294N08161": "YUTU",
    "2019329N09160": "KAMMURI",
    "2019308N13114": "NAKRI",
    "2019354N05151": "PHANFONE",
    "2020291N06141": "SAUDEL",
    "2020299N11144": "GONI",
    "2020313N08135": "VAMCO",
    "2020129N07134": "VONGFONG",
    "2020296N09137": "MOLAVE",
}

for i in range(len(typhoon_id_dict)):
    typhoon_year = list(typhoon_id_dict.keys())[i][0:4]
    new_key = list(typhoon_id_dict.values())[i] + typhoon_year
    new_typhoon_id_dict[new_key] = list(typhoon_id_dict.keys())[i]

print(new_typhoon_id_dict)
```

```python
# Download all tracks from the west pacific basin
sel_ibtracs = TCTracks.from_ibtracs_netcdf(
    provider="usa", year_range=(2006, 2022), basin="WP"
)
sel_ibtracs.size
```

```python
# Get_track returns the first matching track based
# on the track ID. Interpolate from 3 hours to
# 30 minute intervals to create a smooth intensity field.
tc_tracks = TCTracks()
for typhoon_id in new_typhoon_id_dict.values():
    tc_track = sel_ibtracs.get_track(typhoon_id).interp(
        time=pd.date_range(
            track.time.values[0], track.time.values[-1], freq="30T"
        )
    )
    tc_tracks.append(tc_track)

display(tc_tracks.data[:1])
```

```python
# Plot the tracks
# Takes awhile, especially after the interpolation.
tc_track_list.plot()
```

## Construct the windfield

The typhoon tracks will be used to construct the wind field.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.

```python
input_dir = (
    Path(os.getenv("STORM_DATA_DIR")) / "analysis/02_new_model_input/input"
)
filepath = input_dir / "phil_0.1_degree_grid.gpkg"
gdf = gpd.read_file(filepath)

gdf
```

```python
# TODO: this should work once Pauline converts the
# multipolygon data to centroids

cent = Centroids.from_geodataframe(gpd.read_file(filepath))

cent.check()
cent.plot();
```

```python
# construct tropical cyclones
tc = TropCyclone.from_tracks(
    tc_track_list, centroids=cent, store_windfields=True
)
```

```python
# Let's look at the first typhoon in the dictionary as an example.

tc.plot_intensity(new_typhoon_id_dict["NARI2013"])
```

```python
# Then calculate windfield
intensity = tc.intensity
```

```python
tc.intensity
```

```python
windfield_data = intensity.data
windfield_data
```

```python
import scipy.sparse

scipy.sparse.save_npz("sparse_matrix.npz", tc.intensity, compressed=True)
```

```python
sparse_matrix = scipy.sparse.load_npz("sparse_matrix.npz")
```

```python
sparse_matrix.toarray()
```

```python
# TODO: We want to create an output that is a geodataframe with the
# following columns:
#
# Grid point ID
# Grid point centroid
# Typhoon name
# Typhoon year
# Wind speed
#
# From the info we have now, you can create a data frame with the last
# three columns. Then, once Pauline creates a centroid grid,
# you can add the first two columns.
```

# Windfields

This notebook is for downloading typhoon tracks from
IBTrACS and generating the windfields.

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

```python
input_dir = Path(os.getenv("STORM_DATA_DIR")) / "analysis/02_new_model_input"
```

## Get typhoon data

Typhoon IDs from IBTrACS are taken from
[here](https://ncics.org/ibtracs/index.php?name=browse-name)

```python
# Import list of typhoons to a dataframe
typhoons_df = pd.read_csv(input_dir / "01_windfield/typhoons.csv")
typhoons_df
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

```python
# Plot the tracks
# Takes a while, especially after the interpolation.
tc_tracks.plot()
```

## Construct the windfield

The typhoon tracks will be used to construct the wind field.
The wind field grid will be set using a geopackage file that is
used for all other grid-based data.

```python
# input_dir = (Path(os.getenv("STORM_DATA_DIR"))/ "analysis/02_new_model_input/input")
# filepath = input_dir / "phl_0.1_degree_grid_centroids.gpkg"
# gdf = gpd.read_file(filepath)

filepath = (
    input_dir
    / "02_housing_damage/output/phl_0.1_degree_grid_centroids_land_overlap.gpkg"
)
gdf = gpd.read_file(filepath)
gdf["id"] = gdf["id"].astype(int)

gdf
```

```python
# multipolygon data to centroids

cent = Centroids.from_geodataframe(gpd.read_file(filepath))

cent.check()
cent.plot();
```

```python
# Construct tropical cyclones,
# takes a minute to run
tc = TropCyclone.from_tracks(tc_tracks, centroids=cent, store_windfields=True)
```

```python
# Let's look at a specific typhoon as an example.
# It looks weird but I think it's just a
# projection issue, as the parts on land look fine
example_typhoon_id = "2020299N11144"  # Goni 2020
# example_typhoon_id = "2019354N05151"  # Phanfone
tc.plot_intensity(example_typhoon_id)
```

## Save the windfields

Need to extract the windfield per typhoon, and
save it in a dataframe along with the grid points

```python
df_windfield = pd.DataFrame()

for intensity_sparse, event_id in zip(tc.intensity, tc.event_name):
    windfield = intensity_sparse.toarray().flatten()
    npoints = len(windfield)
    typhoon_info = typhoons_df[typhoons_df["typhoon_id"] == event_id]
    df_to_add = pd.DataFrame(
        dict(
            typhoon_id=[event_id] * npoints,
            typhoon_name=[typhoon_info["typhoon_name"].values[0]] * npoints,
            typhoon_year=[typhoon_info["typhoon_year"].values[0]] * npoints,
            wind_speed=windfield,
            grid_point_id=gdf["id"],
        )
    )
    df_windfield = pd.concat([df_windfield, df_to_add], ignore_index=True)
df_windfield
```

```python
# Check that that the grid points match for the example typhoon.
# Looks good to me!
df_example = df_windfield[df_windfield["typhoon_id"] == example_typhoon_id]
gdf_example = gdf.merge(df_example, left_on="id", right_on="grid_point_id")
gdf_example.plot(c=gdf_example["wind_speed"])
```

```python
# Save df as a csv file
df_windfield.to_csv(input_dir / "01_windfield/windfield_data.csv")
```

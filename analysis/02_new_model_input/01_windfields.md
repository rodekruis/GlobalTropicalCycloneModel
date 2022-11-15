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
from climada.hazard import Centroids, TCTracks, TropCyclone
```

```python
help(TCTracks.from_ibtracs_netcdf)
```

Mersedeh:

I would browse [here](https://ncics.org/ibtracs/index.php?name=browse-name)
to create a list of the event IDs that we want. Here I made a start:

```python
typhoon_id_dict = {"GONI": "2020299N11144", "HAIYAN": "2013306N07162"}
```

```python
# Download all tracks from the west pacific basin
sel_ibtracs = TCTracks.from_ibtracs_netcdf(
    provider="usa", year_range=(2006, 2022), basin="WP"
)
```

```python
# Look at how many tracks there are - there are a lot
sel_ibtracs.size
```

```python
# Loop through the typhoons
tc_tracks = TCTracks()
for typhoon_name, typhoon_id in typhoon_id_dict.items():
    tc_track = sel_ibtracs.get_track(typhoon_id)
    # TODO: We need to add a step here that interpolates the timesteps
    tc_tracks.append(tc_track)
```

```python
# Plot the tracks
tc_tracks.plot()
```

```python
# Then calculate windfield
```

```python
help(TropCyclone.from_tracks)
```

```python
# construct centroids
min_lat, max_lat, min_lon, max_lon = 5.0, 20.0, 116.0, 127.0  # TODO: Adjust
cent = Centroids.from_pnt_bounds(
    (min_lon, min_lat, max_lon, max_lat), res=0.12
)
cent.check()
cent.plot();
```

```python
# construct tropical cyclones
tc = TropCyclone.from_tracks(tc_tracks, centroids=cent, store_windfields=True)
```

```python
# Let's look at Goni as an example. For now
# it's clear that we will need to interpolate the timesteps.
tc.plot_intensity(typhoon_id_dict["GONI"])
```

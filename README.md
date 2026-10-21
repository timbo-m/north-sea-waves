# North Sea Waves

Finds North Sea banks and shoals where big seas steepen and break, which is dangerous for ships, and works out the forecast conditions (offshore Hs, Tp, direction) that set each one off.

- `wave_physics.py`: shoaling plus Battjes-Janssen random-wave breaking. It inverts these to give the offshore Hs that makes 1 in 200, 1 in 20 and 1 in 5 waves break on a given depth.
- `find_hazard_banks.py`: the main pipeline. It covers seabed and open sectors, triggers, 25-year ERA5 climate, bank grouping and ranking, ray tracing over the top banks, and the forecast check.
- `make_report.py`: builds the ranked CSV, the KML map, figures and the PDF report in `results/`.
- `gazetteer.py`: approximate names for banks. Check them on an Admiralty chart.
- `tools/fetch_hindcast.py`: downloads ERA5 hindcast points from Open-Meteo. It resumes where it stopped.

## Data (not in the repo, goes in `data/`)
- EMODnet Bathymetry: mean depth tiles `bathy_*.tif`, 0.005 deg.
- EMODnet Human Activities: vessel density `vessel_density.tif`.
- Open-Meteo: ERA5 hindcast (`hindcast_<lat>_<lon>.csv.gz`) and a 7-day forecast (`forecast_all.csv.gz`).
- Natural Earth: 10m land, in `data/ne/`.

## Run
```
python find_hazard_banks.py --bathy "data/bathy_*.tif" --hindcast data --land data/ne/ne_10m_land.shp \
    --forecast data/forecast_all.csv.gz --vessels data/vessel_density.tif --out results
python make_report.py --results results
```
The ray-tracing step imports `point_breaks.trace` from the sibling Novel Surf Spots project. It expects that project checked out at `../surf-finder`. If it isn't there, the ray tracing is skipped and nothing else is affected.

This is a screening study, not a navigational product.

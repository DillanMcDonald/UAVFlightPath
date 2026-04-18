# UAVFlightPath

Generates a **boustrophedon (lawnmower) coverage flight path** for a fixed-wing UAV
over a rectangular survey area, with wind drift correction, terrain following, strip
alignment, and Monte Carlo uncertainty analysis.

## Quick start

```bash
pip install -r requirements.txt
python FlightPathGen.py
```

Outputs:
- `flight_path.html`  — interactive Plotly figure (bird's-eye + altitude profile)
- `flight_path.png`   — static PNG (kaleido; matplotlib fallback)
- `monte_carlo.html`  — MC path spread + statistics dashboard
- `monte_carlo.png`   — static PNG of MC results

## Features

| Feature | Description |
|---|---|
| Lawnmower path | East-West strips, alternating turn direction |
| Strip alignment | Repositioning leg after each turn corrects wind drift; strips start at exact nominal y positions |
| Spatially varying wind | `WindField`: base vector + 4-mode sine perturbations (smooth, realistic mesoscale variation) |
| Terrain following | `TerrainModel`: superposition of Gaussian hills; rate-limited climb/descent to maintain target AGL |
| Monte Carlo | `run_monte_carlo`: 100-run statistical sweep over airspeed, wind speed/direction, and nav noise |
| Coverage metrics | Strip deviation, coverage fraction, path length, reposition distance per run |
| Interactive plots | Plotly HTML for both nominal path and MC results; kaleido/matplotlib PNG export |

## Physics model

| Quantity | Formula |
|---|---|
| Rate of turn | `ROT (deg/s) = 1091 × tan(bank) / TAS(knots)` |
| Ground speed | `v_ground = v_airspeed_vector + v_wind_vector` |
| G-load | `n = 1 / cos(bank_angle)` |
| Position | Euler integration at `dt = 0.01 s` |
| Terrain follow | `z_target = terrain(x,y) + target_agl_ft`, rate-limited by `max_climb_rate_fpm` |

Rate of turn uses **true airspeed**, not ground speed.
Ground speed is the **vector sum** of airspeed and wind (not subtraction).

## Configuration

### UAVConfig

| Parameter | Default | Units | Description |
|---|---|---|---|
| `map_x_km` | 1.0 | km | Survey area east-west extent |
| `map_y_km` | 1.0 | km | Survey area north-south extent |
| `airspeed_knots` | 20.0 | knots | True airspeed (TAS) |
| `max_g` | 2.0 | G | Structural/comfort G-load limit — enforced at construction |
| `bank_angle_deg` | 60.0 | deg | Coordinated turn bank angle |
| `wind_speed_mph` | 5.0 | mph | Mean wind speed |
| `wind_dir_deg` | 120.0 | deg | Wind direction toward (from East, CCW) |
| `swath_width_ft` | 100.0 | ft | Camera footprint swath width — drives strip spacing |
| `target_agl_ft` | 200.0 | ft | Target altitude above ground level |
| `max_climb_rate_fpm` | 500.0 | ft/min | Maximum climb/descent rate for terrain following |

`UAVConfig.__init__` raises `ValueError` if `bank_angle_deg` produces G-load > `max_g`.

### MonteCarloConfig

| Parameter | Default | Description |
|---|---|---|
| `n_runs` | 100 | Number of simulation runs |
| `airspeed_sigma_knots` | 1.0 | Airspeed uncertainty 1-sigma (knots) |
| `wind_speed_sigma_mph` | 2.0 | Wind speed uncertainty 1-sigma (mph) |
| `wind_dir_sigma_deg` | 15.0 | Wind direction uncertainty 1-sigma (deg) |
| `nav_sigma_ft` | 3.0 | GPS/nav position noise 1-sigma per step (ft) |

## Known limitations

- **2-D strip alignment**: repositioning leg snaps strip start to exact nominal y, but
  cross-wind drift during the leg still displaces the path relative to the strip centre line.
  Cross-track error correction (e.g. CTE guidance law) not yet modelled.
- **Flat earth**: coordinates are in feet, no map projection applied.
- **Steady-state wind**: `WindField` models smooth spatial variation but no temporal gusts.
- **MC speed**: each run uses the fast vectorised (uniform-wind) path generator for speed;
  per-run `WindField` terrain integration is available by passing `wind_field=` to
  `generate_lawnmower_path` at the cost of ~10× longer runtime per run.

## Files

| File | Purpose |
|---|---|
| `FlightPathGen.py` | Path generation, physics model, MC analysis, Plotly/matplotlib visualisation |
| `Post.py` | Post-processing stub (planned: GPS log alignment, coverage gap analysis, GeoJSON export) |
| `requirements.txt` | Python dependencies |

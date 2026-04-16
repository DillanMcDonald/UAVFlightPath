# UAVFlightPath

Generates a **boustrophedon (lawnmower) coverage flight path** for a fixed-wing UAV
over a rectangular survey area, accounting for wind drift.

## Quick start

```bash
pip install -r requirements.txt
python FlightPathGen.py
```

Outputs:
- `flight_path.html` — interactive Plotly figure (pan, zoom, hover)
- `flight_path.png`  — static PNG (kaleido; matplotlib fallback)

## Physics model

| Quantity | Formula |
|---|---|
| Rate of turn | `ROT (deg/s) = 1091 × tan(bank) / TAS(knots)` |
| Ground speed | `v_ground = v_airspeed_vector + v_wind_vector` |
| G-load | `n = 1 / cos(bank_angle)` |
| Position | Euler integration at `dt = 0.01 s` |

Rate of turn uses **true airspeed**, not ground speed.
Ground speed is the **vector sum** of airspeed and wind (not subtraction).

## Configuration

Edit the `UAVConfig` constructor arguments in `main()`:

| Parameter | Default | Units | Description |
|---|---|---|---|
| `map_x_km` | 1.0 | km | Survey area east-west extent |
| `map_y_km` | 1.0 | km | Survey area north-south extent |
| `airspeed_knots` | 20.0 | knots | True airspeed (TAS) |
| `max_g` | 2.0 | G | Structural/comfort G-load limit |
| `bank_angle_deg` | 60.0 | deg | Coordinated turn bank angle |
| `wind_speed_mph` | 5.0 | mph | Wind speed |
| `wind_dir_deg` | 120.0 | deg | Wind direction (toward, from East CCW) |
| `swath_width_ft` | 100.0 | ft | Camera footprint swath width |

`UAVConfig.__init__` raises `ValueError` if `bank_angle_deg` exceeds `max_g`.

## Known limitations

- **Strip alignment**: Turn diameter `2r` and `swath_width_ft` may differ.
  If `2r < swath_width`, strips are closer together than one swath width after
  the turn; a repositioning segment (not yet implemented) is needed for exact
  coverage alignment.
- **Flat earth**: No terrain following or altitude variation modelled.
- **Wind constant**: Wind is modelled as uniform and steady over the survey area.

## Files

| File | Purpose |
|---|---|
| `FlightPathGen.py` | Path generation, physics model, Plotly/matplotlib visualisation |
| `Post.py` | Post-processing stub (planned: GPS log alignment, coverage analysis) |
| `requirements.txt` | Python dependencies |

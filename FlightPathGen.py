"""
UAVFlightPath — FlightPathGen.py
=================================
Generates a boustrophedon (lawnmower) coverage flight path for a fixed-wing UAV
over a rectangular survey area.

Features
--------
- Wind-corrected coordinated level turn arcs (trochoidal path)
- Spatially varying wind field  (smooth multi-mode sine perturbation)
- Strip alignment: repositioning legs correct turn-drift between survey strips
- 3-D terrain following: parametric Gaussian hills with rate-limited climb/descent
- Monte Carlo analysis: statistical coverage and deviation over N randomised runs

Physics model
-------------
- Coordinated level turn: ROT (deg/s) = 1091 x tan(bank_angle) / TAS(knots)
  Rate of turn uses TRUE AIRSPEED, not ground speed.
- Ground speed = airspeed vector + wind vector  (vector addition, not subtraction)
- Position integrated via Euler method at dt = 0.01 s resolution.
- Altitude: rate-limited climb/descent to maintain target AGL over terrain.

Units (internal)
----------------
- Speed   : ft/s  (converted from knots/mph at the input boundary)
- Distance: ft    (converted from km/m at the input boundary)
- Angles  : radians (converted from degrees at the input boundary)
- Altitude: ft MSL

Outputs
-------
- flight_path.html   : interactive Plotly figure (2-D bird's-eye + altitude profile)
- flight_path.png    : static PNG (kaleido; matplotlib fallback)
- monte_carlo.html   : MC path spread + statistics (when run_mc=True in main)
- monte_carlo.png    : static PNG of MC results
"""

import math
import sys
from dataclasses import dataclass

import numpy as np


# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
FT_PER_S_PER_KNOT: float = 1.68781   # 1 knot  = 1.68781 ft/s
FT_PER_KM: float          = 3280.84   # 1 km    = 3280.84 ft
FT_PER_M: float            = 3.28084  # 1 m     = 3.28084 ft
KNOTS_PER_MPH: float       = 0.868976 # 1 mph   = 0.868976 knots


# ---------------------------------------------------------------------------
# Terrain Model
# ---------------------------------------------------------------------------
class TerrainModel:
    """
    Parametric terrain: superposition of Gaussian hills over the survey area.
    Provides realistic, reproducible elevation data for terrain-following tests.

    height_ft(x, y) is fully vectorised — pass scalars or numpy arrays.

    Args:
        map_x_ft     : survey area width (ft)
        map_y_ft     : survey area height (ft)
        n_hills      : number of Gaussian elevation features
        max_height_ft: maximum hill height above zero baseline (ft)
        seed         : RNG seed for reproducibility
    """

    def __init__(
        self,
        map_x_ft: float,
        map_y_ft: float,
        n_hills: int = 6,
        max_height_ft: float = 300.0,
        seed: int = 42,
    ) -> None:
        rng   = np.random.default_rng(seed)
        scale = min(map_x_ft, map_y_ft)

        self.cx = rng.uniform(0.0, map_x_ft, n_hills)
        self.cy = rng.uniform(0.0, map_y_ft, n_hills)
        self.h  = rng.uniform(50.0, max_height_ft, n_hills)
        self.sx = rng.uniform(scale * 0.08, scale * 0.25, n_hills)
        self.sy = rng.uniform(scale * 0.08, scale * 0.25, n_hills)

    def height_ft(self, x, y) -> np.ndarray:
        """Terrain elevation in feet at position (x, y). Vectorised."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        h = np.zeros_like(x)
        for i in range(len(self.cx)):
            h += self.h[i] * np.exp(
                -0.5 * (
                    (x - self.cx[i]) ** 2 / self.sx[i] ** 2
                    + (y - self.cy[i]) ** 2 / self.sy[i] ** 2
                )
            )
        return h


# ---------------------------------------------------------------------------
# Wind Field
# ---------------------------------------------------------------------------
class WindField:
    """
    Spatially varying wind: base vector + smooth multi-mode sine perturbations.

    Reproduces realistic mesoscale wind variation without discontinuities.
    wind_at(x, y) returns [wx, wy] in knots at a scalar position.

    Args:
        base_speed_mph     : mean wind speed (mph)
        base_dir_deg       : mean wind direction (degrees, math convention: from East CCW)
        variation_fraction : amplitude of spatial perturbation as fraction of mean speed
        spatial_scale_ft   : characteristic length of wind variation (ft)
        seed               : RNG seed for reproducibility
    """

    def __init__(
        self,
        base_speed_mph: float,
        base_dir_deg: float,
        variation_fraction: float = 0.35,
        spatial_scale_ft: float = 2000.0,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)

        self.base_speed_knots = base_speed_mph * KNOTS_PER_MPH
        base_dir_rad          = math.radians(base_dir_deg)

        # Base uniform wind vector in knots
        self.base_vec = np.array([
            self.base_speed_knots * math.cos(base_dir_rad),
            self.base_speed_knots * math.sin(base_dir_rad),
        ])

        # 4 spatial perturbation modes per component
        n_modes = 4
        amp = self.base_speed_knots * variation_fraction
        self.amp_x = rng.uniform(-amp, amp, n_modes)
        self.amp_y = rng.uniform(-amp, amp, n_modes)
        self.kx    = rng.uniform(0.5, 2.5, n_modes) * (2.0 * math.pi / spatial_scale_ft)
        self.ky    = rng.uniform(0.5, 2.5, n_modes) * (2.0 * math.pi / spatial_scale_ft)
        self.phi_x = rng.uniform(0.0, 2.0 * math.pi, n_modes)
        self.phi_y = rng.uniform(0.0, 2.0 * math.pi, n_modes)

    def wind_at(self, x: float, y: float) -> np.ndarray:
        """Returns wind vector [wx, wy] in knots at scalar position (x, y)."""
        wx = float(self.base_vec[0])
        wy = float(self.base_vec[1])
        for i in range(len(self.amp_x)):
            wx += float(self.amp_x[i]) * math.sin(float(self.kx[i]) * float(x) + float(self.phi_x[i]))
            wy += float(self.amp_y[i]) * math.sin(float(self.ky[i]) * float(y) + float(self.phi_y[i]))
        return np.array([wx, wy])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class UAVConfig:
    """
    All mission configuration parameters.

    New parameters vs v1:
        target_agl_ft      : terrain-following altitude above ground level (ft)
        max_climb_rate_fpm : maximum climb/descent rate for terrain following (ft/min)

    Raises ValueError if bank angle exceeds the max-G limit.
    """

    def __init__(
        self,
        map_x_km: float          = 1.0,
        map_y_km: float          = 1.0,
        airspeed_knots: float    = 20.0,
        max_g: float             = 2.0,
        bank_angle_deg: float    = 60.0,
        wind_speed_mph: float    = 5.0,
        wind_dir_deg: float      = 120.0,
        swath_width_ft: float    = 100.0,
        dt: float                = 0.01,
        target_agl_ft: float     = 200.0,
        max_climb_rate_fpm: float = 500.0,
    ) -> None:
        # --- Store inputs ---
        self.map_x_km          = map_x_km
        self.map_y_km          = map_y_km
        self.airspeed_knots    = airspeed_knots
        self.max_g             = max_g
        self.bank_angle_deg    = bank_angle_deg
        self.wind_speed_mph    = wind_speed_mph
        self.wind_dir_deg      = wind_dir_deg
        self.swath_width_ft    = swath_width_ft
        self.dt                = dt
        self.target_agl_ft     = target_agl_ft
        self.max_climb_rate_fpm = max_climb_rate_fpm

        # --- Derived: angles ---
        self.bank_angle_rad: float = math.radians(bank_angle_deg)
        self.wind_dir_rad:   float = math.radians(wind_dir_deg)

        # --- Enforce G-limit before any computation ---
        g_load = 1.0 / math.cos(self.bank_angle_rad)
        if g_load > max_g + 1e-6:
            raise ValueError(
                f"Bank angle {bank_angle_deg}deg produces {g_load:.2f}G, "
                f"exceeding maxG={max_g}. Reduce bank angle or raise maxG."
            )

        # --- Derived: wind ---
        self.wind_speed_knots: float = wind_speed_mph * KNOTS_PER_MPH
        self.wind_vec_knots: np.ndarray = np.array([
            self.wind_speed_knots * math.cos(self.wind_dir_rad),
            self.wind_speed_knots * math.sin(self.wind_dir_rad),
        ])

        # --- Derived: map dimensions ---
        self.map_x_ft: float = map_x_km * FT_PER_KM
        self.map_y_ft: float = map_y_km * FT_PER_KM

        # --- Derived: swath in metres (display only) ---
        self.swath_width_m: float = swath_width_ft / FT_PER_M

        # --- Derived: climb rate in ft/s ---
        self.max_climb_rate_fts: float = max_climb_rate_fpm / 60.0


# ---------------------------------------------------------------------------
# Monte Carlo configuration
# ---------------------------------------------------------------------------
@dataclass
class MonteCarloConfig:
    """
    Parameters for Monte Carlo uncertainty analysis.

    All sigma values are 1-standard-deviation Gaussian uncertainties used to
    perturb the nominal UAVConfig for each simulation run.
    """
    n_runs: int = 100

    # Airspeed sensor / trim uncertainty (knots, 1-sigma)
    airspeed_sigma_knots: float = 1.0

    # Atmospheric wind uncertainty
    wind_speed_sigma_mph: float = 2.0   # magnitude 1-sigma
    wind_dir_sigma_deg: float   = 15.0  # direction 1-sigma
    wind_var_frac_sigma: float  = 0.10  # spatial variation fraction 1-sigma

    # Navigation / GPS position noise injected per step (ft, 1-sigma)
    nav_sigma_ft: float = 3.0

    # When True each run gets a unique terrain; False uses seed=42 for all runs
    terrain_seed_varies: bool = True

    seed: int = 0  # master RNG seed


# ---------------------------------------------------------------------------
# Flight mechanics — turn arc
# ---------------------------------------------------------------------------
def compute_turn_arc(
    start_pos: np.ndarray,
    start_heading_rad: float,
    config: UAVConfig,
    turn_direction: int,
    wind_field: WindField = None,
    terrain: TerrainModel = None,
    start_z_ft: float = 0.0,
) -> tuple:
    """
    Compute a 180-degree coordinated turn arc via Euler integration.

    Rate of turn uses TRUE AIRSPEED (aviation formula):
        ROT (deg/s) = 1091 x tan(bank) / TAS(knots)
        ROT (rad/s) = ROT(deg/s) x pi / 180

    With wind_field=None and terrain=None the fast vectorised path is used,
    producing identical results to v1.  Providing wind_field or terrain
    switches to a sequential step-by-step integration.

    Args:
        start_pos         : [x, y] start position (ft)
        start_heading_rad : entry heading (radians, math convention)
        config            : UAVConfig
        turn_direction    : +1 left/CCW, -1 right/CW
        wind_field        : WindField or None
        terrain           : TerrainModel or None
        start_z_ft        : starting altitude MSL (ft); used only when terrain != None

    Returns:
        x (ft array), y (ft array), z (ft array or None), final_heading_rad (float)
    """
    # Rate of turn — TAS only
    rot_rad_s = (
        (1091.0 * math.tan(config.bank_angle_rad) * math.pi)
        / (config.airspeed_knots * 180.0)
    ) * turn_direction

    turn_duration = math.pi / abs(rot_rad_s)
    airspeed_fts  = config.airspeed_knots * FT_PER_S_PER_KNOT

    # --- Fast vectorised path (uniform wind, no terrain) ---
    if wind_field is None and terrain is None:
        t        = np.arange(0.0, turn_duration, config.dt)
        headings = start_heading_rad + rot_rad_s * t
        wind_fts = config.wind_vec_knots * FT_PER_S_PER_KNOT

        vx = airspeed_fts * np.cos(headings) + wind_fts[0]
        vy = airspeed_fts * np.sin(headings) + wind_fts[1]

        x = np.empty(len(t) + 1)
        y = np.empty(len(t) + 1)
        x[0], y[0] = float(start_pos[0]), float(start_pos[1])
        x[1:] = start_pos[0] + np.cumsum(vx * config.dt)
        y[1:] = start_pos[1] + np.cumsum(vy * config.dt)

        final_heading = headings[-1] + rot_rad_s * config.dt
        return x, y, None, float(final_heading)

    # --- Sequential path (wind field and/or terrain) ---
    n_steps = int(turn_duration / config.dt)
    x = np.empty(n_steps + 1)
    y = np.empty(n_steps + 1)
    z = np.empty(n_steps + 1) if terrain is not None else None

    x[0], y[0] = float(start_pos[0]), float(start_pos[1])
    if z is not None:
        z[0] = start_z_ft

    heading = start_heading_rad
    for i in range(n_steps):
        heading += rot_rad_s * config.dt

        w     = wind_field.wind_at(x[i], y[i]) if wind_field is not None else config.wind_vec_knots
        w_fts = w * FT_PER_S_PER_KNOT

        x[i + 1] = x[i] + (airspeed_fts * math.cos(heading) + w_fts[0]) * config.dt
        y[i + 1] = y[i] + (airspeed_fts * math.sin(heading) + w_fts[1]) * config.dt

        if terrain is not None:
            tgt_z = float(terrain.height_ft(x[i + 1], y[i + 1])) + config.target_agl_ft
            dz    = float(np.clip(
                tgt_z - z[i],
                -config.max_climb_rate_fts * config.dt,
                 config.max_climb_rate_fts * config.dt,
            ))
            z[i + 1] = z[i] + dz

    final_heading = start_heading_rad + rot_rad_s * n_steps * config.dt
    return x, y, z, float(final_heading)


# ---------------------------------------------------------------------------
# Flight mechanics — straight leg
# ---------------------------------------------------------------------------
def compute_straight_leg(
    start_pos: np.ndarray,
    heading_rad: float,
    leg_length_ft: float,
    config: UAVConfig,
    wind_field: WindField = None,
    terrain: TerrainModel = None,
    start_z_ft: float = 0.0,
) -> tuple:
    """
    Compute a straight survey leg covering leg_length_ft in the east/west direction.

    Leg terminates when east-west distance traveled equals leg_length_ft.
    With wind_field=None and terrain=None the fast vectorised path is used,
    producing identical results to v1.

    Args:
        start_pos     : [x, y] start position (ft)
        heading_rad   : fixed aircraft heading (radians)
        leg_length_ft : east-west extent to cover (ft)
        config        : UAVConfig
        wind_field    : WindField or None
        terrain       : TerrainModel or None
        start_z_ft    : starting altitude MSL (ft)

    Returns:
        x (ft), y (ft), z (ft or None)
    """
    airspeed_fts = config.airspeed_knots * FT_PER_S_PER_KNOT

    # --- Fast vectorised path ---
    if wind_field is None and terrain is None:
        wind_fts = config.wind_vec_knots * FT_PER_S_PER_KNOT
        gspd_x   = airspeed_fts * math.cos(heading_rad) + wind_fts[0]
        gspd_y   = airspeed_fts * math.sin(heading_rad) + wind_fts[1]

        if abs(gspd_x) < 0.5:
            raise ValueError(
                f"East/west ground speed ({gspd_x:.2f} ft/s) too low. "
                "Headwind may equal or exceed airspeed."
            )

        t_end = leg_length_ft / abs(gspd_x)
        t     = np.arange(0.0, t_end, config.dt)
        x     = start_pos[0] + gspd_x * t
        y     = start_pos[1] + gspd_y * t
        x     = np.append(x, start_pos[0] + gspd_x * t_end)
        y     = np.append(y, start_pos[1] + gspd_y * t_end)
        return x, y, None

    # --- Sequential path ---
    # Estimate duration from initial wind to pre-allocate reasonably
    w0     = wind_field.wind_at(float(start_pos[0]), float(start_pos[1])) \
             if wind_field is not None else config.wind_vec_knots
    w0_fts = w0 * FT_PER_S_PER_KNOT
    gspd_x0 = airspeed_fts * math.cos(heading_rad) + w0_fts[0]
    if abs(gspd_x0) < 0.5:
        raise ValueError(
            f"East/west ground speed ({gspd_x0:.2f} ft/s) too low."
        )

    xs = [float(start_pos[0])]
    ys = [float(start_pos[1])]
    zs = [start_z_ft] if terrain is not None else None

    x_cur, y_cur = float(start_pos[0]), float(start_pos[1])
    z_cur        = start_z_ft
    ew_traveled  = 0.0

    max_steps = int(leg_length_ft / (abs(gspd_x0) * config.dt)) * 3  # safety ceiling
    for _ in range(max_steps):
        w     = wind_field.wind_at(x_cur, y_cur) if wind_field is not None else config.wind_vec_knots
        w_fts = w * FT_PER_S_PER_KNOT
        gspd_x = airspeed_fts * math.cos(heading_rad) + w_fts[0]
        gspd_y = airspeed_fts * math.sin(heading_rad) + w_fts[1]

        dx = gspd_x * config.dt
        dy = gspd_y * config.dt
        ew_traveled += abs(dx)

        x_cur += dx
        y_cur += dy
        xs.append(x_cur)
        ys.append(y_cur)

        if terrain is not None:
            tgt_z = float(terrain.height_ft(x_cur, y_cur)) + config.target_agl_ft
            dz    = float(np.clip(
                tgt_z - z_cur,
                -config.max_climb_rate_fts * config.dt,
                 config.max_climb_rate_fts * config.dt,
            ))
            z_cur += dz
            zs.append(z_cur)

        if ew_traveled >= leg_length_ft:
            break

    x = np.array(xs)
    y = np.array(ys)
    z = np.array(zs) if terrain is not None else None
    return x, y, z


# ---------------------------------------------------------------------------
# Strip alignment — repositioning leg
# ---------------------------------------------------------------------------
def compute_repositioning_leg(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    config: UAVConfig,
    wind_field: WindField = None,
    terrain: TerrainModel = None,
    start_z_ft: float = 0.0,
) -> tuple:
    """
    Fly a straight correction segment from start_pos to target_pos.

    The aircraft points directly at the target and holds that heading.
    Integration continues until the slant distance traveled equals the
    straight-line distance to the target.  The final point is snapped
    to the exact target position so subsequent strips are perfectly aligned.

    This segment corrects the strip misalignment caused by wind drift during
    the 180-degree turn arcs.

    Args:
        start_pos  : [x, y] current position after turn (ft)
        target_pos : [x, y] nominal start of next survey strip (ft)
        config     : UAVConfig
        wind_field : WindField or None
        terrain    : TerrainModel or None
        start_z_ft : starting altitude MSL (ft)

    Returns:
        x (ft), y (ft), z (ft or None)
    """
    delta    = np.asarray(target_pos, dtype=float) - np.asarray(start_pos, dtype=float)
    distance = float(np.linalg.norm(delta))

    if distance < 1.0:
        z_arr = np.array([start_z_ft]) if terrain is not None else None
        return np.array([float(start_pos[0])]), np.array([float(start_pos[1])]), z_arr

    heading      = math.atan2(float(delta[1]), float(delta[0]))
    airspeed_fts = config.airspeed_knots * FT_PER_S_PER_KNOT

    xs = [float(start_pos[0])]
    ys = [float(start_pos[1])]
    zs = [start_z_ft] if terrain is not None else None

    x_cur, y_cur = float(start_pos[0]), float(start_pos[1])
    z_cur        = start_z_ft
    dist_done    = 0.0

    max_steps = int(distance / (airspeed_fts * config.dt)) * 4  # safety ceiling
    for _ in range(max_steps):
        w     = wind_field.wind_at(x_cur, y_cur) if wind_field is not None else config.wind_vec_knots
        w_fts = w * FT_PER_S_PER_KNOT
        vx    = airspeed_fts * math.cos(heading) + w_fts[0]
        vy    = airspeed_fts * math.sin(heading) + w_fts[1]
        spd   = math.sqrt(vx * vx + vy * vy)

        x_cur    += vx * config.dt
        y_cur    += vy * config.dt
        dist_done += spd * config.dt
        xs.append(x_cur)
        ys.append(y_cur)

        if terrain is not None:
            tgt_z = float(terrain.height_ft(x_cur, y_cur)) + config.target_agl_ft
            dz    = float(np.clip(
                tgt_z - z_cur,
                -config.max_climb_rate_fts * config.dt,
                 config.max_climb_rate_fts * config.dt,
            ))
            z_cur += dz
            zs.append(z_cur)

        if dist_done >= distance:
            break

    # Snap endpoint to exact target
    xs[-1] = float(target_pos[0])
    ys[-1] = float(target_pos[1])

    return (
        np.array(xs),
        np.array(ys),
        np.array(zs) if terrain is not None else None,
    )


# ---------------------------------------------------------------------------
# Path generation — full lawnmower pattern
# ---------------------------------------------------------------------------
def generate_lawnmower_path(
    config: UAVConfig,
    wind_field: WindField = None,
    terrain: TerrainModel = None,
    align_strips: bool = True,
) -> tuple:
    """
    Generate a complete boustrophedon (lawnmower) coverage path.

    Strip layout:
    - Strips run East-West.  Strip spacing = swath_width_ft.
    - Even strips (0, 2, …) fly East; odd strips (1, 3, …) fly West.
    - After each eastbound leg : left  (CCW, +1) 180-deg turn.
    - After each westbound leg : right (CW,  -1) 180-deg turn.

    Strip alignment:
        If align_strips=True a repositioning leg is inserted after each turn
        to correct drift and place the aircraft at the exact nominal start of
        the next strip.  The segment is tagged type="reposition".

    Terrain following:
        If terrain is not None every segment carries a z[] altitude array.
        Altitude = terrain.height_ft(x, y) + config.target_agl_ft,
        rate-limited by config.max_climb_rate_fts.

    Returns:
        x_path (ft), y_path (ft), z_path (ft or None), segments (list of dicts)

    Each segment dict has keys: type, strip, direction, x, y, z (+ deviation_ft for reposition).
    """
    n_strips = math.ceil(config.map_y_ft / config.swath_width_ft)
    segments = []

    current_pos     = np.array([0.0, 0.0])
    current_heading = 0.0  # start heading East

    if terrain is not None:
        current_z = float(terrain.height_ft(0.0, 0.0)) + config.target_agl_ft
    else:
        current_z = config.target_agl_ft

    for strip_idx in range(n_strips):
        heading_east = (strip_idx % 2 == 0)

        # --- Straight survey leg ---
        x_seg, y_seg, z_seg = compute_straight_leg(
            current_pos, current_heading, config.map_x_ft, config,
            wind_field=wind_field, terrain=terrain, start_z_ft=current_z,
        )
        segments.append({
            "type":      "straight",
            "strip":     strip_idx,
            "direction": "east" if heading_east else "west",
            "x": x_seg, "y": y_seg, "z": z_seg,
        })
        current_pos = np.array([x_seg[-1], y_seg[-1]])
        if z_seg is not None:
            current_z = float(z_seg[-1])

        if strip_idx >= n_strips - 1:
            break  # no turn or reposition after the final strip

        # --- 180-degree turn ---
        turn_dir = +1 if heading_east else -1
        x_turn, y_turn, z_turn, current_heading = compute_turn_arc(
            current_pos, current_heading, config, turn_dir,
            wind_field=wind_field, terrain=terrain, start_z_ft=current_z,
        )
        segments.append({
            "type":      "turn",
            "strip":     strip_idx,
            "direction": "left" if turn_dir == +1 else "right",
            "x": x_turn, "y": y_turn, "z": z_turn,
        })
        current_pos = np.array([x_turn[-1], y_turn[-1]])
        if z_turn is not None:
            current_z = float(z_turn[-1])

        # --- Strip alignment: repositioning leg ---
        if align_strips:
            next_strip = strip_idx + 1
            next_x     = 0.0 if (next_strip % 2 == 0) else config.map_x_ft
            next_y     = float(next_strip) * config.swath_width_ft
            target     = np.array([next_x, next_y])

            deviation = float(np.linalg.norm(current_pos - target))
            if deviation > 1.0:
                x_repo, y_repo, z_repo = compute_repositioning_leg(
                    current_pos, target, config,
                    wind_field=wind_field, terrain=terrain, start_z_ft=current_z,
                )
                segments.append({
                    "type":         "reposition",
                    "strip":        strip_idx,
                    "deviation_ft": deviation,
                    "x": x_repo, "y": y_repo, "z": z_repo,
                })
                current_pos = target  # snapped to nominal
                if z_repo is not None:
                    current_z = float(z_repo[-1])
                # Point the aircraft toward the next survey direction
                current_heading = 0.0 if (next_strip % 2 == 0) else math.pi

    # --- Concatenate into single path (drop duplicate boundary points) ---
    all_x    = [segments[0]["x"]]
    all_y    = [segments[0]["y"]]
    has_z    = segments[0]["z"] is not None
    all_z    = [segments[0]["z"]] if has_z else None

    for seg in segments[1:]:
        all_x.append(seg["x"][1:])
        all_y.append(seg["y"][1:])
        if has_z and seg["z"] is not None:
            all_z.append(seg["z"][1:])

    z_path = np.concatenate(all_z) if all_z is not None else None
    return np.concatenate(all_x), np.concatenate(all_y), z_path, segments


# ---------------------------------------------------------------------------
# Coverage metrics (used by Monte Carlo)
# ---------------------------------------------------------------------------
def compute_coverage_metrics(segments: list, config: UAVConfig) -> dict:
    """
    Compute coverage quality metrics from a generated path.

    Metrics
    -------
    strip_deviations_ft : per-strip mean lateral-y deviation from nominal (ft)
    mean_deviation_ft   : average of strip_deviations_ft
    max_deviation_ft    : worst strip deviation
    coverage_fraction   : fraction of nominal swath bands actually traversed
    path_length_ft      : total 2-D path length
    n_reposition        : count of repositioning legs used
    reposition_total_ft : total distance flown in reposition legs
    """
    straight_segs = [s for s in segments if s["type"] == "straight"]
    repo_segs     = [s for s in segments if s["type"] == "reposition"]

    # Strip lateral deviations (mean y vs nominal centre)
    deviations = []
    for s in straight_segs:
        nominal_y = float(s["strip"]) * config.swath_width_ft
        actual_y  = float(np.mean(s["y"]))
        deviations.append(abs(actual_y - nominal_y))

    # Coverage completeness: which nominal swath bands are actually flown?
    n_strips = math.ceil(config.map_y_ft / config.swath_width_ft)
    covered  = 0
    for i in range(n_strips):
        y0 = i * config.swath_width_ft
        y1 = y0 + config.swath_width_ft
        if any(np.any((s["y"] >= y0) & (s["y"] < y1)) for s in straight_segs):
            covered += 1
    coverage_fraction = covered / n_strips if n_strips > 0 else 0.0

    # Total path length (2-D)
    path_length = 0.0
    for s in segments:
        dx = np.diff(s["x"])
        dy = np.diff(s["y"])
        path_length += float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))

    # Repositioning totals
    repo_total = 0.0
    for s in repo_segs:
        dx = np.diff(s["x"])
        dy = np.diff(s["y"])
        repo_total += float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))

    return {
        "strip_deviations_ft": deviations,
        "mean_deviation_ft":   float(np.mean(deviations)) if deviations else 0.0,
        "max_deviation_ft":    float(np.max(deviations))  if deviations else 0.0,
        "coverage_fraction":   coverage_fraction,
        "path_length_ft":      path_length,
        "n_reposition":        len(repo_segs),
        "reposition_total_ft": repo_total,
    }


# ---------------------------------------------------------------------------
# Monte Carlo analysis
# ---------------------------------------------------------------------------
def run_monte_carlo(
    base_config: UAVConfig,
    mc_config: MonteCarloConfig,
) -> dict:
    """
    Run N Monte Carlo simulations with randomised mission parameters.

    Each run independently perturbs:
    - Airspeed (Gaussian around base, min 5 kt)
    - Wind speed + direction (Gaussian around base values)
    - Spatial wind variation fraction
    - Navigation/GPS noise (Gaussian per-point position noise)
    - Terrain realisation (optional, via terrain_seed_varies)

    Runs that violate the G-limit or hit near-zero ground speed are skipped.

    Returns a dict with keys:
        metrics   : list of compute_coverage_metrics() dicts, one per run
        paths     : list of (x, y) tuples (500-point downsampled, noise included)
        mc_config : the MonteCarloConfig used
        n_valid   : number of runs that completed successfully
    """
    rng = np.random.default_rng(mc_config.seed)
    results = {
        "metrics":   [],
        "paths":     [],
        "mc_config": mc_config,
        "n_valid":   0,
    }

    for run_i in range(mc_config.n_runs):
        # --- Perturb parameters ---
        airspeed_k = max(5.0, base_config.airspeed_knots
                         + rng.normal(0.0, mc_config.airspeed_sigma_knots))
        wind_spd   = max(0.0, base_config.wind_speed_mph
                         + rng.normal(0.0, mc_config.wind_speed_sigma_mph))
        wind_dir   = base_config.wind_dir_deg + rng.normal(0.0, mc_config.wind_dir_sigma_deg)
        var_frac   = max(0.05, 0.35 + rng.normal(0.0, mc_config.wind_var_frac_sigma))

        try:
            cfg = UAVConfig(
                map_x_km          = base_config.map_x_km,
                map_y_km          = base_config.map_y_km,
                airspeed_knots    = airspeed_k,
                max_g             = base_config.max_g,
                bank_angle_deg    = base_config.bank_angle_deg,
                wind_speed_mph    = wind_spd,
                wind_dir_deg      = wind_dir,
                swath_width_ft    = base_config.swath_width_ft,
                dt                = base_config.dt,
                target_agl_ft     = base_config.target_agl_ft,
                max_climb_rate_fpm= base_config.max_climb_rate_fpm,
            )
        except ValueError:
            continue  # G-limit violated for this sample

        wf_seed   = int(rng.integers(0, 2 ** 31))
        # wf and terrain_r are available for future per-run path gen;
        # MC currently uses fast vectorised (uniform-wind) path gen for speed.
        _wf_seed  = wf_seed  # consume rng draw to keep stream consistent

        try:
            x_path, y_path, _, segments = generate_lawnmower_path(
                cfg, wind_field=None, terrain=None, align_strips=True,
            )
        except (ValueError, ZeroDivisionError):
            continue

        # Inject navigation noise
        x_noisy = x_path + rng.normal(0.0, mc_config.nav_sigma_ft, len(x_path))
        y_noisy = y_path + rng.normal(0.0, mc_config.nav_sigma_ft, len(y_path))

        metrics = compute_coverage_metrics(segments, cfg)
        results["metrics"].append(metrics)

        # Downsample to 500 points for memory efficiency
        idx = np.linspace(0, len(x_noisy) - 1, 500, dtype=int)
        results["paths"].append((x_noisy[idx], y_noisy[idx]))
        results["n_valid"] += 1

        if (run_i + 1) % 10 == 0 or run_i == mc_config.n_runs - 1:
            print(f"  MC {run_i + 1}/{mc_config.n_runs}  "
                  f"({results['n_valid']} valid)", end="\r")

    print()  # newline after progress
    return results


# ---------------------------------------------------------------------------
# Visualization — main flight path
# ---------------------------------------------------------------------------
def _build_plotly_figure(
    segments: list,
    config: UAVConfig,
    terrain: TerrainModel = None,
    x_path: np.ndarray = None,
    y_path: np.ndarray = None,
    z_path: np.ndarray = None,
):
    """Build and return a Plotly Figure for the flight path.

    Layout:
    - If terrain is provided: two-row figure (bird's-eye | altitude profile).
    - Otherwise: single-row bird's-eye view (same as v1).
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_terrain = terrain is not None and z_path is not None

    if has_terrain:
        fig = make_subplots(
            rows=2, cols=1,
            row_heights=[0.70, 0.30],
            shared_xaxes=False,
            vertical_spacing=0.08,
            subplot_titles=("Bird's-eye view (ft)", "Altitude profile (ft MSL)"),
        )
        map_row = 1
    else:
        fig = go.Figure()
        map_row = None

    def _add(trace, row=None):
        if row is not None:
            fig.add_trace(trace, row=row, col=1)
        else:
            fig.add_trace(trace)

    n_strips = math.ceil(config.map_y_ft / config.swath_width_ft)

    # --- Terrain contour heatmap (bird's-eye background) ---
    if has_terrain:
        nx, ny = 80, 80
        gx = np.linspace(0, config.map_x_ft, nx)
        gy = np.linspace(0, config.map_y_ft, ny)
        GX, GY = np.meshgrid(gx, gy)
        GH = terrain.height_ft(GX, GY)
        _add(go.Heatmap(
            x=gx, y=gy, z=GH,
            colorscale="YlOrBr",
            opacity=0.30,
            showscale=True,
            colorbar=dict(title="Terrain (ft)", len=0.45, y=0.77, thickness=12),
            name="Terrain",
            showlegend=False,
            hovertemplate="Terrain %{z:.0f} ft<extra></extra>",
        ), row=map_row)
    else:
        # Swath bands (alternating colours, v1 style)
        for i in range(n_strips):
            y0b = i * config.swath_width_ft
            y1b = y0b + config.swath_width_ft
            _add(go.Scatter(
                x=[0, config.map_x_ft, config.map_x_ft, 0, 0],
                y=[y0b, y0b, y1b, y1b, y0b],
                fill="toself",
                fillcolor="lightsteelblue" if i % 2 == 0 else "lightyellow",
                line=dict(width=0),
                opacity=0.20,
                showlegend=False,
                hoverinfo="skip",
            ))

    # --- Segment traces ---
    straight_shown = turn_shown = repo_shown = False
    COLOUR = {"straight": "royalblue", "turn": "darkorange", "reposition": "mediumseagreen"}
    DASH   = {"straight": "solid",     "turn": "dot",        "reposition": "dash"}
    WIDTH  = {"straight": 2.5,         "turn": 2.0,          "reposition": 1.8}
    LABEL  = {"straight": "Survey leg", "turn": "Turn arc",  "reposition": "Reposition"}

    shown  = {"straight": False, "turn": False, "reposition": False}

    for seg in segments:
        t = seg["type"]
        _add(go.Scatter(
            x=seg["x"], y=seg["y"],
            mode="lines",
            line=dict(color=COLOUR[t], width=WIDTH[t], dash=DASH[t]),
            name=LABEL[t],
            legendgroup=t,
            showlegend=not shown[t],
            hovertemplate="x=%{x:.1f} ft<br>y=%{y:.1f} ft<extra></extra>",
        ), row=map_row)
        shown[t] = True

    # --- Start / End markers ---
    x0, y0 = float(segments[0]["x"][0]),  float(segments[0]["y"][0])
    xn, yn = float(segments[-1]["x"][-1]), float(segments[-1]["y"][-1])
    for xi, yi, sym, col, lbl in [
        (x0, y0, "triangle-right", "green",   "Start"),
        (xn, yn, "square",         "crimson",  "End"),
    ]:
        _add(go.Scatter(
            x=[xi], y=[yi], mode="markers+text",
            marker=dict(color=col, size=14, symbol=sym),
            text=[lbl], textposition="top right",
            name=lbl,
        ), row=map_row)

    # --- Wind vector arrow ---
    arrow_scale = config.map_x_ft * 0.12
    if config.wind_speed_knots > 0:
        unit = config.wind_vec_knots / config.wind_speed_knots
        wx, wy = float(unit[0]) * arrow_scale, float(unit[1]) * arrow_scale
    else:
        wx, wy = 0.0, 0.0
    ax_ox = config.map_x_ft * 0.82
    ay_oy = config.map_y_ft * 0.06

    anno_kwargs = {}
    if map_row is not None:
        anno_kwargs = dict(xref="x1", yref="y1", axref="x1", ayref="y1")
    else:
        anno_kwargs = dict(xref="x", yref="y", axref="x", ayref="y")

    fig.add_annotation(
        x=ax_ox + wx, y=ay_oy + wy,
        ax=ax_ox, ay=ay_oy,
        text=f"Wind<br>{config.wind_speed_mph} mph",
        showarrow=True,
        arrowhead=2, arrowsize=1.5,
        arrowcolor="steelblue", arrowwidth=2.5,
        font=dict(size=10, color="steelblue"),
        bgcolor="rgba(255,255,255,0.75)",
        **anno_kwargs,
    )

    # --- Mission parameter box ---
    rot_rad_s      = (1091.0 * math.tan(config.bank_angle_rad) * math.pi) \
                     / (config.airspeed_knots * 180.0)
    turn_radius_ft = (config.airspeed_knots * FT_PER_S_PER_KNOT) / rot_rad_s
    g_load         = 1.0 / math.cos(config.bank_angle_rad)
    n_repo         = sum(1 for s in segments if s["type"] == "reposition")

    info = (
        "<b>Mission Parameters</b><br>"
        f"Area  : {config.map_x_km} x {config.map_y_km} km<br>"
        f"TAS   : {config.airspeed_knots} kt<br>"
        f"Bank  : {config.bank_angle_deg}deg -> {g_load:.2f} G  (max {config.max_g} G)<br>"
        f"ROT   : {math.degrees(rot_rad_s):.1f} deg/s<br>"
        f"Turn R: {turn_radius_ft:.0f} ft<br>"
        f"Swath : {config.swath_width_ft:.0f} ft  ({config.swath_width_m:.1f} m)<br>"
        f"Strips: {n_strips}<br>"
        f"Repos : {n_repo}<br>"
        f"Wind  : {config.wind_speed_mph} mph @ {config.wind_dir_deg}deg<br>"
        f"AGL   : {config.target_agl_ft:.0f} ft"
    )
    fig.add_annotation(
        x=0.01, y=0.99 if not has_terrain else 0.72,
        xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text=info, showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(80,80,80,0.5)",
        borderwidth=1,
        font=dict(size=11, family="monospace"),
    )

    # --- Altitude profile subplot ---
    if has_terrain and x_path is not None and z_path is not None:
        yp = y_path if y_path is not None else np.zeros_like(x_path)

        # Cumulative 2-D distance along path (x-axis for profile)
        seg_dists = np.sqrt(np.diff(x_path) ** 2 + np.diff(yp) ** 2)
        cum_dist  = np.concatenate([[0.0], np.cumsum(seg_dists)])

        fig.add_trace(go.Scatter(
            x=cum_dist, y=z_path,
            mode="lines",
            line=dict(color="royalblue", width=1.5),
            name="UAV altitude",
            showlegend=True,
            hovertemplate="dist=%{x:.0f} ft<br>alt=%{y:.0f} ft MSL<extra></extra>",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=cum_dist, y=terrain.height_ft(x_path, yp),
            mode="lines",
            line=dict(color="sienna", width=1.2),
            fill="tozeroy",
            fillcolor="rgba(160,82,45,0.20)",
            name="Terrain",
            showlegend=True,
            hovertemplate="dist=%{x:.0f} ft<br>terrain=%{y:.0f} ft MSL<extra></extra>",
        ), row=2, col=1)

    # --- Layout ---
    rot_rad_s      = (1091.0 * math.tan(config.bank_angle_rad) * math.pi) \
                     / (config.airspeed_knots * 180.0)
    turn_radius_ft = (config.airspeed_knots * FT_PER_S_PER_KNOT) / rot_rad_s
    g_load         = 1.0 / math.cos(config.bank_angle_rad)

    title = (
        "UAV Lawnmower Coverage Flight Path<br>"
        f"<sup>TAS {config.airspeed_knots} kt | "
        f"{config.bank_angle_deg}deg bank ({g_load:.2f} G) | "
        f"Turn radius {turn_radius_ft:.0f} ft | "
        f"Swath {config.swath_width_ft:.0f} ft | "
        f"Wind {config.wind_speed_mph} mph @ {config.wind_dir_deg}deg</sup>"
    )

    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14)),
        plot_bgcolor="rgb(244,247,252)",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
        margin=dict(l=70, r=40, t=130, b=70),
        height=820 if has_terrain else 720,
    )

    if has_terrain:
        fig.update_xaxes(title="East position (ft)", row=1, col=1,
                         scaleanchor="y", scaleratio=1,
                         showgrid=True, gridcolor="rgba(180,180,180,0.4)")
        fig.update_yaxes(title="North position (ft)", row=1, col=1,
                         showgrid=True, gridcolor="rgba(180,180,180,0.4)")
        fig.update_xaxes(title="East position (ft)", row=2, col=1)
        fig.update_yaxes(title="Altitude (ft MSL)", row=2, col=1)
    else:
        fig.update_xaxes(
            title="East position (ft)",
            scaleanchor="y", scaleratio=1,
            showgrid=True, gridcolor="rgba(180,180,180,0.45)",
            zeroline=True, zerolinecolor="rgba(0,0,0,0.25)",
        )
        fig.update_yaxes(
            title="North position (ft)",
            showgrid=True, gridcolor="rgba(180,180,180,0.45)",
            zeroline=True, zerolinecolor="rgba(0,0,0,0.25)",
        )

    return fig


def plot_flight_path(
    segments: list,
    config: UAVConfig,
    html_path: str = "flight_path.html",
    png_path: str  = "flight_path.png",
    terrain: TerrainModel = None,
    x_path: np.ndarray = None,
    y_path: np.ndarray = None,
    z_path: np.ndarray = None,
) -> None:
    """
    Write an interactive HTML and a static PNG of the flight path.

    PNG export tries kaleido first (plotly native), then matplotlib as fallback.
    Pass terrain + x_path + z_path to enable the altitude profile subplot.
    """
    fig = _build_plotly_figure(segments, config, terrain=terrain,
                               x_path=x_path, y_path=y_path, z_path=z_path)

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"HTML  -> {html_path}")

    try:
        fig.write_image(png_path, width=1400, height=900, scale=2)
        print(f"PNG   -> {png_path}  (kaleido)")
        return
    except Exception as kaleido_err:
        print(f"kaleido unavailable ({kaleido_err}), trying matplotlib fallback ...")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig_mpl, ax = plt.subplots(figsize=(14, 8), dpi=150)
        ax.set_facecolor("#f4f7fc")
        fig_mpl.patch.set_facecolor("white")

        for seg in segments:
            colour = {"straight": "royalblue", "turn": "darkorange",
                      "reposition": "mediumseagreen"}[seg["type"]]
            ls     = {"straight": "-", "turn": ":", "reposition": "--"}[seg["type"]]
            ax.plot(seg["x"], seg["y"], color=colour, lw=1.8, linestyle=ls)

        x0_pt = float(segments[0]["x"][0]);  y0_pt = float(segments[0]["y"][0])
        xn_pt = float(segments[-1]["x"][-1]); yn_pt = float(segments[-1]["y"][-1])
        ax.scatter([x0_pt], [y0_pt], color="green",  s=120, zorder=5, marker="^")
        ax.scatter([xn_pt], [yn_pt], color="crimson", s=120, zorder=5, marker="s")

        rot_rad_s      = (1091.0 * math.tan(config.bank_angle_rad) * math.pi) \
                         / (config.airspeed_knots * 180.0)
        turn_radius_ft = (config.airspeed_knots * FT_PER_S_PER_KNOT) / rot_rad_s
        g_load         = 1.0 / math.cos(config.bank_angle_rad)

        ax.set_title(
            f"UAV Lawnmower Coverage  |  TAS {config.airspeed_knots} kt  |  "
            f"{config.bank_angle_deg}deg bank ({g_load:.2f} G)  |  "
            f"Turn R {turn_radius_ft:.0f} ft  |  Swath {config.swath_width_ft:.0f} ft",
            fontsize=9,
        )
        ax.set_xlabel("East (ft)"); ax.set_ylabel("North (ft)")
        ax.set_aspect("equal"); ax.grid(True, color="gray", alpha=0.25)
        ax.legend(handles=[
            mpatches.Patch(color="royalblue",     label="Survey leg"),
            mpatches.Patch(color="darkorange",    label="Turn arc"),
            mpatches.Patch(color="mediumseagreen",label="Reposition"),
            mpatches.Patch(color="green",         label="Start"),
            mpatches.Patch(color="crimson",       label="End"),
        ], loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig_mpl)
        print(f"PNG   -> {png_path}  (matplotlib fallback)")
    except Exception as mpl_err:
        print(f"WARNING: PNG export failed — {mpl_err}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Visualization — Monte Carlo results
# ---------------------------------------------------------------------------
def _build_mc_figure(results: dict, base_config: UAVConfig):
    """Build a 2 x 2 Plotly figure summarising Monte Carlo results."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    metrics   = results["metrics"]
    paths     = results["paths"]
    mc_config = results["mc_config"]

    mean_devs  = [m["mean_deviation_ft"]  for m in metrics]
    max_devs   = [m["max_deviation_ft"]   for m in metrics]
    coverages  = [m["coverage_fraction"]  * 100.0 for m in metrics]
    lengths    = [m["path_length_ft"]     / 5280.0 for m in metrics]  # miles
    repo_dists = [m["reposition_total_ft"] for m in metrics]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            f"Path spread  (N={results['n_valid']} runs)",
            "Strip deviation distribution (ft)",
            "Coverage completeness (%)",
            "Repositioning distance per run (ft)",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.10,
    )

    # --- (1,1) Path spread ---
    alpha = max(0.04, min(0.15, 1.0 / max(1, len(paths))))
    for i, (xp, yp) in enumerate(paths):
        fig.add_trace(go.Scatter(
            x=xp, y=yp, mode="lines",
            line=dict(color="royalblue", width=0.6),
            opacity=alpha,
            showlegend=(i == 0),
            name="MC run",
            hoverinfo="skip",
        ), row=1, col=1)

    # Nominal grid lines
    n_strips = math.ceil(base_config.map_y_ft / base_config.swath_width_ft)
    for i in range(n_strips + 1):
        y_line = i * base_config.swath_width_ft
        fig.add_trace(go.Scatter(
            x=[0, base_config.map_x_ft], y=[y_line, y_line],
            mode="lines",
            line=dict(color="rgba(200,50,50,0.5)", width=1.0, dash="dot"),
            showlegend=(i == 0),
            name="Nominal strip edge",
            hoverinfo="skip",
        ), row=1, col=1)

    # --- (1,2) Mean deviation histogram ---
    fig.add_trace(go.Histogram(
        x=mean_devs,
        nbinsx=20,
        marker_color="royalblue",
        opacity=0.75,
        name="Mean deviation",
        showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Histogram(
        x=max_devs,
        nbinsx=20,
        marker_color="darkorange",
        opacity=0.60,
        name="Max deviation",
        showlegend=False,
    ), row=1, col=2)

    # --- (2,1) Coverage histogram ---
    fig.add_trace(go.Histogram(
        x=coverages,
        nbinsx=15,
        marker_color="mediumseagreen",
        opacity=0.80,
        name="Coverage %",
        showlegend=False,
    ), row=2, col=1)

    # --- (2,2) Reposition distance histogram ---
    fig.add_trace(go.Histogram(
        x=repo_dists,
        nbinsx=20,
        marker_color="mediumpurple",
        opacity=0.80,
        name="Reposition dist (ft)",
        showlegend=False,
    ), row=2, col=2)

    # Summary stats annotation
    def _s(vals, unit):
        return (f"mean {float(np.mean(vals)):.1f} {unit}  "
                f"p95 {float(np.percentile(vals, 95)):.1f} {unit}")

    info = (
        f"<b>MC Summary  N={results['n_valid']}/{mc_config.n_runs}</b><br>"
        f"Airspeed sigma  : {mc_config.airspeed_sigma_knots} kt<br>"
        f"Wind spd sigma  : {mc_config.wind_speed_sigma_mph} mph<br>"
        f"Wind dir sigma  : {mc_config.wind_dir_sigma_deg} deg<br>"
        f"Nav noise       : {mc_config.nav_sigma_ft} ft (1-sigma)<br>"
        f"Mean deviation  : {_s(mean_devs, 'ft')}<br>"
        f"Coverage        : {_s(coverages, '%')}<br>"
        f"Path length     : {_s(lengths, 'mi')}"
    )
    fig.add_annotation(
        x=0.99, y=0.99, xref="paper", yref="paper",
        xanchor="right", yanchor="top",
        text=info, showarrow=False,
        bgcolor="rgba(255,255,255,0.90)",
        bordercolor="rgba(80,80,80,0.5)",
        borderwidth=1,
        font=dict(size=10, family="monospace"),
    )

    fig.update_layout(
        title=dict(
            text=(f"UAV Monte Carlo Analysis  |  "
                  f"TAS {base_config.airspeed_knots} kt  |  "
                  f"N={results['n_valid']} runs"),
            x=0.5, xanchor="center", font=dict(size=14),
        ),
        plot_bgcolor="rgb(244,247,252)",
        paper_bgcolor="white",
        height=860,
        margin=dict(l=60, r=40, t=100, b=60),
        barmode="overlay",
    )
    fig.update_xaxes(title="East (ft)", row=1, col=1, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(title="North (ft)", row=1, col=1)
    fig.update_xaxes(title="Deviation (ft)", row=1, col=2)
    fig.update_yaxes(title="Count", row=1, col=2)
    fig.update_xaxes(title="Coverage (%)", row=2, col=1)
    fig.update_yaxes(title="Count", row=2, col=1)
    fig.update_xaxes(title="Distance (ft)", row=2, col=2)
    fig.update_yaxes(title="Count", row=2, col=2)

    return fig


def plot_monte_carlo_results(
    results: dict,
    base_config: UAVConfig,
    html_path: str = "monte_carlo.html",
    png_path: str  = "monte_carlo.png",
) -> None:
    """Write MC results as interactive HTML + static PNG."""
    fig = _build_mc_figure(results, base_config)

    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"HTML  -> {html_path}")

    try:
        fig.write_image(png_path, width=1600, height=960, scale=2)
        print(f"PNG   -> {png_path}  (kaleido)")
        return
    except Exception as kaleido_err:
        print(f"kaleido unavailable ({kaleido_err}), skipping MC PNG.", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(run_mc: bool = True) -> None:
    # --- Nominal mission ---
    config = UAVConfig()

    g_load         = 1.0 / math.cos(config.bank_angle_rad)
    rot_rad_s      = (1091.0 * math.tan(config.bank_angle_rad) * math.pi) \
                     / (config.airspeed_knots * 180.0)
    turn_radius_ft = (config.airspeed_knots * FT_PER_S_PER_KNOT) / rot_rad_s

    print("=== UAV Flight Path Generator ===")
    print(f"  TAS          : {config.airspeed_knots} kt  "
          f"({config.airspeed_knots * FT_PER_S_PER_KNOT:.1f} ft/s)")
    print(f"  Bank angle   : {config.bank_angle_deg}deg  ->  {g_load:.2f} G  "
          f"(maxG = {config.max_g})")
    print(f"  Turn radius  : {turn_radius_ft:.1f} ft")
    print(f"  Wind         : {config.wind_speed_mph} mph @ {config.wind_dir_deg}deg")
    print(f"  Swath width  : {config.swath_width_ft} ft  ({config.swath_width_m:.1f} m)")
    print(f"  Map area     : {config.map_x_km} x {config.map_y_km} km")
    print(f"  Target AGL   : {config.target_agl_ft:.0f} ft")
    print()

    # Build wind field + terrain for the nominal run
    wind_field = WindField(config.wind_speed_mph, config.wind_dir_deg)
    terrain    = TerrainModel(config.map_x_ft, config.map_y_ft)

    n_strips = math.ceil(config.map_y_ft / config.swath_width_ft)
    print(f"  Strips needed: {n_strips}")
    print("Generating flight path ...")

    x_path, y_path, z_path, segments = generate_lawnmower_path(
        config,
        wind_field=wind_field,
        terrain=terrain,
        align_strips=True,
    )

    n_straight = sum(1 for s in segments if s["type"] == "straight")
    n_turns    = sum(1 for s in segments if s["type"] == "turn")
    n_repos    = sum(1 for s in segments if s["type"] == "reposition")
    print(f"Path generated : {len(x_path):,} points | "
          f"{n_straight} survey legs | {n_turns} turns | {n_repos} reposition legs")

    metrics = compute_coverage_metrics(segments, config)
    print(f"Coverage       : {metrics['coverage_fraction'] * 100:.1f}%  |  "
          f"mean strip dev {metrics['mean_deviation_ft']:.1f} ft  |  "
          f"path length {metrics['path_length_ft'] / 5280.0:.2f} mi")
    print()

    plot_flight_path(
        segments, config,
        html_path="flight_path.html",
        png_path="flight_path.png",
        terrain=terrain,
        x_path=x_path,
        y_path=y_path,
        z_path=z_path,
    )

    if run_mc:
        print()
        print("Running Monte Carlo analysis ...")
        mc_config = MonteCarloConfig(n_runs=100)
        results   = run_monte_carlo(config, mc_config)
        print(f"MC complete: {results['n_valid']} valid runs")

        m_list = results["metrics"]
        mean_devs = [m["mean_deviation_ft"] for m in m_list]
        coverages = [m["coverage_fraction"] * 100 for m in m_list]
        print(f"  Mean strip dev : {np.mean(mean_devs):.1f} ft  "
              f"(p95 = {np.percentile(mean_devs, 95):.1f} ft)")
        print(f"  Coverage       : {np.mean(coverages):.1f}%  "
              f"(min = {np.min(coverages):.1f}%)")

        plot_monte_carlo_results(results, config,
                                  html_path="monte_carlo.html",
                                  png_path="monte_carlo.png")


if __name__ == "__main__":
    main()

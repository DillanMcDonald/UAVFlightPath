"""
UAVFlightPath — FlightPathGen.py
=================================
Generates a boustrophedon (lawnmower) coverage flight path for a fixed-wing UAV
over a rectangular survey area, accounting for wind drift.

Physics model
-------------
- Coordinated level turn: ROT (deg/s) = 1091 × tan(bank_angle) / TAS(knots)
  Rate of turn uses TRUE AIRSPEED, not ground speed.
- Ground speed = airspeed vector + wind vector  (vector addition, not subtraction)
- Position integrated via vectorized Euler method at dt=0.01 s resolution.

Units (internal)
----------------
- Speed   : ft/s  (converted from knots/mph at the input boundary)
- Distance: ft    (converted from km/m at the input boundary)
- Angles  : radians (converted from degrees at the input boundary)

Outputs
-------
- flight_path.html  : interactive Plotly figure
- flight_path.png   : static PNG (via kaleido; matplotlib fallback if unavailable)
"""

import math
import sys
import numpy as np

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
FT_PER_S_PER_KNOT: float = 1.68781   # 1 knot  = 1.68781 ft/s
FT_PER_KM: float = 3280.84            # 1 km    = 3280.84 ft
FT_PER_M: float = 3.28084             # 1 m     = 3.28084 ft
KNOTS_PER_MPH: float = 0.868976       # 1 mph   = 0.868976 knots


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class UAVConfig:
    """
    All mission configuration parameters.

    Input units match common engineering usage; derived values in ft/rad/knots
    are computed on construction and stored as attributes.

    Raises ValueError if the bank angle exceeds the max-G limit.
    """

    def __init__(
        self,
        map_x_km: float = 1.0,
        map_y_km: float = 1.0,
        airspeed_knots: float = 20.0,
        max_g: float = 2.0,
        bank_angle_deg: float = 60.0,
        wind_speed_mph: float = 5.0,
        wind_dir_deg: float = 120.0,
        swath_width_ft: float = 100.0,
        dt: float = 0.01,
    ) -> None:
        # --- Store inputs ---
        self.map_x_km = map_x_km
        self.map_y_km = map_y_km
        self.airspeed_knots = airspeed_knots
        self.max_g = max_g
        self.bank_angle_deg = bank_angle_deg
        self.wind_speed_mph = wind_speed_mph
        self.wind_dir_deg = wind_dir_deg   # direction wind blows TOWARD, from East CCW
        self.swath_width_ft = swath_width_ft
        self.dt = dt

        # --- Derived: angles ---
        self.bank_angle_rad: float = math.radians(bank_angle_deg)
        self.wind_dir_rad: float = math.radians(wind_dir_deg)

        # --- Enforce G-limit before any computation ---
        g_load = 1.0 / math.cos(self.bank_angle_rad)
        if g_load > max_g + 1e-6:
            raise ValueError(
                f"Bank angle {bank_angle_deg}deg produces {g_load:.2f}G, "
                f"exceeding maxG={max_g}. Reduce bank angle or raise maxG."
            )

        # --- Derived: wind ---
        self.wind_speed_knots: float = wind_speed_mph * KNOTS_PER_MPH
        # Ground speed = airspeed vector + wind vector (both point in direction of motion)
        self.wind_vec_knots: np.ndarray = np.array([
            self.wind_speed_knots * math.cos(self.wind_dir_rad),
            self.wind_speed_knots * math.sin(self.wind_dir_rad),
        ])

        # --- Derived: map dimensions in feet ---
        self.map_x_ft: float = map_x_km * FT_PER_KM
        self.map_y_ft: float = map_y_km * FT_PER_KM

        # --- Derived: swath in meters (display only) ---
        self.swath_width_m: float = swath_width_ft / FT_PER_M


# ---------------------------------------------------------------------------
# Flight mechanics — turn arc
# ---------------------------------------------------------------------------
def compute_turn_arc(
    start_pos: np.ndarray,
    start_heading_rad: float,
    config: UAVConfig,
    turn_direction: int,
) -> tuple:
    """
    Compute a 180deg coordinated turn arc via vectorized numerical integration.

    Rate of turn uses TRUE AIRSPEED (aviation formula):
        ROT (deg/s) = 1091 × tan(bank) / TAS(knots)
        ROT (rad/s) = ROT(deg/s) × π / 180

    Wind causes positional drift during the arc (trochoidal path).

    Args:
        start_pos         : [x, y] start position in feet
        start_heading_rad : aircraft heading at turn entry (radians, math convention)
        config            : UAVConfig
        turn_direction    : +1 = left/CCW, -1 = right/CW  (math convention)

    Returns:
        x (ft array), y (ft array), final_heading_rad (float)
    """
    # Rate of turn — TAS only, never ground speed
    rot_rad_s = (
        (1091.0 * math.tan(config.bank_angle_rad) * math.pi)
        / (config.airspeed_knots * 180.0)
    ) * turn_direction

    turn_duration = math.pi / abs(rot_rad_s)   # seconds for exactly 180deg
    t = np.arange(0.0, turn_duration, config.dt)

    # Heading evolves linearly during a coordinated level turn
    headings = start_heading_rad + rot_rad_s * t

    # Ground speed = airspeed vector + wind vector (both converted to ft/s)
    airspeed_fts = config.airspeed_knots * FT_PER_S_PER_KNOT
    wind_fts = config.wind_vec_knots * FT_PER_S_PER_KNOT

    vx = airspeed_fts * np.cos(headings) + wind_fts[0]
    vy = airspeed_fts * np.sin(headings) + wind_fts[1]

    # Euler integration: x[n+1] = x[n] + vx[n]*dt
    x = np.empty(len(t) + 1)
    y = np.empty(len(t) + 1)
    x[0], y[0] = float(start_pos[0]), float(start_pos[1])
    x[1:] = start_pos[0] + np.cumsum(vx * config.dt)
    y[1:] = start_pos[1] + np.cumsum(vy * config.dt)

    final_heading = headings[-1] + rot_rad_s * config.dt
    return x, y, float(final_heading)


# ---------------------------------------------------------------------------
# Flight mechanics — straight leg
# ---------------------------------------------------------------------------
def compute_straight_leg(
    start_pos: np.ndarray,
    heading_rad: float,
    leg_length_ft: float,
    config: UAVConfig,
) -> tuple:
    """
    Compute a straight flight segment covering leg_length_ft along the heading axis.

    Heading is fixed; ground speed = airspeed + wind (constant vector).
    Cross-wind component causes lateral drift in y.

    Args:
        start_pos     : [x, y] start position in feet
        heading_rad   : fixed aircraft heading (radians)
        leg_length_ft : distance to travel in the heading axis (feet)
        config        : UAVConfig

    Returns:
        x (ft array), y (ft array)  — heading is unchanged
    """
    airspeed_fts = config.airspeed_knots * FT_PER_S_PER_KNOT
    wind_fts = config.wind_vec_knots * FT_PER_S_PER_KNOT

    # Ground speed components — constant during straight flight at fixed heading
    gspd_x = airspeed_fts * math.cos(heading_rad) + wind_fts[0]
    gspd_y = airspeed_fts * math.sin(heading_rad) + wind_fts[1]

    # Guard: near-zero east/west progress means headwind ≈ airspeed
    if abs(gspd_x) < 0.5:
        raise ValueError(
            f"East/west ground speed ({gspd_x:.2f} ft/s) too low. "
            "Headwind may equal or exceed airspeed."
        )

    # Time to cover leg_length_ft in the east/west direction
    t_end = leg_length_ft / abs(gspd_x)
    t = np.arange(0.0, t_end, config.dt)

    x = start_pos[0] + gspd_x * t
    y = start_pos[1] + gspd_y * t

    # Append exact endpoint (arange may not include it)
    x = np.append(x, start_pos[0] + gspd_x * t_end)
    y = np.append(y, start_pos[1] + gspd_y * t_end)

    return x, y


# ---------------------------------------------------------------------------
# Path generation — full lawnmower pattern
# ---------------------------------------------------------------------------
def generate_lawnmower_path(config: UAVConfig) -> tuple:
    """
    Generate a complete boustrophedon (lawnmower) coverage path.

    Strip layout:
    - Strips run East–West.
    - Strip spacing = swath_width_ft.
    - After each eastbound leg  : left (CCW, +1) 180deg turn -> displaces north.
    - After each westbound leg  : right (CW,  -1) 180deg turn -> displaces north.
    - Total strips = ceil(map_y_ft / swath_width_ft).

    Note: Turn diameter (2 × turn_radius) and swath_width_ft may differ.
    In operational use a repositioning segment aligns strips exactly.

    Returns:
        x_path (ft), y_path (ft), segments (list of dicts with type/x/y/strip)
    """
    n_strips = math.ceil(config.map_y_ft / config.swath_width_ft)
    segments = []

    current_pos = np.array([0.0, 0.0])   # start position [x, y] in feet
    current_heading = 0.0                 # start heading East (0 rad)

    for strip_idx in range(n_strips):
        heading_east = (strip_idx % 2 == 0)

        # --- Straight survey leg ---
        x_seg, y_seg = compute_straight_leg(
            current_pos, current_heading, config.map_x_ft, config
        )
        segments.append({
            "type": "straight",
            "strip": strip_idx,
            "direction": "east" if heading_east else "west",
            "x": x_seg,
            "y": y_seg,
        })
        current_pos = np.array([x_seg[-1], y_seg[-1]])

        # --- 180deg turn (omitted after the final strip) ---
        if strip_idx < n_strips - 1:
            # Left (CCW, +1) after east leg: arc through North -> arrive heading West
            # Right (CW,  -1) after west leg: arc through North -> arrive heading East
            turn_dir = +1 if heading_east else -1
            x_turn, y_turn, current_heading = compute_turn_arc(
                current_pos, current_heading, config, turn_dir
            )
            segments.append({
                "type": "turn",
                "strip": strip_idx,
                "direction": "left" if turn_dir == +1 else "right",
                "x": x_turn,
                "y": y_turn,
            })
            current_pos = np.array([x_turn[-1], y_turn[-1]])
        # heading for next straight leg is set by compute_turn_arc;
        # for strip 0 (east) the next heading is already West after the turn.

    # Concatenate into a single path, skipping duplicate boundary points
    all_x = [segments[0]["x"]]
    all_y = [segments[0]["y"]]
    for seg in segments[1:]:
        all_x.append(seg["x"][1:])
        all_y.append(seg["y"][1:])

    return np.concatenate(all_x), np.concatenate(all_y), segments


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
def _build_plotly_figure(segments: list, config: UAVConfig):
    """Build and return a Plotly Figure for the flight path."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # --- Swath coverage bands (alternating colours) ---
    n_strips = math.ceil(config.map_y_ft / config.swath_width_ft)
    for i in range(n_strips):
        y0_band = i * config.swath_width_ft
        y1_band = y0_band + config.swath_width_ft
        fig.add_hrect(
            y0=y0_band, y1=y1_band,
            fillcolor="lightsteelblue" if i % 2 == 0 else "lightyellow",
            opacity=0.20,
            line_width=0,
        )

    # --- Segment traces (coloured by type) ---
    straight_legend_shown = False
    turn_legend_shown = False

    for seg in segments:
        if seg["type"] == "straight":
            fig.add_trace(go.Scatter(
                x=seg["x"], y=seg["y"],
                mode="lines",
                line=dict(color="royalblue", width=2.5),
                name="Survey leg",
                legendgroup="straight",
                showlegend=not straight_legend_shown,
                hovertemplate="x=%{x:.1f} ft<br>y=%{y:.1f} ft<extra></extra>",
            ))
            straight_legend_shown = True
        else:
            fig.add_trace(go.Scatter(
                x=seg["x"], y=seg["y"],
                mode="lines",
                line=dict(color="darkorange", width=2.0, dash="dot"),
                name="Turn arc",
                legendgroup="turn",
                showlegend=not turn_legend_shown,
                hovertemplate="x=%{x:.1f} ft<br>y=%{y:.1f} ft<extra></extra>",
            ))
            turn_legend_shown = True

    # --- Start / End markers ---
    x0, y0 = float(segments[0]["x"][0]), float(segments[0]["y"][0])
    xn, yn = float(segments[-1]["x"][-1]), float(segments[-1]["y"][-1])

    fig.add_trace(go.Scatter(
        x=[x0], y=[y0], mode="markers+text",
        marker=dict(color="green", size=14, symbol="triangle-right"),
        text=["START"], textposition="top right",
        name="Start",
    ))
    fig.add_trace(go.Scatter(
        x=[xn], y=[yn], mode="markers+text",
        marker=dict(color="crimson", size=14, symbol="square"),
        text=["END"], textposition="top right",
        name="End",
    ))

    # --- Wind vector arrow ---
    arrow_scale = config.map_x_ft * 0.12
    if config.wind_speed_knots > 0:
        unit = config.wind_vec_knots / config.wind_speed_knots
        wx, wy = unit[0] * arrow_scale, unit[1] * arrow_scale
    else:
        wx, wy = 0.0, 0.0
    ax_ox, ay_oy = config.map_x_ft * 0.82, config.map_y_ft * 0.06

    fig.add_annotation(
        x=ax_ox + wx, y=ay_oy + wy,
        ax=ax_ox, ay=ay_oy,
        xref="x", yref="y", axref="x", ayref="y",
        text=f"Wind<br>{config.wind_speed_mph} mph",
        showarrow=True,
        arrowhead=2, arrowsize=1.5,
        arrowcolor="steelblue", arrowwidth=2.5,
        font=dict(size=10, color="steelblue"),
        bgcolor="rgba(255,255,255,0.75)",
    )

    # --- Mission parameter box ---
    rot_rad_s = (1091.0 * math.tan(config.bank_angle_rad) * math.pi) / (config.airspeed_knots * 180.0)
    turn_radius_ft = (config.airspeed_knots * FT_PER_S_PER_KNOT) / rot_rad_s
    g_load = 1.0 / math.cos(config.bank_angle_rad)

    info = (
        "<b>Mission Parameters</b><br>"
        f"Area : {config.map_x_km} × {config.map_y_km} km<br>"
        f"TAS  : {config.airspeed_knots} kt<br>"
        f"Bank : {config.bank_angle_deg}deg -> {g_load:.2f} G  (max {config.max_g} G)<br>"
        f"ROT  : {math.degrees(rot_rad_s):.1f} deg/s<br>"
        f"Turn R: {turn_radius_ft:.0f} ft<br>"
        f"Swath: {config.swath_width_ft:.0f} ft  ({config.swath_width_m:.1f} m)<br>"
        f"Strips: {n_strips}<br>"
        f"Wind : {config.wind_speed_mph} mph @ {config.wind_dir_deg}deg"
    )
    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        xanchor="left", yanchor="top",
        text=info, showarrow=False,
        bgcolor="rgba(255,255,255,0.88)",
        bordercolor="rgba(80,80,80,0.5)",
        borderwidth=1,
        font=dict(size=11, family="monospace"),
    )

    # --- Layout ---
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
        xaxis=dict(
            title="East position (ft)",
            scaleanchor="y", scaleratio=1,
            showgrid=True, gridcolor="rgba(180,180,180,0.45)",
            zeroline=True, zerolinecolor="rgba(0,0,0,0.25)",
        ),
        yaxis=dict(
            title="North position (ft)",
            showgrid=True, gridcolor="rgba(180,180,180,0.45)",
            zeroline=True, zerolinecolor="rgba(0,0,0,0.25)",
        ),
        plot_bgcolor="rgb(244,247,252)",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1),
        margin=dict(l=70, r=40, t=130, b=70),
        height=720,
    )

    return fig


def plot_flight_path(
    segments: list,
    config: UAVConfig,
    html_path: str = "flight_path.html",
    png_path: str = "flight_path.png",
) -> None:
    """
    Write an interactive HTML and a static PNG of the flight path.

    PNG export tries kaleido first (plotly native), then matplotlib as fallback.

    Args:
        segments  : output of generate_lawnmower_path
        config    : UAVConfig
        html_path : destination for interactive Plotly HTML
        png_path  : destination for static PNG
    """
    fig = _build_plotly_figure(segments, config)

    # --- HTML ---
    fig.write_html(html_path, include_plotlyjs="cdn")
    print(f"HTML  -> {html_path}")

    # --- PNG: kaleido ---
    try:
        fig.write_image(png_path, width=1400, height=800, scale=2)
        print(f"PNG   -> {png_path}  (kaleido)")
        return
    except Exception as kaleido_err:
        print(f"kaleido unavailable ({kaleido_err}), trying matplotlib fallback …")

    # --- PNG: matplotlib fallback ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        fig_mpl, ax = plt.subplots(figsize=(14, 8), dpi=150)
        ax.set_facecolor("#f4f7fc")
        fig_mpl.patch.set_facecolor("white")

        n_strips = math.ceil(config.map_y_ft / config.swath_width_ft)
        colours = ["#b0c4de", "#fffacd"]
        for i in range(n_strips):
            y0_band = i * config.swath_width_ft
            ax.axhspan(y0_band, y0_band + config.swath_width_ft,
                       color=colours[i % 2], alpha=0.25)

        for seg in segments:
            if seg["type"] == "straight":
                ax.plot(seg["x"], seg["y"], color="royalblue", lw=1.8)
            else:
                ax.plot(seg["x"], seg["y"], color="darkorange",
                        lw=1.5, linestyle="dotted")

        x0, y0 = float(segments[0]["x"][0]), float(segments[0]["y"][0])
        xn, yn = float(segments[-1]["x"][-1]), float(segments[-1]["y"][-1])
        ax.scatter([x0], [y0], color="green",  s=120, zorder=5, marker="^", label="Start")
        ax.scatter([xn], [yn], color="crimson", s=120, zorder=5, marker="s", label="End")

        straight_patch = mpatches.Patch(color="royalblue",  label="Survey leg")
        turn_patch     = mpatches.Patch(color="darkorange", label="Turn arc")
        ax.legend(handles=[straight_patch, turn_patch,
                            mpatches.Patch(color="green",  label="Start"),
                            mpatches.Patch(color="crimson", label="End")],
                  loc="upper right")

        rot_rad_s = (1091.0 * math.tan(config.bank_angle_rad) * math.pi) / (config.airspeed_knots * 180.0)
        turn_radius_ft = (config.airspeed_knots * FT_PER_S_PER_KNOT) / rot_rad_s
        g_load = 1.0 / math.cos(config.bank_angle_rad)

        ax.set_title(
            f"UAV Lawnmower Coverage Flight Path\n"
            f"TAS {config.airspeed_knots} kt | {config.bank_angle_deg}deg bank "
            f"({g_load:.2f} G) | Turn radius {turn_radius_ft:.0f} ft | "
            f"Swath {config.swath_width_ft:.0f} ft | "
            f"Wind {config.wind_speed_mph} mph @ {config.wind_dir_deg}deg",
            fontsize=10,
        )
        ax.set_xlabel("East position (ft)")
        ax.set_ylabel("North position (ft)")
        ax.set_aspect("equal")
        ax.grid(True, color="gray", alpha=0.25)

        plt.tight_layout()
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig_mpl)
        print(f"PNG   -> {png_path}  (matplotlib fallback)")
    except Exception as mpl_err:
        print(f"WARNING: PNG export failed — {mpl_err}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    config = UAVConfig()

    g_load = 1.0 / math.cos(config.bank_angle_rad)
    rot_rad_s = (1091.0 * math.tan(config.bank_angle_rad) * math.pi) / (config.airspeed_knots * 180.0)
    turn_radius_ft = (config.airspeed_knots * FT_PER_S_PER_KNOT) / rot_rad_s

    print("=== UAV Flight Path Generator ===")
    print(f"  TAS          : {config.airspeed_knots} kt  "
          f"({config.airspeed_knots * FT_PER_S_PER_KNOT:.1f} ft/s)")
    print(f"  Bank angle   : {config.bank_angle_deg}deg  ->  {g_load:.2f} G  "
          f"(maxG = {config.max_g})")
    print(f"  Turn radius  : {turn_radius_ft:.1f} ft")
    print(f"  Wind         : {config.wind_speed_mph} mph @ {config.wind_dir_deg}deg  "
          f"->  {config.wind_vec_knots} kt")
    print(f"  Swath width  : {config.swath_width_ft} ft  ({config.swath_width_m:.1f} m)")
    print(f"  Map area     : {config.map_x_km} × {config.map_y_km} km")

    n_strips = math.ceil(config.map_y_ft / config.swath_width_ft)
    print(f"  Strips needed: {n_strips}")
    print()

    x_path, y_path, segments = generate_lawnmower_path(config)

    n_straight = sum(1 for s in segments if s["type"] == "straight")
    n_turns    = sum(1 for s in segments if s["type"] == "turn")
    print(f"Path generated : {len(x_path):,} points | "
          f"{n_straight} survey legs | {n_turns} turn arcs")

    plot_flight_path(
        segments, config,
        html_path="flight_path.html",
        png_path="flight_path.png",
    )


if __name__ == "__main__":
    main()

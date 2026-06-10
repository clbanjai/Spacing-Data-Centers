import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import orekit
from orekit.pyhelpers import setup_orekit_curdir

# =========================
# USER SETTINGS
# =========================
CSV_FILE = "best_meo_15000km_eclipse_results.csv"
OUTPUT_FIG = "final_orbit.png"

# Ground track duration
GROUND_TRACK_DAYS = 2

# 3D Earth-fixed orbit duration
N_ORBITS_3D = 2.0

# Subsampling
# If your CSV is every 300 seconds:
# STEP = 1  -> every 5 minutes
# STEP = 3  -> every 15 minutes
# STEP = 6  -> every 30 minutes
GROUND_STEP = 1
ORBIT_STEP = 1

EARTH_RADIUS_M = 6378137.0
EARTH_FLATTENING = 1.0 / 298.257223563
MU_EARTH = 3.986004418e14

# =========================
# COLORS / STYLE
# =========================
BG = "#0b1220"
AX_BG = "#111827"
GRID = "#334155"
TEXT = "#e5e7eb"
MUTED = "#94a3b8"

ORBIT = "#7dd3fc"
EARTH = "#67e8f9"
GROUND = "#a78bfa"
SUNLIGHT = "#facc15"
ECLIPSE = "#fb7185"

# =========================
# OREKIT INIT
# =========================
vm = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)

from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.bodies import OneAxisEllipsoid
from org.orekit.utils import Constants, IERSConventions
from org.hipparchus.geometry.euclidean.threed import Vector3D


# =========================
# HELPERS
# =========================
def absolute_date_from_string(s):
    dt = pd.to_datetime(s, utc=True)
    utc = TimeScalesFactory.getUTC()

    return AbsoluteDate(
        int(dt.year),
        int(dt.month),
        int(dt.day),
        int(dt.hour),
        int(dt.minute),
        float(dt.second + dt.microsecond / 1e6),
        utc,
    )


def geodetic_to_ecef(lat_rad, lon_rad, alt_m):
    a = EARTH_RADIUS_M
    f = EARTH_FLATTENING
    e2 = f * (2.0 - f)

    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    cos_lon = math.cos(lon_rad)
    sin_lon = math.sin(lon_rad)

    N = a / math.sqrt(1.0 - e2 * sin_lat**2)

    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + alt_m) * sin_lat

    return x, y, z


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - radius, x_middle + radius])
    ax.set_ylim3d([y_middle - radius, y_middle + radius])
    ax.set_zlim3d([z_middle - radius, z_middle + radius])


def style_2d_axes(ax):
    ax.set_facecolor(AX_BG)

    for spine in ax.spines.values():
        spine.set_color(MUTED)

    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.45)


def dataframe_to_itrf_and_groundtrack(df, earth, inertial_frame):
    r_itrf = []
    lats_deg = []
    lons_deg = []

    for _, row in df.iterrows():
        ad = absolute_date_from_string(row["date"])

        sat_pos_eci = Vector3D(
            float(row["x_m"]),
            float(row["y_m"]),
            float(row["z_m"]),
        )

        gp = earth.transform(sat_pos_eci, inertial_frame, ad)

        lat_rad = gp.getLatitude()
        lon_rad = gp.getLongitude()
        alt_m = gp.getAltitude()

        x_itrf, y_itrf, z_itrf = geodetic_to_ecef(lat_rad, lon_rad, alt_m)

        r_itrf.append([x_itrf, y_itrf, z_itrf])

        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(lon_rad)
        lon_deg = ((lon_deg + 180.0) % 360.0) - 180.0

        lats_deg.append(lat_deg)
        lons_deg.append(lon_deg)

    return (
        np.array(r_itrf, dtype=float),
        np.array(lats_deg, dtype=float),
        np.array(lons_deg, dtype=float),
    )


def break_longitude_wrap(lons_deg, lats_deg):
    lon_plot = lons_deg.copy()
    lat_plot = lats_deg.copy()

    for i in range(1, len(lon_plot)):
        if abs(lon_plot[i] - lon_plot[i - 1]) > 180.0:
            lon_plot[i] = np.nan
            lat_plot[i] = np.nan

    return lon_plot, lat_plot


# =========================
# LOAD CSV
# =========================
df_all = pd.read_csv(CSV_FILE)
df_all["datetime"] = pd.to_datetime(df_all["date"], utc=True)

required_cols = [
    "t_seconds",
    "date",
    "x_m",
    "y_m",
    "z_m",
    "any_eclipse",
]

for col in required_cols:
    if col not in df_all.columns:
        raise ValueError(f"Missing required column in CSV: {col}")

t_start = float(df_all["t_seconds"].min())
t_available_end = float(df_all["t_seconds"].max())
available_duration_s = t_available_end - t_start

# Estimate orbital period from mean orbital radius
r_all = df_all[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
radius_all = np.linalg.norm(r_all, axis=1)

mean_radius_m = float(np.nanmean(radius_all))
mean_altitude_km = (mean_radius_m - EARTH_RADIUS_M) / 1000.0

orbit_period_s = 2.0 * math.pi * math.sqrt(mean_radius_m**3 / MU_EARTH)
orbit_period_min = orbit_period_s / 60.0
orbit_period_hr = orbit_period_s / 3600.0

# Separate windows for each plot
orbit_duration_s = N_ORBITS_3D * orbit_period_s
ground_duration_s = GROUND_TRACK_DAYS * 86400.0

orbit_end = min(t_start + orbit_duration_s, t_available_end)
ground_end = min(t_start + ground_duration_s, t_available_end)

df_orbit = df_all[
    (df_all["t_seconds"] >= t_start) &
    (df_all["t_seconds"] <= orbit_end)
].copy()

df_ground = df_all[
    (df_all["t_seconds"] >= t_start) &
    (df_all["t_seconds"] <= ground_end)
].copy()

df_orbit = df_orbit.iloc[::ORBIT_STEP].copy().reset_index(drop=True)
df_ground = df_ground.iloc[::GROUND_STEP].copy().reset_index(drop=True)

# =========================
# OREKIT FRAMES / EARTH MODEL
# =========================
inertial_frame = FramesFactory.getEME2000()
earth_fixed_frame = FramesFactory.getITRF(IERSConventions.IERS_2010, True)

earth = OneAxisEllipsoid(
    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
    Constants.WGS84_EARTH_FLATTENING,
    earth_fixed_frame,
)

# =========================
# CONVERT DATA
# =========================
r_itrf_orbit, orbit_lats_deg, orbit_lons_deg = dataframe_to_itrf_and_groundtrack(
    df_orbit,
    earth,
    inertial_frame,
)

r_itrf_ground, ground_lats_deg, ground_lons_deg = dataframe_to_itrf_and_groundtrack(
    df_ground,
    earth,
    inertial_frame,
)

ground_lon_plot, ground_lat_plot = break_longitude_wrap(
    ground_lons_deg,
    ground_lats_deg,
)

ground_any_eclipse = df_ground["any_eclipse"].astype(bool).to_numpy()
ground_sun_mask = ~ground_any_eclipse
ground_ecl_mask = ground_any_eclipse

# =========================
# PRINT SUMMARY
# =========================
actual_orbit_plot_duration_s = float(df_orbit["t_seconds"].max() - df_orbit["t_seconds"].min())
actual_ground_plot_duration_s = float(df_ground["t_seconds"].max() - df_ground["t_seconds"].min())

print("\n--- Orbit / Plot Summary ---")
print(f"CSV file: {CSV_FILE}")
print(f"Total CSV duration: {available_duration_s / 3600.0:.3f} hours")
print()
print(f"Mean orbital radius: {mean_radius_m / 1000.0:.3f} km")
print(f"Mean altitude: {mean_altitude_km:.3f} km")
print(f"Estimated orbital period: {orbit_period_min:.3f} minutes")
print(f"Estimated orbital period: {orbit_period_hr:.3f} hours")
print()
print(f"Requested 3D orbit window: {N_ORBITS_3D:.2f} orbit(s)")
print(f"Actual 3D orbit window: {actual_orbit_plot_duration_s / 3600.0:.3f} hours")
print(f"3D orbit points plotted: {len(df_orbit)}")
print()
print(f"Requested ground-track window: {GROUND_TRACK_DAYS:.2f} day(s)")
print(f"Actual ground-track window: {actual_ground_plot_duration_s / 3600.0:.3f} hours")
print(f"Ground-track points plotted: {len(df_ground)}")
print()
print(f"Ground-track latitude range: {np.nanmin(ground_lats_deg):.3f} to {np.nanmax(ground_lats_deg):.3f} deg")
print(f"Ground-track longitude range: {np.nanmin(ground_lons_deg):.3f} to {np.nanmax(ground_lons_deg):.3f} deg")

# =========================
# PLOT
# =========================
plt.close("all")
fig = plt.figure(figsize=(16, 7), facecolor=BG)

# ---------- LEFT: 3D ORBIT ----------
ax1 = fig.add_subplot(121, projection="3d")
ax1.set_facecolor(AX_BG)

ax1.xaxis.pane.set_facecolor((0.10, 0.12, 0.18, 1.0))
ax1.yaxis.pane.set_facecolor((0.10, 0.12, 0.18, 1.0))
ax1.zaxis.pane.set_facecolor((0.10, 0.12, 0.18, 1.0))

for axis in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
    axis.label.set_color(TEXT)

ax1.tick_params(colors=TEXT)
ax1.title.set_color(TEXT)

# Bright Earth sphere
u = np.linspace(0, 2 * np.pi, 120)
v_sphere = np.linspace(0, np.pi, 60)

xs = EARTH_RADIUS_M * np.outer(np.cos(u), np.sin(v_sphere))
ys = EARTH_RADIUS_M * np.outer(np.sin(u), np.sin(v_sphere))
zs = EARTH_RADIUS_M * np.outer(np.ones_like(u), np.cos(v_sphere))

ax1.plot_surface(
    xs,
    ys,
    zs,
    color=EARTH,
    alpha=0.38,
    linewidth=0,
    shade=True,
)

ax1.plot(
    r_itrf_orbit[:, 0],
    r_itrf_orbit[:, 1],
    r_itrf_orbit[:, 2],
    color=ORBIT,
    linewidth=2.4,
    alpha=0.98,
    label=f"Orbit path ({N_ORBITS_3D:g} orbit(s))",
)

ax1.scatter(
    [r_itrf_orbit[0, 0]],
    [r_itrf_orbit[0, 1]],
    [r_itrf_orbit[0, 2]],
    s=85,
    color="white",
    edgecolors="black",
    linewidths=0.7,
    label="Start",
)

ax1.set_title("3D Orbit in Earth-Fixed Frame", pad=12)
ax1.set_xlabel("x (m)")
ax1.set_ylabel("y (m)")
ax1.set_zlabel("z (m)")

set_axes_equal(ax1)

ax1.legend(
    loc="upper left",
    fontsize=9,
    facecolor=AX_BG,
    edgecolor=MUTED,
    labelcolor=TEXT,
)

# ---------- RIGHT: GROUND TRACK ----------
ax2 = fig.add_subplot(122)
style_2d_axes(ax2)

ax2.plot(
    ground_lon_plot,
    ground_lat_plot,
    color=GROUND,
    linewidth=1.8,
    alpha=0.95,
    label=f"Ground track ({GROUND_TRACK_DAYS:g} day)",
)

ax2.scatter(
    ground_lons_deg[ground_sun_mask],
    ground_lats_deg[ground_sun_mask],
    s=18,
    color=SUNLIGHT,
    alpha=0.95,
    label="Sunlight",
)

if np.any(ground_ecl_mask):
    ax2.scatter(
        ground_lons_deg[ground_ecl_mask],
        ground_lats_deg[ground_ecl_mask],
        s=28,
        color=ECLIPSE,
        alpha=0.95,
        label="Eclipse",
    )

ax2.set_xlim(-180, 180)
ax2.set_ylim(-90, 90)
ax2.set_xlabel("Longitude (deg)")
ax2.set_ylabel("Latitude (deg)")
ax2.set_title("Ground Track")

ax2.legend(
    loc="upper right",
    fontsize=9,
    facecolor=AX_BG,
    edgecolor=MUTED,
    labelcolor=TEXT,
)

fig.suptitle(
    f"Orbit Visualization",
    color=TEXT,
    fontsize=16,
    y=0.98,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUTPUT_FIG, dpi=250, facecolor=BG, bbox_inches="tight")
plt.show()

print(f"\nSaved figure to: {OUTPUT_FIG}")
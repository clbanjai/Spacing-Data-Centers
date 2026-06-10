import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Load CSV
# -------------------------

df = pd.read_csv("eclipse_experiment.csv")
# or:
# df = pd.read_csv("orekit_earth_moon_eclipse_experiment.csv")

# Convert meters to Earth radii for nicer plotting
R_EARTH = 6378137.0

df["x_re"] = df["x_m"] / R_EARTH
df["y_re"] = df["y_m"] / R_EARTH
df["z_re"] = df["z_m"] / R_EARTH

# -------------------------
# Create Earth sphere
# -------------------------

u = np.linspace(0, 2 * np.pi, 80)
v = np.linspace(0, np.pi, 40)

earth_x = np.outer(np.cos(u), np.sin(v))
earth_y = np.outer(np.sin(u), np.sin(v))
earth_z = np.outer(np.ones_like(u), np.cos(v))

# -------------------------
# Plot
# -------------------------

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection="3d")

# Earth
ax.plot_surface(
    earth_x,
    earth_y,
    earth_z,
    alpha=0.25,
    linewidth=0
)

# Full orbit trace
ax.plot(
    df["x_re"],
    df["y_re"],
    df["z_re"],
    linewidth=1,
    alpha=0.5,
    label="Orbit path"
)

# Plot points by status
statuses = df["status"].unique()

for status in statuses:
    sub = df[df["status"] == status]

    ax.scatter(
        sub["x_re"],
        sub["y_re"],
        sub["z_re"],
        s=8,
        label=status
    )

# Start and end markers
ax.scatter(
    df["x_re"].iloc[0],
    df["y_re"].iloc[0],
    df["z_re"].iloc[0],
    s=80,
    marker="o",
    label="Start"
)

ax.scatter(
    df["x_re"].iloc[-1],
    df["y_re"].iloc[-1],
    df["z_re"].iloc[-1],
    s=80,
    marker="x",
    label="End"
)

# -------------------------
# Equal aspect ratio
# -------------------------

max_range = max(
    df["x_re"].abs().max(),
    df["y_re"].abs().max(),
    df["z_re"].abs().max(),
    1.2
)

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)

ax.set_xlabel("x [Earth radii]")
ax.set_ylabel("y [Earth radii]")
ax.set_zlabel("z [Earth radii]")
ax.set_title("Satellite Orbit Around Earth with Eclipse Status")

ax.legend()
plt.tight_layout()
print("Plotting orbit and eclipse status...")
plt.savefig("orbit_eclipse_plot.png")
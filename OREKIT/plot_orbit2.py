import numpy as np
import pandas as pd
import plotly.graph_objects as go

df = pd.read_csv("eclipse_experiment.csv")
# or use:
# df = pd.read_csv("orekit_earth_moon_eclipse_experiment.csv")

R_EARTH = 6378137.0

df["x_re"] = df["x_m"] / R_EARTH
df["y_re"] = df["y_m"] / R_EARTH
df["z_re"] = df["z_m"] / R_EARTH

# Earth sphere
u = np.linspace(0, 2 * np.pi, 150)
v = np.linspace(0, np.pi, 80)

x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones_like(u), np.cos(v))

fig = go.Figure()

# Earth
fig.add_trace(go.Surface(
    x=x,
    y=y,
    z=z,
    opacity=0.75,
    colorscale="Blues",
    showscale=False,
    name="Earth"
))

# Orbit path
fig.add_trace(go.Scatter3d(
    x=df["x_re"],
    y=df["y_re"],
    z=df["z_re"],
    mode="lines",
    line=dict(width=5),
    name="Orbit path"
))

# Status points
for status in df["status"].unique():
    sub = df[df["status"] == status]

    fig.add_trace(go.Scatter3d(
        x=sub["x_re"],
        y=sub["y_re"],
        z=sub["z_re"],
        mode="markers",
        marker=dict(size=3),
        name=status
    ))

# Start
fig.add_trace(go.Scatter3d(
    x=[df["x_re"].iloc[0]],
    y=[df["y_re"].iloc[0]],
    z=[df["z_re"].iloc[0]],
    mode="markers",
    marker=dict(size=8),
    name="Start"
))

# End
fig.add_trace(go.Scatter3d(
    x=[df["x_re"].iloc[-1]],
    y=[df["y_re"].iloc[-1]],
    z=[df["z_re"].iloc[-1]],
    mode="markers",
    marker=dict(size=8, symbol="x"),
    name="End"
))

max_range = max(
    df["x_re"].abs().max(),
    df["y_re"].abs().max(),
    df["z_re"].abs().max(),
    1.2
)

fig.update_layout(
    title="3D Earth Orbit with Eclipse Status",
    scene=dict(
        xaxis=dict(title="x [Earth radii]", range=[-max_range, max_range]),
        yaxis=dict(title="y [Earth radii]", range=[-max_range, max_range]),
        zaxis=dict(title="z [Earth radii]", range=[-max_range, max_range]),
        aspectmode="cube",
    ),
    width=1000,
    height=900
)
print("Saving 3D plot to orbit_3d_render.html...")
fig.write_html("orbit_3d_render.html")
# fig.show()
import math
import pandas as pd
import matplotlib.pyplot as plt

from optimizer import optimize_orbit
from orbit_placement import run_orbital_placement


R_EARTH_M = 6378137.0


def earth_view_factor(semi_major_axis_m):
    """
    Approximate Earth view factor from orbit altitude.

    Uses the solid angle of Earth seen from the satellite:
        VF = 0.5 * (1 - cos(theta))
    where:
        theta = asin(R_earth / r)

    r is distance from Earth's center.
    """
    theta = math.asin(R_EARTH_M / semi_major_axis_m)
    vf = 0.5 * (1.0 - math.cos(theta))
    return vf


def get_capacity_factor(stats):
    """
    Capacity factor proxy:
        CF = fraction of simulated time in full sunlight.
    """
    total_time = stats["full_sunlight"] + stats["any_eclipse"]

    if total_time <= 0:
        return 0.0

    return stats["full_sunlight"] / total_time


def run_altitude_sweep(
    altitude_points_km,
    base_config,
    output_csv="altitude_sweep_results.csv",
    output_plot="altitude_sweep_cf_vf.png",
):
    results = []

    for altitude_km in altitude_points_km:
        print("\n" + "=" * 70)
        print(f"Optimizing altitude = {altitude_km:.1f} km")
        print("=" * 70)

        # Fix altitude by setting min=max
        config = base_config.copy()
        config["min_altitude_km"] = float(altitude_km)
        config["max_altitude_km"] = float(altitude_km)

        best_params, opt_result, history_df = optimize_orbit(config)

        semi_major_axis_m = float(best_params["semi_major_axis_m"])

        print("\nRunning final orbit placement for this altitude...")

        df, stats = run_orbital_placement(
            semi_major_axis_m=semi_major_axis_m,
            eccentricity=float(config["eccentricity"]),
            inclination_deg=float(best_params["inclination_deg"]),
            argument_of_perigee_deg=float(config["argument_of_perigee_deg"]),
            raan_deg=float(best_params["raan_deg"]),
            true_anomaly_deg=float(best_params["true_anomaly_deg"]),
            start_date_tuple=config["start_date_tuple"],
            end_date_tuple=config["end_date_tuple"],
            timestep_seconds=float(config["timestep_seconds"]),
            save_csv=False,
        )

        cf = get_capacity_factor(stats)
        vf = earth_view_factor(semi_major_axis_m)

        row = {
            "altitude_km": float(altitude_km),
            "semi_major_axis_m": semi_major_axis_m,

            "inclination_deg": float(best_params["inclination_deg"]),
            "raan_deg": float(best_params["raan_deg"]),
            "true_anomaly_deg": float(best_params["true_anomaly_deg"]),
            "eccentricity": float(config["eccentricity"]),
            "argument_of_perigee_deg": float(config["argument_of_perigee_deg"]),

            "capacity_factor": cf,
            "view_factor": vf,

            "full_sunlight_hours": stats["full_sunlight"] / 3600.0,
            "any_eclipse_hours": stats["any_eclipse"] / 3600.0,
            "any_umbra_hours": stats["any_umbra"] / 3600.0,
            "any_penumbra_hours": stats["any_penumbra"] / 3600.0,

            "full_sunlight_percent": 100.0 * cf,
            "any_eclipse_percent": 100.0 * stats["any_eclipse"] / (
                stats["full_sunlight"] + stats["any_eclipse"]
            ),

            "best_score": float(best_params["best_score"]),
            "optimizer_iterations": int(best_params["num_optimizer_iterations"]),
            "function_evaluations": int(best_params["num_function_evaluations"]),
        }

        results.append(row)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)

        print(f"\nAltitude {altitude_km:.1f} km done")
        print(f"CF = {cf:.4f}")
        print(f"VF = {vf:.6f}")
        print(f"Intermediate results saved to {output_csv}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

    plot_cf_vf_vs_altitude(results_df, output_plot)

    print("\nSweep complete.")
    print(f"Saved CSV: {output_csv}")
    print(f"Saved plot: {output_plot}")

    return results_df


def plot_cf_vf_vs_altitude(results_df, output_plot):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(
        results_df["altitude_km"],
        results_df["capacity_factor"],
        marker="o",
        label="Capacity Factor"
    )
    ax1.set_xlabel("Altitude [km]")
    ax1.set_ylabel("Capacity Factor")
    ax1.set_ylim(0, 1.05)

    ax2 = ax1.twinx()
    ax2.plot(
        results_df["altitude_km"],
        results_df["view_factor"],
        marker="s",
        linestyle="--",
        label="Earth View Factor"
    )
    ax2.set_ylabel("Earth View Factor")

    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title("Optimized Capacity Factor and Earth View Factor vs Altitude")
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    plt.show()


if __name__ == "__main__":
    # LEO to GEO-ish sweep
    altitude_points_km = [
        400,
                800,
        1500,
        3000,
        5000,
        10000,
        15000,
        20000,
        25000,
        30000,
        35786,
    ]


    base_config = {
    "min_altitude_km": 400,
    "max_altitude_km": 35786,

    "min_inclination_deg": 0,
    "max_inclination_deg": 120,

    "eccentricity": 0.0,
    "argument_of_perigee_deg": 0.0,

    # One-year simulation window
    "start_date_tuple": (2026, 1, 1, 0, 0, 0.0),
    "end_date_tuple": (2027, 1, 1, 0, 0, 0.0),

    # Reasonable full-year timestep
    "timestep_seconds": 3600.0,

    # Keep optimizer light
    "maxiter": 3,
    "popsize": 4,
}
    run_altitude_sweep(
        altitude_points_km=altitude_points_km,
        base_config=base_config,
        output_csv="altitude_sweep_results.csv",
        output_plot="altitude_sweep_cf_vf.png",
    )
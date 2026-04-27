# optimizer.py

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from orbit_placement import run_orbital_placement
history = []

def score_orbit(params, config, history=None):
    altitude_km, inclination_deg, raan_deg, true_anomaly_deg = [float(x) for x in params]

    earth_radius_m = 6378137.0
    semi_major_axis_m = float(earth_radius_m + altitude_km * 1000.0)

    df, stats = run_orbital_placement(
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=float(config["eccentricity"]),
        inclination_deg=float(inclination_deg),
        argument_of_perigee_deg=float(config["argument_of_perigee_deg"]),
        raan_deg=float(raan_deg),
        true_anomaly_deg=float(true_anomaly_deg),
        start_date_tuple=config["start_date_tuple"],
        end_date_tuple=config["end_date_tuple"],
        timestep_seconds=float(config["timestep_seconds"]),
        save_csv=False,
    )

    total_time = stats["full_sunlight"] + stats["any_eclipse"]

    sunlight_fraction = stats["full_sunlight"] / total_time
    eclipse_fraction = stats["any_eclipse"] / total_time
    umbra_fraction = stats["any_umbra"] / total_time
    penumbra_fraction = stats["any_penumbra"] / total_time

    altitude_penalty = altitude_km / config["max_altitude_km"]

    score = (
        1.0 * sunlight_fraction
        - 5.0 * umbra_fraction
        - 1.0 * penumbra_fraction
        - 0.05 * altitude_penalty
    )

    objective_value = -score

    if history is not None:
        history.append({
            "eval_number": len(history) + 1,
            "altitude_km": altitude_km,
            "inclination_deg": inclination_deg,
            "raan_deg": raan_deg,
            "true_anomaly_deg": true_anomaly_deg,
            "score": score,
            "objective_value": objective_value,
            "sunlight_fraction": sunlight_fraction,
            "eclipse_fraction": eclipse_fraction,
            "umbra_fraction": umbra_fraction,
            "penumbra_fraction": penumbra_fraction,
        })

    return objective_value

def optimize_orbit(config):
    history = []

    bounds = [
        (config["min_altitude_km"], config["max_altitude_km"]),
        (config["min_inclination_deg"], config["max_inclination_deg"]),
        (0.0, 360.0),
        (0.0, 360.0),
    ]

    result = differential_evolution(
        lambda x: score_orbit(x, config, history),
        bounds=bounds,
        maxiter=config.get("maxiter", 20),
        popsize=config.get("popsize", 8),
        polish=False,
        workers=1,
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv("optimization_history.csv", index=False)

    altitude_km, inclination_deg, raan_deg, true_anomaly_deg = result.x

    earth_radius_m = 6378137.0
    semi_major_axis_m = earth_radius_m + altitude_km * 1000.0

    best_params = {
        "altitude_km": float(altitude_km),
        "semi_major_axis_m": float(semi_major_axis_m),
        "inclination_deg": float(inclination_deg),
        "raan_deg": float(raan_deg),
        "true_anomaly_deg": float(true_anomaly_deg),
        "best_score": float(-result.fun),
        "objective_value": float(result.fun),
        "num_function_evaluations": result.nfev,
        "num_optimizer_iterations": result.nit,
    }

    return best_params, result, history_df

if __name__ == "__main__":
    config = {
        "min_altitude_km": 400,
        "max_altitude_km": 50000,

        "min_inclination_deg": 0,
        "max_inclination_deg": 120,

        "eccentricity": 0.0,
        "argument_of_perigee_deg": 0.0,

        "start_date_tuple": (2026, 1, 1, 0, 0, 0.0),
        "end_date_tuple": (2026, 1, 8, 0, 0, 0.0),
        "timestep_seconds": 300.0,

        "maxiter": 15,
        "popsize": 6,
    }

    best_params, result, history_df = optimize_orbit(config)

    print("Optimizer iterations:", result.nit)
    print("Function evaluations:", result.nfev)

    print(history_df.sort_values("score", ascending=False).head())
    print("\nBest orbit found:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\nRunning final best orbit and saving CSV...")

    df, stats = run_orbital_placement(
        semi_major_axis_m=best_params["semi_major_axis_m"],
        eccentricity=config["eccentricity"],
        inclination_deg=best_params["inclination_deg"],
        argument_of_perigee_deg=config["argument_of_perigee_deg"],
        raan_deg=best_params["raan_deg"],
        true_anomaly_deg=best_params["true_anomaly_deg"],
        start_date_tuple=config["start_date_tuple"],
        end_date_tuple=config["end_date_tuple"],
        timestep_seconds=config["timestep_seconds"],
        save_csv=True,
        csv_filename="best_orbit_eclipse_results.csv",
    )
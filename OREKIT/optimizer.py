# optimizer.py

import contextlib
import io
import math

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from orbit_placement import run_orbital_placement


EARTH_RADIUS_M = 6378137.0
MU_EARTH = 3.986004418e14
PENALTY_SCORE = 1e9


def orbital_period_seconds(semi_major_axis_m):
    """
    Keplerian orbital period.
    """
    return 2.0 * math.pi * math.sqrt(semi_major_axis_m**3 / MU_EARTH)


def is_near_critical_inclination(inclination_deg, buffer_deg=1.0):
    """
    Eckstein-Hechler can have issues near critical inclinations.
    The optimizer should skip these instead of crashing.
    """

    critical_incls = [63.43494882, 116.56505118]

    return any(
        abs(inclination_deg - crit) < buffer_deg
        for crit in critical_incls
    )


def append_history(history, row):
    if history is not None:
        history.append(row)


def run_orbit_case(
    semi_major_axis_m,
    eccentricity,
    inclination_deg,
    argument_of_perigee_deg,
    raan_deg,
    true_anomaly_deg,
    config,
    save_csv=False,
    csv_filename=None,
    suppress_output=True,
):
    """
    Wrapper around run_orbital_placement.

    Your orbit_placement.py expects OrbitPropagator.
    """

    kwargs = dict(
        semi_major_axis_m=semi_major_axis_m,
        eccentricity=eccentricity,
        inclination_deg=inclination_deg,
        argument_of_perigee_deg=argument_of_perigee_deg,
        raan_deg=raan_deg,
        true_anomaly_deg=true_anomaly_deg,
        start_date_tuple=config["start_date_tuple"],
        end_date_tuple=config["end_date_tuple"],
        timestep_seconds=float(config["timestep_seconds"]),
        OrbitPropagator=config.get("OrbitPropagator", "j2"),
        save_csv=save_csv,
    )

    if csv_filename is not None:
        kwargs["csv_filename"] = csv_filename

    if suppress_output:
        with contextlib.redirect_stdout(io.StringIO()):
            return run_orbital_placement(**kwargs)

    return run_orbital_placement(**kwargs)


def score_orbit(params, config, history=None):
    """
    Optimizer variables for fixed 15,000 km MEO:

        params[0] = inclination_deg
        params[1] = raan_deg
        params[2] = true_anomaly_deg

    Altitude is fixed using:

        config["altitude_km"] = 15000.0
    """

    inclination_deg, raan_deg, true_anomaly_deg = [float(x) for x in params]

    altitude_km = float(config["altitude_km"])
    eccentricity = float(config["eccentricity"])
    argument_of_perigee_deg = float(config["argument_of_perigee_deg"])

    semi_major_axis_m = EARTH_RADIUS_M + altitude_km * 1000.0

    if is_near_critical_inclination(
        inclination_deg,
        buffer_deg=float(config.get("critical_inclination_buffer_deg", 1.0)),
    ):
        append_history(
            history,
            {
                "altitude_km": altitude_km,
                "semi_major_axis_m": semi_major_axis_m,
                "inclination_deg": inclination_deg,
                "raan_deg": raan_deg,
                "true_anomaly_deg": true_anomaly_deg,
                "score": PENALTY_SCORE,
                "failed": True,
                "failure_reason": "near_critical_inclination",
            },
        )

        return PENALTY_SCORE

    try:
        df, stats = run_orbit_case(
            semi_major_axis_m=semi_major_axis_m,
            eccentricity=eccentricity,
            inclination_deg=inclination_deg,
            argument_of_perigee_deg=argument_of_perigee_deg,
            raan_deg=raan_deg,
            true_anomaly_deg=true_anomaly_deg,
            config=config,
            save_csv=False,
            suppress_output=bool(config.get("suppress_optimizer_output", True)),
        )

    except Exception as e:
        if config.get("print_failed_orbits", True):
            print(
                f"Skipping failed orbit: "
                f"alt={altitude_km:.2f} km, "
                f"inc={inclination_deg:.3f} deg, "
                f"raan={raan_deg:.3f} deg, "
                f"nu={true_anomaly_deg:.3f} deg"
            )
            print(f"Reason: {e}")

        append_history(
            history,
            {
                "altitude_km": altitude_km,
                "semi_major_axis_m": semi_major_axis_m,
                "inclination_deg": inclination_deg,
                "raan_deg": raan_deg,
                "true_anomaly_deg": true_anomaly_deg,
                "score": PENALTY_SCORE,
                "failed": True,
                "failure_reason": str(e),
            },
        )

        return PENALTY_SCORE

    full_sunlight_seconds = float(stats["full_sunlight"])
    any_eclipse_seconds = float(stats["any_eclipse"])
    total_seconds = full_sunlight_seconds + any_eclipse_seconds

    if total_seconds <= 0:
        eclipse_fraction = 1.0
        sunlight_fraction = 0.0
    else:
        eclipse_fraction = any_eclipse_seconds / total_seconds
        sunlight_fraction = full_sunlight_seconds / total_seconds

    # Lower is better.
    score = eclipse_fraction

    append_history(
        history,
        {
            "altitude_km": altitude_km,
            "semi_major_axis_m": semi_major_axis_m,
            "inclination_deg": inclination_deg,
            "raan_deg": raan_deg,
            "true_anomaly_deg": true_anomaly_deg,
            "score": score,
            "failed": False,
            "failure_reason": "",
            "eclipse_fraction": eclipse_fraction,
            "sunlight_fraction": sunlight_fraction,
            "full_sunlight_seconds": full_sunlight_seconds,
            "any_eclipse_seconds": any_eclipse_seconds,
            "earth_umbra_seconds": stats.get("earth_umbra", np.nan),
            "earth_penumbra_seconds": stats.get("earth_penumbra", np.nan),
            "moon_umbra_seconds": stats.get("moon_umbra", np.nan),
            "moon_penumbra_seconds": stats.get("moon_penumbra", np.nan),
        },
    )

    return score


def optimize_orbit(config):
    history = []

    bounds = [
        (config["min_inclination_deg"], config["max_inclination_deg"]),
        (0.0, 360.0),  # RAAN
        (0.0, 360.0),  # true anomaly
    ]

    result = differential_evolution(
        lambda x: score_orbit(x, config, history),
        bounds=bounds,
        maxiter=config.get("maxiter", 20),
        popsize=config.get("popsize", 8),
        tol=config.get("tol", 0.01),
        polish=config.get("polish", False),
        workers=1,
        seed=config.get("seed", 7),
    )

    history_df = pd.DataFrame(history)
    history_df.to_csv("optimization_history.csv", index=False)

    if result.fun >= PENALTY_SCORE:
        raise RuntimeError(
            "Optimization failed because all tested orbits were invalid or penalized. "
            "Try widening inclination bounds, reducing the critical inclination buffer, "
            "or checking your orbit_placement.py propagator option."
        )

    inclination_deg, raan_deg, true_anomaly_deg = [float(x) for x in result.x]

    altitude_km = float(config["altitude_km"])
    eccentricity = float(config["eccentricity"])
    argument_of_perigee_deg = float(config["argument_of_perigee_deg"])
    semi_major_axis_m = EARTH_RADIUS_M + altitude_km * 1000.0

    period_s = orbital_period_seconds(semi_major_axis_m)

    best_params = {
        "altitude_km": altitude_km,
        "semi_major_axis_m": float(semi_major_axis_m),
        "semi_major_axis_km": float(semi_major_axis_m / 1000.0),
        "eccentricity": eccentricity,
        "inclination_deg": float(inclination_deg),
        "argument_of_perigee_deg": argument_of_perigee_deg,
        "raan_deg": float(raan_deg),
        "true_anomaly_deg": float(true_anomaly_deg),
        "orbital_period_seconds": float(period_s),
        "orbital_period_minutes": float(period_s / 60.0),
        "orbital_period_hours": float(period_s / 3600.0),
        "objective_value": float(result.fun),
        "best_eclipse_fraction": float(result.fun),
        "best_sunlight_fraction": float(1.0 - result.fun),
        "num_function_evaluations": int(result.nfev),
        "num_optimizer_iterations": int(result.nit),
    }

    return best_params, result, history_df


if __name__ == "__main__":

    config = {
        # Fixed MEO altitude.
        # This means altitude above Earth's surface, not semi-major axis.
        "altitude_km": 15000.0,

        # For this MEO case, altitude is fixed.
        # The optimizer chooses inclination, RAAN, and true anomaly.
        "min_inclination_deg": 1.0,
        "max_inclination_deg": 179.0,

        "eccentricity": 0.0,
        "argument_of_perigee_deg": 0.0,

        "start_date_tuple": (2026, 1, 1, 0, 0, 0.0),
        "end_date_tuple": (2026, 1, 8, 0, 0, 0.0),
        "timestep_seconds": 300.0,

        # This must match the option inside orbit_placement.py.
        # In your file, "j2" selects EcksteinHechlerPropagator.
        "OrbitPropagator": "j2",

        # Avoid Eckstein-Hechler critical-inclination issues.
        "critical_inclination_buffer_deg": 1.0,

        "maxiter": 15,
        "popsize": 6,
        "tol": 0.01,
        "polish": False,
        "seed": 7,

        # Keeps the optimizer from printing one eclipse summary per candidate.
        # The final best orbit will still print normally.
        "suppress_optimizer_output": True,
        "print_failed_orbits": True,
    }

    fixed_altitude_km = float(config["altitude_km"])
    fixed_semi_major_axis_m = EARTH_RADIUS_M + fixed_altitude_km * 1000.0
    fixed_period_s = orbital_period_seconds(fixed_semi_major_axis_m)

    print("\nFixed MEO setup:")
    print(f"Altitude: {fixed_altitude_km:.3f} km")
    print(f"Semi-major axis: {fixed_semi_major_axis_m / 1000.0:.3f} km")
    print(f"Assumed eccentricity: {config['eccentricity']}")
    print(f"Approx orbital period: {fixed_period_s / 60.0:.3f} minutes")
    print(f"Approx orbital period: {fixed_period_s / 3600.0:.3f} hours")

    best_params, result, history_df = optimize_orbit(config)

    print("\nOptimizer iterations:", result.nit)
    print("Function evaluations:", result.nfev)

    print("\nBest evaluated orbits:")
    valid_history = history_df[history_df["failed"] == False].copy()

    if len(valid_history) > 0:
        print(valid_history.sort_values("score", ascending=True).head())
    else:
        print("No valid orbits were evaluated.")

    print("\nBest orbit found:")
    for key, value in best_params.items():
        print(f"{key}: {value}")

    print("\nRunning final best orbit and saving CSV...")

    df, stats = run_orbit_case(
        semi_major_axis_m=best_params["semi_major_axis_m"],
        eccentricity=best_params["eccentricity"],
        inclination_deg=best_params["inclination_deg"],
        argument_of_perigee_deg=best_params["argument_of_perigee_deg"],
        raan_deg=best_params["raan_deg"],
        true_anomaly_deg=best_params["true_anomaly_deg"],
        config=config,
        save_csv=True,
        csv_filename="best_meo_15000km_eclipse_results.csv",
        suppress_output=False,
    )

    print("\nFinal best orbit stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\nFinal decided parameters used for the saved MEO run:")
    print(f"Altitude above Earth: {best_params['altitude_km']:.3f} km")
    print(f"Semi-major axis: {best_params['semi_major_axis_km']:.3f} km")
    print(f"Eccentricity: {best_params['eccentricity']:.6f}")
    print(f"Inclination: {best_params['inclination_deg']:.6f} deg")
    print(f"Argument of perigee: {best_params['argument_of_perigee_deg']:.6f} deg")
    print(f"RAAN: {best_params['raan_deg']:.6f} deg")
    print(f"True anomaly: {best_params['true_anomaly_deg']:.6f} deg")
    print(f"Orbital period: {best_params['orbital_period_minutes']:.6f} minutes")
    print(f"Orbital period: {best_params['orbital_period_hours']:.6f} hours")
    print(f"Best eclipse fraction: {best_params['best_eclipse_fraction']:.8f}")
    print(f"Best sunlight fraction: {best_params['best_sunlight_fraction']:.8f}")
import math
import pandas as pd

import orekit
from orekit.pyhelpers import setup_orekit_curdir

vm = orekit.initVM()
setup_orekit_curdir(from_pip_library=True)

from org.orekit.bodies import CelestialBodyFactory, OneAxisEllipsoid
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.frames import FramesFactory
from org.orekit.orbits import KeplerianOrbit, PositionAngleType
from org.orekit.propagation.analytical import KeplerianPropagator
from org.orekit.propagation.events import EclipseDetector
from org.orekit.utils import Constants, IERSConventions


def run_orbital_placement(
    semi_major_axis_m,
    eccentricity,
    inclination_deg,
    argument_of_perigee_deg,
    raan_deg,
    true_anomaly_deg,
    start_date_tuple,
    end_date_tuple,
    timestep_seconds,
    save_csv=False,
    csv_filename="eclipse_experiment.csv",
):
    """
    Runs Earth/Moon umbra + penumbra eclipse detection for a satellite orbit.

    Parameters
    ----------
    semi_major_axis_m : float
        Semi-major axis of the orbit in meters. This controls the size of the orbit
        (i.e., altitude for near-circular orbits).

    eccentricity : float
        Orbital eccentricity (0 = circular, 0 < e < 1 = elliptical). Determines how
        stretched the orbit is.

    inclination_deg : float
        Inclination of the orbit in degrees. Angle between the orbital plane and
        Earth's equatorial plane.

    argument_of_perigee_deg : float
        Argument of perigee in degrees. Defines where the closest point to Earth
        (perigee) lies within the orbital plane.

    raan_deg : float
        Right Ascension of the Ascending Node (RAAN) in degrees. Determines the
        orientation of the orbital plane around Earth (rotation about the Earth’s axis).

    true_anomaly_deg : float
        True anomaly in degrees. Specifies the satellite’s position along the orbit
        at the start of the simulation.

    start_date_tuple : tuple
        Start date and time of the simulation in UTC, formatted as
        (year, month, day, hour, minute, second).

    end_date_tuple : tuple
        End date and time of the simulation in UTC, formatted as
        (year, month, day, hour, minute, second).

    timestep_seconds : float
        Time step in seconds for sampling the orbit and evaluating eclipse conditions.

    save_csv : bool, optional
        If True, saves the simulation results to a CSV file. Default is False.

    csv_filename : str, optional
        Name of the CSV file to save results if save_csv is True.
        Default is "eclipse_experiment.csv".

    Description
    -----------
    Simulates a satellite orbit over the specified time range and computes when the
    spacecraft is in sunlight, Earth umbra/penumbra, or Moon umbra/penumbra. Outputs
    eclipse durations and can optionally save time-series results.
    """

    # -------------------------
    # Setup
    # -------------------------

    utc = TimeScalesFactory.getUTC()
    inertial_frame = FramesFactory.getEME2000()

    earth_fixed_frame = FramesFactory.getITRF(
        IERSConventions.IERS_2010,
        True
    )

    start_date = AbsoluteDate(*start_date_tuple, utc)
    end_date = AbsoluteDate(*end_date_tuple, utc)

    total_duration_seconds = end_date.durationFrom(start_date)

    if total_duration_seconds <= 0:
        raise ValueError("end_date must be after start_date")

    mu = Constants.EIGEN5C_EARTH_MU

    sun = CelestialBodyFactory.getSun()
    moon = CelestialBodyFactory.getMoon()

    sun_radius = 696_000_000.0
    moon_radius = 1_737_400.0

    earth = OneAxisEllipsoid(
        Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
        Constants.WGS84_EARTH_FLATTENING,
        earth_fixed_frame
    )

    moon_body = OneAxisEllipsoid(
        moon_radius,
        0.0,
        moon.getBodyOrientedFrame()
    )

    # -------------------------
    # Orbit
    # -------------------------

    orbit = KeplerianOrbit(
        semi_major_axis_m,
        eccentricity,
        math.radians(inclination_deg),
        math.radians(argument_of_perigee_deg),
        math.radians(raan_deg),
        math.radians(true_anomaly_deg),
        PositionAngleType.TRUE,
        inertial_frame,
        start_date,
        mu
    )

    propagator = KeplerianPropagator(orbit)

    print(f"Orbital period: {orbit.getKeplerianPeriod() / 3600:.2f} hours")
    print(f"Simulation duration: {total_duration_seconds / 3600:.2f} hours")
    print(f"Timestep: {timestep_seconds:.2f} seconds")

    # -------------------------
    # Eclipse detectors
    # -------------------------

    detectors = {
        "earth_umbra": EclipseDetector(sun, sun_radius, earth).withUmbra(),
        "earth_penumbra": EclipseDetector(sun, sun_radius, earth).withPenumbra(),
        "moon_umbra": EclipseDetector(sun, sun_radius, moon_body).withUmbra(),
        "moon_penumbra": EclipseDetector(sun, sun_radius, moon_body).withPenumbra(),
    }

    # -------------------------
    # Experiment loop
    # -------------------------

    num_steps = int(total_duration_seconds // timestep_seconds)

    time_counter = {
        "full_sunlight": 0.0,
        "earth_umbra": 0.0,
        "earth_penumbra": 0.0,
        "moon_umbra": 0.0,
        "moon_penumbra": 0.0,
        "any_umbra": 0.0,
        "any_penumbra": 0.0,
        "any_eclipse": 0.0,
    }

    rows = []

    for k in range(num_steps + 1):
        t = float(k * timestep_seconds)
        date = start_date.shiftedBy(t)

        state = propagator.propagate(date)
        pos = state.getPVCoordinates().getPosition()

        raw_flags = {}

        for name, detector in detectors.items():
            raw_flags[name] = detector.g(state) < 0.0

        flags = {
            "earth_umbra": raw_flags["earth_umbra"],
            "earth_penumbra": raw_flags["earth_penumbra"] and not raw_flags["earth_umbra"],
            "moon_umbra": raw_flags["moon_umbra"],
            "moon_penumbra": raw_flags["moon_penumbra"] and not raw_flags["moon_umbra"],
        }

        active_flags = [
            name
            for name, is_active in flags.items()
            if is_active
        ]

        if len(active_flags) == 0:
            status = "full_sunlight"
        elif len(active_flags) == 1:
            status = active_flags[0]
        else:
            status = "mixed_eclipse"

        grouped_flags = {
            "any_umbra": flags["earth_umbra"] or flags["moon_umbra"],
            "any_penumbra": flags["earth_penumbra"] or flags["moon_penumbra"],
            "any_eclipse": len(active_flags) > 0,
        }

        if k < num_steps:
            if len(active_flags) == 0:
                time_counter["full_sunlight"] += timestep_seconds

            for name in active_flags:
                time_counter[name] += timestep_seconds

            for name, is_active in grouped_flags.items():
                if is_active:
                    time_counter[name] += timestep_seconds

        row = {
            "t_seconds": t,
            "date": date.toString(),
            "x_m": pos.getX(),
            "y_m": pos.getY(),
            "z_m": pos.getZ(),
            "status": status,
        }

        for name, is_active in flags.items():
            row[name] = is_active

        for name, is_active in grouped_flags.items():
            row[name] = is_active

        for name, detector in detectors.items():
            row[f"{name}_g"] = detector.g(state)

        rows.append(row)

    df = pd.DataFrame(rows)

    # -------------------------
    # Statistics
    # -------------------------

    print("\nEclipse summary:")
    print("----------------")

    for key, seconds in time_counter.items():
        hours = seconds / 3600.0
        percent = 100.0 * seconds / total_duration_seconds

        print(f"{key:18s}: {hours:8.3f} hours  ({percent:6.2f}%)")

    if save_csv:
        df.to_csv(csv_filename, index=False)
        print(f"\nSaved results to {csv_filename}")

    return df, time_counter

if __name__ == "__main__":
    df, stats = run_orbital_placement(
    semi_major_axis_m=7000e3,      # ~LEO (~600-700 km altitude)
    eccentricity=0.0,
    inclination_deg=98.0,          # Sun-synchronous-like
    argument_of_perigee_deg=0.0,
    raan_deg=0.0,
    true_anomaly_deg=0.0,
    start_date_tuple=(2024, 6, 21, 0, 0, 0.0),   # near solstice (high beta)
    end_date_tuple=(2024, 6, 22, 0, 0, 0.0),
    timestep_seconds=10.0,
    save_csv=False,
)
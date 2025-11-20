# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Compose the Great Britain focused PyPSA network described in
``doc/gb-model/index.rst``.

The rule assembles the clustered PyPSA-Eur base network with GB-specific
artefacts (manual region shapes, neighbouring countries, adjusted grid
connection costs) so that downstream rules can import a consistent
``networks/composed_{clusters}.nc`` snapshot.
"""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypsa
import xarray as xr
from scipy.optimize import minimize_scalar

from scripts._helpers import configure_logging, set_scenario_config
from scripts.add_electricity import (
    add_missing_carriers,
    attach_conventional_generators,
    attach_hydro,
    flatten,
)

logger = logging.getLogger(__name__)


@dataclass
class CompositionContext:
    """Context for network composition containing paths and configuration."""

    resources_root: Path
    countries: tuple[str, ...]
    costs_path: Path
    costs_config: dict[str, Any]
    max_hours: dict[str, Any] | None


def create_context(
    network_path: str,
    costs_path: str,
    countries: list[str],
    costs_config: dict[str, Any],
    max_hours: dict[str, Any] | None,
) -> CompositionContext:
    """
    Create composition context from network path and configuration.

    Parameters
    ----------
    network_path : str
        Path to the input network file
    costs_path : str
        Path to the costs CSV file
    countries : list[str]
        List of country codes to include
    costs_config : dict
        Costs configuration dictionary
    max_hours : dict or None
        Maximum hours configuration

    Returns
    -------
    CompositionContext
        Context object with paths and configuration
    """
    resources_root = Path(network_path).parents[1]

    return CompositionContext(
        resources_root=resources_root,
        countries=tuple(countries),
        costs_path=Path(costs_path),
        costs_config=copy.deepcopy(costs_config),
        max_hours=max_hours,
    )


def _add_timeseries_data_to_network(
    pypsa_t_dict: dict, data: pd.DataFrame, attribute: str
) -> None:
    """
    Add/update timeseries data to a network attribute.

    This is a robust approach to add data to the network when it is not known whether there is already data attached to the attribute.
    Any existing columns in the network attribute that are also in the incoming data will be overwritten.
    All other columns will remain as-is.

    Args:
        pypsa_t_dict (dict): PyPSA network timeseries component dictionary (e.g., n.loads_t).
        data (pd.DataFrame): Timeseries data to add.
        attribute (str): Network timeseries attribute to update.
    """
    assert pypsa_t_dict[attribute].index.equals(data.index), (
        f"Snapshot indices do not match between network attribute {attribute} and data being added."
    )
    logger.info(
        "Updating network timeseries attribute '%s' with %d columns of data.",
        attribute,
        len(data.columns),
    )
    pypsa_t_dict[attribute] = (
        pypsa_t_dict[attribute]
        .loc[:, ~pypsa_t_dict[attribute].columns.isin(data.columns)]
        .join(data)
    )


def _load_powerplants(
    powerplants_path: str,
    year: int,
) -> pd.DataFrame:
    """
    Load powerplant data.

    Parameters
    ----------
    powerplants_path : str
        Path to powerplants CSV file
    year : int
        Year to filter powerplants

    Returns
    -------
    pd.DataFrame
        Powerplant data filtered by year
    """
    ppl = pd.read_csv(powerplants_path, index_col=0, dtype={"bus": "str"})
    ppl = ppl[ppl.build_year == year]
    ppl["max_hours"] = 0  # Initialize max_hours column

    return ppl


def _integrate_renewables(
    n: pypsa.Network,
    electricity_config: dict[str, Any],
    renewable_config: dict[str, Any],
    costs: pd.DataFrame,
    renewable_profiles: dict[str, str],
    ppl: pd.DataFrame,
    hydro_capacities_path: str | None,
) -> None:
    """
    Integrate renewable generators into the network.

    Parameters
    ----------
    network : pypsa.Network
        Network to modify
    electricity_config : dict
        Electricity configuration dictionary
    renewable_config : dict
        Renewable configuration dictionary
    clustering_config : dict
        Clustering configuration dictionary
    line_length_factor : float
        Line length multiplication factor
    costs : pd.DataFrame
        Cost data
    renewable_profiles : dict
        Mapping of carrier names to profile file paths
    powerplants_path : str
        Path to powerplants CSV file
    hydro_capacities_path : str or None
        Path to hydro capacities CSV file
    """
    renewable_carriers = list(electricity_config["renewable_carriers"])
    extendable_carriers = electricity_config["extendable_carriers"]

    if not renewable_carriers:
        logger.info("No renewable carriers configured; skipping integration")
        return

    if "hydro" not in renewable_carriers:
        return

    non_hydro_carriers = [
        carrier for carrier in renewable_carriers if carrier != "hydro"
    ]
    non_hydro_profiles = {
        k: v for k, v in renewable_profiles.items() if k != "profile_hydro"
    }

    if renewable_profiles:
        attach_wind_and_solar(
            n,
            costs,
            ppl,
            non_hydro_profiles,
            non_hydro_carriers,
            extendable_carriers,
        )

    if "hydro" in renewable_carriers:
        hydro_cfg = copy.deepcopy(renewable_config["hydro"])
        carriers = hydro_cfg.pop("carriers")

        attach_hydro(
            n,
            costs,
            ppl,
            renewable_profiles["profile_hydro"],
            hydro_capacities_path,
            carriers,
            **hydro_cfg,
        )


def add_gb_components(
    n: pypsa.Network,
    context: CompositionContext,
) -> pypsa.Network:
    """
    Add GB-specific components and filter to target countries.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify
    context : CompositionContext
        Composition context

    Returns
    -------
    pypsa.Network
        Modified network
    """
    if context.countries:
        keep = n.buses.country.isin(context.countries)
        drop = n.buses.index[~keep]
        if len(drop) > 0:
            logger.info("Removing %d buses outside target countries", len(drop))
            n.mremove("Bus", drop)

    meta = n.meta.setdefault("gb_model", {})
    if context.countries:
        meta["countries"] = list(context.countries)

    return n


def process_demand_data(
    annual_demand: str,
    clustered_demand_profile: str,
    eur_demand: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Process the demand data for a particular demand type

    Parameters
    ----------
    n : pypsa.Network
        Network to finalize
    demand_list: list[str]
        CSV paths for demand data for each demand type
    clustered_demand_profile_list: list[str]
        CSV paths for demand shape for each demand type
    eur_demand: pd.DataFrame
        European annual demand data
    year:
        Year used in the modelling
    """

    # Read the files
    demand = pd.read_csv(annual_demand, index_col=["bus", "year"])
    demand_all = pd.concat([demand, eur_demand])
    demand_profile = pd.read_csv(
        clustered_demand_profile, index_col=[0], parse_dates=True
    )

    # Group demand data by year and bus and filter the data for required year
    demand_this_year = demand_all.xs(year, level="year")

    # Filtering those buses that are present in both the dataframes
    if diff_bus := set(
        demand_this_year.index.get_level_values("bus")
    ).symmetric_difference(set(demand_profile.columns)):
        logger.warning(
            "The following buses are missing demand profile or annual demand data and will be ignored: %s",
            diff_bus,
        )

    # Scale the profile by the annual demand from FES
    load = demand_profile.drop(columns=diff_bus).mul(demand_this_year["p_set"])
    assert not load.isnull().values.any(), "NaN values found in processed load data"
    return load


def add_load(
    n: pypsa.Network,
    demands: dict[str, list[str]],
    eur_demand: str,
    year: int,
):
    """
    Add load as a timeseries to PyPSA network

    Parameters
    ----------
    n : pypsa.Network
        Network to finalize
    demands: dict[str, list[str]]
        Mapping of demand types to list of CSV paths for demand data (GB annual demand, profile shapes)
    eur_demand: str
        Path to European annual demand data CSV
    year:
        Year used in the modelling
    """
    eur_demand_df = pd.read_csv(eur_demand, index_col=["load_type", "bus", "year"])
    # Iterate through each demand type
    for demand_type, (annual_demand, clustered_demand_profile) in demands.items():
        # Process data for the demand type
        load = process_demand_data(
            annual_demand, clustered_demand_profile, eur_demand_df.xs(demand_type), year
        )

        # Add the load to pypsa Network
        suffix = f" {demand_type}"
        n.add("Load", load.columns + suffix, bus=load.columns)
        _add_timeseries_data_to_network(n.loads_t, load.add_suffix(suffix), "p_set")


def add_EVs(
    n: pypsa.Network,
    ev_data: dict[str, str],
    ev_params: dict[str, float],
    year: int,
):
    """
    Add EV load as a timeseries to PyPSA network

    Parameters
    ----------
    n : pypsa.Network
        Network to finalize
    ev_data: dict[str, str]
        Dictionary containing paths to EV data files:
            ev_demand_annual:
                CSV path for annual EV demand
            ev_demand_peak:
                CSV path for peak EV demand
            ev_demand_shape:
                CSV path for EV demand shape
            ev_storage_capacity:
                CSV path for EV storage capacity
            ev_smart_charging:
                CSV path for EV smart charging (DSR) data
            ev_v2g:
                CSV path for EV V2G data
    ev_params: dict[str, float]
        Dictionary containing EV profile adjustment parameters such as:
            relative_peak_tolerance: float
                Relative tolerance for peak load
            relative_energy_tolerance: float
                Relative tolerance for energy load
            upper_optimization_bound: float
                Upper bound for optimization
            lower_optimization_bound: float
                Lower bound for optimization
    year:
        Year used in the modelling
    """
    # Compute EV demand profile using demand shape, annual EV demand and peak EV demand
    ev_demand_profile = _estimate_ev_demand_profile(
        ev_data["ev_demand_shape"],
        ev_data["ev_demand_annual"],
        ev_data["ev_demand_peak"],
        year=year,
        ev_params=ev_params,
    )

    # Add EV bus
    n.add(
        "Bus",
        ev_demand_profile.columns,
        suffix=" EV",
        carrier="EV",
        x=n.buses.loc[ev_demand_profile.columns].x,
        y=n.buses.loc[ev_demand_profile.columns].y,
        country=n.buses.loc[ev_demand_profile.columns].country,
    )

    # Add the EV load to pypsa Network
    n.add(
        "Load",
        ev_demand_profile.columns,
        suffix=" EV",
        bus=ev_demand_profile.columns + " EV",
        carrier="EV",
    )
    _add_timeseries_data_to_network(
        n.loads_t, ev_demand_profile.add_suffix(" EV"), "p_set"
    )

    # Add EV unmanaged charging
    n.add(
        "Link",
        ev_demand_profile.columns,
        suffix=" EV unmanaged charging",
        bus0=ev_demand_profile.columns,
        bus1=ev_demand_profile.columns + " EV",
        p_nom=ev_demand_profile.max(),
        efficiency=1.0,
        carrier="EV unmanaged charging",
    )

    # Load EV storage data
    ev_storage_capacity = pd.read_csv(
        ev_data["ev_storage_capacity"], index_col=["bus", "year"]
    )
    ev_storage_capacity = ev_storage_capacity.xs(year, level="year")

    # Add EV storage buses
    n.add(
        "Bus",
        ev_storage_capacity.index,
        suffix=" EV store",
        carrier="EV store",
        x=n.buses.loc[ev_storage_capacity.index].x,
        y=n.buses.loc[ev_storage_capacity.index].y,
        country=n.buses.loc[ev_storage_capacity.index].country,
    )

    # Add the EV store to pypsa Network
    ev_dsm_profile = pd.read_csv(
        ev_data["ev_dsm_profile"], index_col=0, parse_dates=True
    )
    n.add(
        "Store",
        ev_storage_capacity.index,
        suffix=" EV store",
        bus=ev_storage_capacity.index + " EV store",
        e_nom=ev_storage_capacity["MWh"],
        e_cyclic=True,
        carrier="EV store",
        e_min_pu=ev_dsm_profile.loc[n.snapshots, ev_storage_capacity.index],
    )

    # Load EV dsr and V2G data
    ev_dsr = pd.read_csv(ev_data["ev_smart_charging"], index_col=["bus", "year"])
    ev_v2g = pd.read_csv(ev_data["ev_v2g"], index_col=["bus", "year"])

    # Filter data for the given year
    ev_dsr = ev_dsr.xs(year, level="year")
    ev_v2g = ev_v2g.xs(year, level="year")

    # Add the EV DSR to the PyPSA network
    n.add(
        "Link",
        ev_dsr.index,
        suffix=" EV DSR",
        bus0=ev_dsr.index + " EV store",
        bus1=ev_dsr.index + " EV",
        p_nom=ev_dsr["p_nom"].abs(),
        efficiency=1.0,
        carrier="EV DSR",
    )
    n.add(
        "Link",
        ev_dsr.index,
        suffix=" EV DSR reverse",
        bus0=ev_dsr.index + " EV",
        bus1=ev_dsr.index + " EV store",
        p_nom=ev_dsr["p_nom"].abs(),
        efficiency=1.0,
        carrier="EV DSR reverse",
    )

    # Add EV V2G to the PyPSA network
    n.add(
        "Link",
        ev_v2g.index,
        suffix=" EV V2G",
        bus0=ev_v2g.index + " EV store",
        bus1=ev_v2g.index,
        p_nom=ev_v2g["p_nom"].abs(),
        efficiency=1.0,
        carrier="EV V2G",
    )


def _normalize(series: pd.Series) -> pd.Series:
    """Normalize a pandas Series so that its sum equals 1."""
    normalized = series / series.sum()
    return normalized


def _transform_ev_profile_with_shape_adjustment(
    shape_series: pd.Series,
    peak_target: float,
    annual_target: float,
    relative_peak_tolerance: float,
    relative_energy_tolerance: float,
    upper_optimization_bound: float,
    lower_optimization_bound: float,
) -> pd.Series:
    """
    Transform EV profile to match both peak and annual targets by adjusting the shape.

    This function can squeeze (make peakier) or widen (make flatter) the profile
    to satisfy both constraints simultaneously.

    Parameters
    ----------
    shape_series : pd.Series
        Normalized EV demand shape (sum = 1)
    peak_target : float
        Target peak demand (MW)
    annual_target : float
        Target annual energy (MWh)

    Returns
    -------
    pd.Series
        Transformed profile satisfying both constraints
    """
    # Normalize input to ensure sum = 1
    normalized_shape = _normalize(shape_series)

    # Try simple scaling first
    simple_scaled = normalized_shape * annual_target
    scaled_peak = simple_scaled.max()

    # If simple scaling satisfies peak constraint, return it
    if np.isclose(scaled_peak, peak_target, rtol=relative_peak_tolerance, atol=0):
        return simple_scaled

    # Need to adjust the shape - use power transformation
    # Higher gamma = more peaked, lower gamma = flatter

    def objective_function(gamma):
        """Objective function to find optimal shape parameter."""
        if gamma <= 0:
            return float("inf")

        # Apply power transformation and scale to match annual target
        scaled = _normalize(normalized_shape**gamma) * annual_target

        # Check how well we satisfy both constraints
        peak_error = abs(scaled.max() - peak_target) / peak_target
        annual_error = abs(scaled.sum() - annual_target) / annual_target

        return peak_error + annual_error

    result = minimize_scalar(
        objective_function,
        bounds=(lower_optimization_bound, upper_optimization_bound),
        method="bounded",
    )
    optimal_gamma = result.x

    # Apply optimal transformation
    final_profile = _normalize(normalized_shape**optimal_gamma) * annual_target

    # Verify constraints
    final_peak = final_profile.max()
    final_annual = final_profile.sum()

    logger.info(
        "Gamma optimization result for bus %s: optimal_gamma=%.4f, final_peak=%.2f MW, target_peak=%.2f MW, final_annual=%.2f MWh, target_annual=%.2f MWh",
        shape_series.name,
        optimal_gamma,
        final_peak,
        peak_target,
        final_annual,
        annual_target,
    )

    assert np.isclose(final_peak, peak_target, rtol=relative_peak_tolerance, atol=0), (
        f"Peak constraint violated after optimization for bus {shape_series.name} - "
        f"Expected: {peak_target:.2f} MW, Obtained: {final_peak:.2f} MW"
    )

    assert np.isclose(
        final_annual, annual_target, rtol=relative_energy_tolerance, atol=0
    ), (
        f"Annual constraint violated after optimization for bus {shape_series.name} - "
        f"Expected: {annual_target:.2f} MWh, Obtained: {final_annual:.2f} MWh"
    )

    return final_profile


def _estimate_ev_demand_profile(
    ev_demand_shape_path: str,
    ev_demand_annual_path: str,
    ev_demand_peak_path: str,
    year: int,
    ev_params: dict[str, float],
) -> pd.DataFrame:
    """
    Estimate the EV demand profile for the given year using shape, annual and peak data.

    Parameters
    ----------
    ev_demand_shape_path : str
        CSV path for EV demand shape
    ev_demand_annual_path : str
        CSV path for annual EV demand
    ev_demand_peak_path : str
        CSV path for peak EV demand
    year : int
        Year used in the modelling
    ev_params: dict[str, float]
        Dictionary containing EV profile adjustment parameters

    Returns
    -------
    pd.DataFrame
        Estimated EV demand profile
    """
    # Load the files
    ev_demand_shape = pd.read_csv(ev_demand_shape_path, index_col=[0], parse_dates=True)
    ev_demand_annual = pd.read_csv(ev_demand_annual_path, index_col=["bus", "year"])
    ev_demand_peak = pd.read_csv(ev_demand_peak_path, index_col=["bus", "year"])

    # Select data for given year
    ev_demand_annual = ev_demand_annual.xs(year, level="year")
    ev_demand_peak = ev_demand_peak.xs(year, level="year")

    # Affine transformation to scale the maximum with peak and total energy with annual demand
    ev_demand_profile = pd.DataFrame(index=ev_demand_shape.index)
    for bus in ev_demand_shape.columns:
        if bus not in ev_demand_annual.index or bus not in ev_demand_peak.index:
            continue

        peak = ev_demand_peak.loc[bus, "p_nom"]
        annual = ev_demand_annual.loc[bus, "p_set"]

        # Use shape-adjusting transformation to satisfy both peak and annual constraints
        ev_demand_profile[bus] = _transform_ev_profile_with_shape_adjustment(
            ev_demand_shape[bus],
            peak,
            annual,
            relative_peak_tolerance=ev_params["relative_peak_tolerance"],
            relative_energy_tolerance=ev_params["relative_energy_tolerance"],
            upper_optimization_bound=ev_params["upper_optimization_bound"],
            lower_optimization_bound=ev_params["lower_optimization_bound"],
        )

    logger.info(
        "EV unmanaged charging demand profile successfully generated with both peak and annual constraints satisfied."
    )

    return ev_demand_profile


def finalise_composed_network(
    n: pypsa.Network,
    context: CompositionContext,
) -> pypsa.Network:
    """
    Finalize network composition with topology and consistency checks.

    Parameters
    ----------
    n : pypsa.Network
        Network to finalize
    context : CompositionContext
        Composition context

    Returns
    -------
    pypsa.Network
        Finalized network
    """
    n.determine_network_topology()
    meta = n.meta.setdefault("gb_model", {})
    meta["resources_root"] = str(context.resources_root)
    meta["composed"] = True
    n.consistency_check()
    return n


def attach_chp_constraints(n: pypsa.Network, p_min_pu: pd.DataFrame) -> None:
    """
    Attach CHP operating constraints to the network.

    Args:
        n (pypsa.Network): The PyPSA network
        p_min_pu (pd.DataFrame): Minimum operation profile for CHP generators
    """
    chp_generators = n.generators[n.generators["set"] == "CHP"]

    if chp_generators.empty:
        logger.info(
            "No CHP generators found in the network. "
            f"Total generators: {len(n.generators)}, "
            f"generators by set: {n.generators.groupby('set').size().to_dict()}"
        )
        return

    logger.info(
        f"Applying CHP constraints to {len(chp_generators)} generators "
        f"with total capacity {chp_generators.p_nom.sum():.1f} MW"
    )

    # Map minimum operation to generators (vectorized)
    # Each generator inherits the profile of its bus
    gen_to_bus = chp_generators["bus"]

    # Filter to only generators with available heat demand data
    valid_gens = gen_to_bus[gen_to_bus.isin(p_min_pu.columns)]
    missing_gens = gen_to_bus[~gen_to_bus.isin(p_min_pu.columns)]

    if not missing_gens.empty:
        logger.warning(
            f"No heat demand data for {len(missing_gens)} generators at buses: {list(missing_gens.unique())}. "
            "These generators will have no CHP constraint."
        )

    # Vectorized assignment: rename p_min_pu columns from bus names to generator indices
    # Select columns for each generator's bus, then rename to generator index
    p_min_pu_for_gens = p_min_pu[valid_gens.values].copy()
    p_min_pu_for_gens.columns = valid_gens.index

    # Assign all generators at once
    _add_timeseries_data_to_network(n.generators_t, p_min_pu_for_gens, "p_min_pu")


def attach_wind_and_solar(
    n: pypsa.Network,
    costs: pd.DataFrame,
    ppl: pd.DataFrame,
    profile_filenames: dict,
    carriers: list | set,
    extendable_carriers: list | set,
) -> None:
    """
    Attach wind and solar generators to the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to attach the generators to.
    costs : pd.DataFrame
        DataFrame containing the cost data.
    ppl : pd.DataFrame
        DataFrame containing the power plant data.
    profile_filenames : dict
        Dictionary containing the paths to the wind and solar profiles.
    carriers : list | set
        List of renewable energy carriers to attach.
    extendable_carriers : list | set
        List of extendable renewable energy carriers.
    """
    add_missing_carriers(n, carriers)

    for car in carriers:
        if car == "hydro":
            continue

        with xr.open_dataset(profile_filenames["profile_" + car]) as ds:
            if ds.indexes["bus"].empty:
                continue

            # if-statement for compatibility with old profiles
            if "year" in ds.indexes:
                ds = ds.sel(year=ds.year.min(), drop=True)

            ds = ds.stack(bus_bin=["bus", "bin"])

            supcar = car.split("-", 2)[0]
            capital_cost = costs.at[supcar, "capital_cost"]

            buses = ds.indexes["bus_bin"].get_level_values("bus")
            bus_bins = ds.indexes["bus_bin"].map(flatten)

            p_nom_max = ds["p_nom_max"].to_pandas()
            p_nom_max.index = p_nom_max.index.map(flatten)

            p_max_pu = ds["profile"].to_pandas()
            p_max_pu.columns = p_max_pu.columns.map(flatten)

            if not ppl.query("carrier == @supcar").empty:
                caps = ppl.query("carrier == @supcar").groupby("bus").p_nom.sum()
                caps = caps.reindex(buses).fillna(0)
                caps = pd.Series(data=caps.values, index=bus_bins)
            else:
                caps = pd.Series(index=bus_bins).fillna(0)

            n.add(
                "Generator",
                bus_bins,
                suffix=" " + supcar,
                bus=buses,
                carrier=supcar,
                p_nom=caps,
                p_nom_min=caps,
                p_nom_extendable=car in extendable_carriers["Generator"],
                p_nom_max=p_nom_max,
                marginal_cost=costs.at[supcar, "marginal_cost"],
                capital_cost=capital_cost,
                efficiency=costs.at[supcar, "efficiency"],
                p_max_pu=p_max_pu,
                lifetime=costs.at[supcar, "lifetime"],
            )


def _prepare_costs(
    ppl: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """
    Prepare costs DataFrame from powerplant data.
    """
    costs = ppl[ppl.build_year == year]
    costs = costs[~costs.set_index("carrier").index.duplicated(keep="first")].set_index(
        "carrier"
    )
    costs = costs[
        [
            "set",
            "capital_cost",
            "marginal_cost",
            "lifetime",
            "efficiency",
            "CO2 intensity",
        ]
    ]
    return costs


def compose_network(
    network_path: str,
    output_path: str,
    costs_path: str,
    powerplants_path: str,
    hydro_capacities_path: str | None,
    chp_p_min_pu_path: str,
    renewable_profiles: dict[str, str],
    countries: list[str],
    costs_config: dict[str, Any],
    electricity_config: dict[str, Any],
    clustering_config: dict[str, Any],
    renewable_config: dict[str, Any],
    lines_config: dict[str, Any],
    demands: dict[str, list[str]],
    eur_demand: str,
    year: int,
    enable_chp: bool,
    ev_data: dict[str, str],
    ev_params: dict[str, float],
) -> None:
    """
    Main composition function to create GB market model network.

    Parameters
    ----------
    network_path : str
        Path to input base network
    output_path : str
        Path to save composed network
    costs_path : str
        Path to costs CSV file
    powerplants_path : str
        Path to powerplants CSV file
    hydro_capacities_path : str or None
        Path to hydro capacities CSV file
    renewable_profiles : dict
        Mapping of carrier names to profile file paths
    heat_demand_path : str
        Path to hourly heat demand NetCDF file for CHP constraints
    countries : list[str]
        List of country codes to include
    costs_config : dict
        Costs configuration dictionary
    electricity_config : dict
        Electricity configuration dictionary
    clustering_config : dict
        Clustering configuration dictionary
    renewable_config : dict
        Renewable configuration dictionary
    lines_config : dict
        Lines configuration dictionary
    demand: list[str]
        List of paths to the demand data for each demand type
    clustered_demand_profile: list[str]
        List of paths to the clustered shape profile for each demand type
    demand_types: list[str]
        List of str for demand types
    year: int
        Modelling year
    enable_chp : bool
        Whether to enable CHP constraints
    ev_data : dict[str, str]
        Dictionary containing EV demand and flexibility data
    ev_params : dict[str, float]
        Dictionary containing EV profile adjustment parameters
    """
    network = pypsa.Network(network_path)
    max_hours = electricity_config["max_hours"]
    context = create_context(
        network_path, costs_path, countries, costs_config, max_hours
    )
    add_gb_components(network, context)

    # Load FES powerplants data (already enriched with costs from create_powerplants_table)
    ppl = _load_powerplants(powerplants_path, year)

    # Define costs file
    costs = _prepare_costs(ppl, year)

    _integrate_renewables(
        network,
        electricity_config,
        renewable_config,
        costs,
        renewable_profiles,
        ppl,
        hydro_capacities_path,
    )

    conventional_carriers = list(electricity_config["conventional_carriers"])

    attach_conventional_generators(
        network,
        costs,
        ppl,
        conventional_carriers,
        extendable_carriers={"Generator": []},
        conventional_params={},
        conventional_inputs={},
        unit_commitment=None,
    )

    # Add simplified CHP constraints if enabled
    if enable_chp:
        logger.info("Adding simplified CHP constraints based on heat demand.")
        chp_p_min_pu = pd.read_csv(
            chp_p_min_pu_path, index_col="snapshot", parse_dates=True
        )
        attach_chp_constraints(network, chp_p_min_pu)

    add_load(network, demands, eur_demand, year)

    add_EVs(network, ev_data, ev_params, year)

    finalise_composed_network(network, context)

    network.export_to_netcdf(output_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("compose_network", clusters="clustered", year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Extract renewable profiles from inputs
    renewable_carriers = snakemake.params.electricity["renewable_carriers"]
    renewable_profile_keys = [f"profile_{carrier}" for carrier in renewable_carriers]
    renewable_profiles = {key: snakemake.input[key] for key in renewable_profile_keys}
    demands = {
        k.replace("demand_", ""): v
        for k, v in snakemake.input.items()
        if k.startswith("demand_")
    }
    ev_data = {
        "ev_demand_annual": snakemake.input.ev_demand_annual,
        "ev_demand_shape": snakemake.input.ev_demand_shape,
        "ev_demand_peak": snakemake.input.ev_demand_peak,
        "ev_storage_capacity": snakemake.input.ev_storage_capacity,
        "ev_smart_charging": snakemake.input.regional_fes_ev_dsm,
        "ev_v2g": snakemake.input.regional_fes_ev_v2g,
        "ev_dsm_profile": snakemake.input.ev_dsm_profile,
    }
    ev_params = {
        "relative_peak_tolerance": snakemake.params.ev_profile_config[
            "relative_peak_tolerance"
        ],
        "relative_energy_tolerance": snakemake.params.ev_profile_config[
            "relative_energy_tolerance"
        ],
        "upper_optimization_bound": snakemake.params.ev_profile_config[
            "upper_optimization_bound"
        ],
        "lower_optimization_bound": snakemake.params.ev_profile_config[
            "lower_optimization_bound"
        ],
    }
    compose_network(
        network_path=snakemake.input.network,
        output_path=snakemake.output.network,
        costs_path=snakemake.input.tech_costs,
        powerplants_path=snakemake.input.powerplants,
        hydro_capacities_path=snakemake.input.hydro_capacities,
        renewable_profiles=renewable_profiles,
        chp_p_min_pu_path=snakemake.input.chp_p_min_pu,
        eur_demand=snakemake.input.eur_demand,
        countries=snakemake.params.countries,
        costs_config=snakemake.params.costs_config,
        electricity_config=snakemake.params.electricity,
        clustering_config=snakemake.params.clustering,
        renewable_config=snakemake.params.renewable,
        lines_config=snakemake.params.lines,
        demands=demands,
        year=int(snakemake.wildcards.year),
        enable_chp=snakemake.params.enable_chp,
        ev_data=ev_data,
        ev_params=ev_params,
    )

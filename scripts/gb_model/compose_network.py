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

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config
from scripts.add_electricity import (
    attach_conventional_generators,
    attach_hydro,
    attach_wind_and_solar,
    load_and_aggregate_powerplants,
    load_costs,
    sanitize_carriers,
    sanitize_locations,
)
from scripts.prepare_sector_network import add_electricity_grid_connection

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
    n: pypsa.Network, data: pd.DataFrame, attribute: str
) -> None:
    """
    Add/update timeseries data to a network attribute.

    This is a robust approach to add data to the network when it is not known whether there is already data attached to the attribute.
    Any existing columns in the network attribute that are also in the incoming data will be overwritten.
    All other columns will remain as-is.

    Args:
        n (pypsa.Network): PyPSA network.
        data (pd.DataFrame): Timeseries data to add.
        attribute (str): Network timeseries attribute to update.
    """
    assert n.generators_t[attribute].index.equals(data.index), (
        f"Snapshot indices do not match between network attribute {attribute} and data being added."
    )
    logger.info(
        "Updating network timeseries attribute '%s' with %d columns of data.",
        attribute,
        len(data.columns),
    )
    n.generators_t[attribute] = (
        n.generators_t[attribute]
        .loc[:, ~n.generators_t[attribute].columns.isin(data.columns)]
        .join(data)
    )


def load_powerplants(
    powerplants_path: str,
    costs: pd.DataFrame | None,
    clustering_config: dict[str, Any],
) -> pd.DataFrame:
    """
    Load and aggregate powerplant data.

    Parameters
    ----------
    powerplants_path : str
        Path to powerplants CSV file
    costs : pd.DataFrame or None
        Cost data DataFrame
    clustering_config : dict
        Clustering configuration dictionary

    Returns
    -------
    pd.DataFrame
        Aggregated powerplant data
    """
    consider_efficiency = clustering_config["consider_efficiency_classes"]
    aggregation_strategies = clustering_config["aggregation_strategies"]
    exclude_carriers = clustering_config["exclude_carriers"]

    return load_and_aggregate_powerplants(
        powerplants_path,
        costs,
        consider_efficiency_classes=consider_efficiency,
        aggregation_strategies=aggregation_strategies,
        exclude_carriers=exclude_carriers,
    )


def integrate_renewables(
    n: pypsa.Network,
    electricity_config: dict[str, Any],
    renewable_config: dict[str, Any],
    clustering_config: dict[str, Any],
    line_length_factor: float,
    costs: pd.DataFrame,
    renewable_profiles: dict[str, str],
    powerplants_path: str,
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

    # Load FES powerplants data (already enriched with costs from create_powerplants_table)
    ppl = pd.read_csv(powerplants_path, index_col=0, dtype={"bus": "str"})

    if renewable_profiles:
        landfall_lengths = {
            tech: settings["landfall_length"]
            for tech, settings in renewable_config.items()
            if isinstance(settings, dict) and "landfall_length" in settings
        }

        attach_wind_and_solar(
            n,
            costs,
            ppl,
            non_hydro_profiles,
            non_hydro_carriers,
            extendable_carriers,
            line_length_factor,
            landfall_lengths,
        )

    if "hydro" not in renewable_profiles:
        logger.warning("Hydro profile not available; skipping hydro integration")
        return

    if hydro_capacities_path is None:
        logger.warning("Hydro capacities file missing; skipping hydro integration")
        return

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


def add_pypsaeur_components(
    n: pypsa.Network,
    electricity_config: dict[str, Any],
    context: CompositionContext,
    costs: pd.DataFrame | None,
) -> pypsa.Network:
    """
    Add PyPSA-Eur components like grid connections and sanitize network.

    Parameters
    ----------
    n : pypsa.Network
        Network to modify
    electricity_config : dict
        Electricity configuration dictionary
    context : CompositionContext
        Composition context
    costs : pd.DataFrame or None
        Cost data

    Returns
    -------
    pypsa.Network
        Modified network
    """
    if costs is not None:
        add_electricity_grid_connection(n, costs)
        n.meta.setdefault("gb_model", {})["costs_path"] = str(context.costs_path)

    sanitize_locations(n)
    try:
        # Pass full config dict for sanitize_carriers (it needs various config sections)
        full_config = {"electricity": electricity_config}
        sanitize_carriers(n, full_config)
    except KeyError as exc:  # pragma: no cover - tolerate partial configs
        logger.debug("Skipping carrier sanitisation due to missing config: %s", exc)
    return n


def process_demand_data(
    demand_list: list[str],
    clustered_demand_profile_list: list[str],
    demand_type: list[str],
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
    demand_type:
        demand type for which data is to be processed
    year:
        Year used in the modelling
    """

    # Filter file path for demand type
    demand_path = [x for x in demand_list if demand_type in x][0]
    demand_profile_path = [
        x for x in clustered_demand_profile_list if demand_type in x
    ][0]

    # Read the files
    demand = pd.read_csv(demand_path)
    demand_profile = pd.read_csv(demand_profile_path, index_col=[0])

    # Group demand data by year and bus and filter the data for required year
    demand_grouped = demand.groupby(["year", "bus"]).sum().loc[year]

    # Filtering those buses that are present in both the dataframes
    list_of_buses = list(set(demand["bus"]) & set(demand_profile.columns))

    # Scale the profile by the annual demand from FES
    load = demand_profile[list_of_buses].mul(demand_grouped["p_set"])

    # Convert load index to datetime dtype to avoid flagging an assertion error from pypsa
    load.index = pd.to_datetime(load.index)

    return load


def add_load(
    n: pypsa.Network,
    demand_list: list,
    clustered_demand_profile_list: list,
    demand_types: list,
    year: int,
):
    """
    Add load as a timeseries to PyPSA network

    Parameters
    ----------
    n : pypsa.Network
        Network to finalize
    demand_list: list[str]
        CSV paths for demand data for each demand type
    clustered_demand_profile_list: list[str]
        CSV paths for demand shape for each demand type
    demand_types:
        Keywords to map the demand files to each demand type
    year:
        Year used in the modelling
    """

    # Iterate through each demand type
    for demand_type in demand_types:
        # Process data for the demand type
        load = process_demand_data(
            demand_list, clustered_demand_profile_list, demand_type, year
        )

        # Add the load to pypsa Network
        suffix = f" {demand_type}"
        n.add("Load", load.columns + suffix, bus=load.columns)
        n.loads_t.p_set = pd.concat([n.loads_t.p_set, load.add_suffix(suffix)], axis=1)


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
    _add_timeseries_data_to_network(n, p_min_pu_for_gens, "p_min_pu")


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
    demand: list[str],
    clustered_demand_profile: list[str],
    demand_types: list[str],
    year: int,
    enable_chp: bool,
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
    """
    network = pypsa.Network(network_path)
    max_hours = electricity_config["max_hours"]
    context = create_context(
        network_path, costs_path, countries, costs_config, max_hours
    )
    add_gb_components(network, context)

    costs = None
    if context.costs_path.exists():
        weights = network.snapshot_weightings.objective
        nyears = float(weights.sum()) / 8760.0
        costs = load_costs(
            str(context.costs_path),
            context.costs_config,
            max_hours=context.max_hours,
            nyears=nyears,
        )

    line_length_factor = lines_config["length_factor"]
    integrate_renewables(
        network,
        electricity_config,
        renewable_config,
        clustering_config,
        line_length_factor,
        costs,
        renewable_profiles,
        powerplants_path,
        hydro_capacities_path,
    )

    conventional_carriers = list(electricity_config["conventional_carriers"])
    # Load FES powerplants data (already enriched with costs from create_powerplants_table)
    ppl = pd.read_csv(powerplants_path, index_col=0, dtype={"bus": "str"})
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

    add_pypsaeur_components(network, electricity_config, context, costs)

    add_load(network, demand, clustered_demand_profile, demand_types, year)

    finalise_composed_network(network, context)

    network.export_to_netcdf(output_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("compose_network", clusters=100)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Extract renewable profiles from inputs
    renewable_carriers = snakemake.params.electricity["renewable_carriers"]
    renewable_profile_keys = [f"profile_{carrier}" for carrier in renewable_carriers]
    renewable_profiles = {key: snakemake.input[key] for key in renewable_profile_keys}

    compose_network(
        network_path=snakemake.input.network,
        output_path=snakemake.output.network,
        costs_path=snakemake.input.tech_costs,
        powerplants_path=snakemake.input.powerplants,
        hydro_capacities_path=snakemake.input.hydro_capacities,
        renewable_profiles=renewable_profiles,
        chp_p_min_pu_path=snakemake.input.chp_p_min_pu,
        countries=snakemake.params.countries,
        costs_config=snakemake.params.costs_config,
        electricity_config=snakemake.params.electricity,
        clustering_config=snakemake.params.clustering,
        renewable_config=snakemake.params.renewable,
        lines_config=snakemake.params.lines,
        demand=snakemake.input.demand,
        clustered_demand_profile=snakemake.input.clustered_demand_profile,
        demand_types=snakemake.params.demand_types,
        year=int(snakemake.wildcards.year),
        enable_chp=snakemake.params.enable_chp,
    )

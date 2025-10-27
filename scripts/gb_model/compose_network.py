# SPDX-FileCopyrightText: gb-open-market-model contributors
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
from scripts.gb_model.chp_utils import attach_chp_constraints
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


def compose_network(
    network_path: str,
    output_path: str,
    costs_path: str,
    powerplants_path: str,
    hydro_capacities_path: str | None,
    renewable_profiles: dict[str, str],
    heat_demand_path: str,
    countries: list[str],
    costs_config: dict[str, Any],
    electricity_config: dict[str, Any],
    clustering_config: dict[str, Any],
    renewable_config: dict[str, Any],
    lines_config: dict[str, Any],
    chp_config: dict[str, Any],
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
    chp_config : dict
        CHP configuration with required keys:
        - 'enable': bool
        - 'heat_to_power_ratio': float
        - 'min_operation_level': float
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
    if chp_config["enable"]:
        logger.info("Adding simplified CHP constraints based on heat demand.")
        attach_chp_constraints(
            network,
            ppl,
            heat_demand_path=heat_demand_path,
            heat_to_power_ratio=chp_config["heat_to_power_ratio"],
            min_operation_level=chp_config["min_operation_level"],
        )

    add_pypsaeur_components(network, electricity_config, context, costs)
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
        heat_demand_path=snakemake.input.hourly_heat_demand_total,
        countries=snakemake.params.countries,
        costs_config=snakemake.params.costs_config,
        electricity_config=snakemake.params.electricity,
        clustering_config=snakemake.params.clustering,
        renewable_config=snakemake.params.renewable,
        lines_config=snakemake.params.lines,
        chp_config=snakemake.params.chp,
    )

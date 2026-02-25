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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pypsa
import xarray as xr

from scripts._helpers import configure_logging, set_scenario_config
from scripts.add_electricity import (
    add_missing_carriers,
    attach_conventional_generators,
    attach_hydro,
    flatten,
)
from scripts.gb_model._helpers import time_difference_hours

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


def _input_list_to_dict(input_list: list[str], parent: bool = False) -> dict[str, str]:
    """
    Convert a list of input file paths to a dictionary mapping types to paths.

    Args:
        input_list (list[str]): List of input file paths.
        parent (bool): Whether to use the parent directory name as the key.

    Returns:
        dict[str, str]: Dictionary mapping input types to file paths.
    """
    input_dict = {}
    for input_path in input_list:
        key = (Path(input_path).parent if parent else Path(input_path)).stem
        input_dict[key] = input_path
    return input_dict


def _add_timeseries_data_to_network(
    pypsa_t_dict: dict, data: pd.DataFrame, attribute: str, conflicts: str = "overwrite"
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
        conflicts (str, optional): How to handle conflicting columns ('overwrite' or 'mul'). Defaults to 'overwrite'.
    """
    assert pypsa_t_dict[attribute].index.equals(data.index), (
        f"Snapshot indices do not match between network attribute {attribute} and data being added."
    )
    logger.info(
        "Updating network timeseries attribute '%s' with %d columns of data.",
        attribute,
        len(data.columns),
    )
    conflict_cols = pypsa_t_dict[attribute].columns.intersection(data.columns)
    updated_data = (
        pypsa_t_dict[attribute]
        .loc[:, ~pypsa_t_dict[attribute].columns.isin(conflict_cols)]
        .join(data.loc[:, ~data.columns.isin(conflict_cols)])
    )
    if conflict_cols.any():
        logger.info(
            "Conflicting columns found when updating network attribute '%s': %s",
            attribute,
            list(conflict_cols),
        )
        match conflicts:
            case "overwrite":
                updated_data = updated_data.join(data.loc[:, conflict_cols])
            case "mul":
                multiplied = (
                    pypsa_t_dict[attribute]
                    .loc[:, conflict_cols]
                    .mul(data.loc[:, conflict_cols], fill_value=1)
                )
                updated_data = pd.concat(
                    [updated_data, multiplied],
                    axis=1,
                )
    if not all(pypsa_t_dict[attribute].columns.isin(updated_data.columns)):
        logger.warning(
            "Some columns lost when updating network attribute %s.",
            attribute,
        )
    pypsa_t_dict[attribute] = updated_data


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
    ppl = pd.read_csv(powerplants_path, index_col="name", dtype={"bus": "str"})
    ppl = ppl[ppl.build_year == year]
    ppl["max_hours"] = 0  # Initialize max_hours column
    ppl["efficiency"] = 1.0  # Initialize efficiency column

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
    n: pypsa.Network, demands: dict[str, str], load_bus_suffixes: dict[str, str]
):
    """
    Add load as a timeseries to PyPSA network

    Parameters
    ----------
    n : pypsa.Network
        Network to finalize
    demands: dict[str, str]
        Mapping of demand types to CSV paths for demand data
    """
    # Iterate through each demand type
    for demand_type, demand_path in demands.items():
        if not demand_type.endswith("_demand"):
            raise ValueError(
                f"Unexpected demand type: {demand_type}. "
                "All demand types should start with 'demand_'."
            )
        # Process data for the demand type
        suffix = load_bus_suffixes[demand_type.removesuffix("_demand")]
        load = pd.read_csv(demand_path, index_col=[0], parse_dates=True)
        _add_load_bus(n, load, suffix)


def _add_load_bus(n: pypsa.Network, load: pd.DataFrame, suffix: str):
    """
    Add EV load as a timeseries to PyPSA network

    Parameters
    ----------
    n : pypsa.Network
        Network to finalize
    load : pd.DataFrame
        DataFrame containing load data indexed by time and bus
    suffix : str
        Suffix to append to load buses
    """
    # Add load carrier to pypsa Network
    carrier = suffix.strip()
    link_carrier = f"{carrier} unmanaged load"
    n.add("Carrier", carrier)

    # Add unmanaged load carrier to pypsa Network
    n.add("Carrier", link_carrier)

    # Add load bus
    n.add(
        "Bus",
        load.columns,
        suffix=suffix,
        carrier=carrier,
        x=n.buses.loc[load.columns].x,
        y=n.buses.loc[load.columns].y,
        country=n.buses.loc[load.columns].country,
    )

    # Add the load to pypsa Network
    n.add(
        "Load",
        load.columns,
        suffix=suffix,
        bus=load.columns + suffix,
        carrier=carrier,
        p_set=load.add_suffix(suffix),
    )

    # Link the load profile to the AC bus
    n.add(
        "Link",
        load.columns,
        suffix=" " + link_carrier,
        bus0=load.columns,
        bus1=load.columns + suffix,
        p_nom=load.max(),
        efficiency=1.0,
        carrier=link_carrier,
    )


def _load_regional_data(path: str, year: int) -> pd.DataFrame:
    """
    Load regional data from CSV file and filter by year.
    """
    df = pd.read_csv(path, index_col=["bus", "year"])
    df_year = df.xs(year, level="year")
    return df_year


def add_EV_V2G(
    n: pypsa.Network,
    year: int,
    regional_ev_v2g_inc_eur: str,
    ev_v2g_storage_capacity_path: str,
    ev_availability_profile: pd.DataFrame,
):
    """
    Add EV DSR and V2G components to PyPSA network

    Parameters
    ----------
    n : pypsa.Network
        Network to finalize
    year: int
        Year used in the modelling
    regional_ev_v2g_inc_eur: str
        CSV path for EV V2G data
    ev_v2g_storage_capacity_path : str
        Path to EV storage capacity CSV
    ev_avail_profile : pd.DataFrame
        DataFrame containing EV availability profile
    """
    # Load EV V2G data
    ev_v2g_df = _load_regional_data(regional_ev_v2g_inc_eur, year)
    ev_v2g_storage_capacity = _load_regional_data(ev_v2g_storage_capacity_path, year)

    # Add EV V2G carrier to the PyPSA network
    n.add(
        "Carrier",
        "ev V2G",
    )

    # Add EV V2G bus to the PyPSA network
    n.add(
        "Bus",
        ev_v2g_df.index,
        suffix=" EV V2G bus",
        carrier="ev V2G",
        x=n.buses.loc[ev_v2g_df.index].x,
        y=n.buses.loc[ev_v2g_df.index].y,
        country=n.buses.loc[ev_v2g_df.index].country,
    )

    # Add link from EV bus to V2G bus to the PyPSA network
    n.add(
        "Link",
        ev_v2g_df.index,
        suffix=" EV V2G feed-in",
        bus0=ev_v2g_df.index + " EV",
        bus1=ev_v2g_df.index + " EV V2G bus",
        p_nom=ev_v2g_df.p_nom.abs(),
        p_max_pu=ev_availability_profile.loc[:, ev_v2g_df.index],
        efficiency=1.0,
        carrier="ev V2G",
    )

    # Add link from V2G bus to AC bus to the PyPSA network
    n.add(
        "Link",
        ev_v2g_df.index,
        suffix=" EV V2G",
        bus0=ev_v2g_df.index + " EV V2G bus",
        bus1=ev_v2g_df.index,
        p_nom=ev_v2g_df.p_nom.abs(),
        p_max_pu=ev_availability_profile.loc[:, ev_v2g_df.index],
        efficiency=1.0,
        carrier="ev V2G",
    )

    # Add the EV V2G store to the PyPSA network
    n.add(
        "Store",
        ev_v2g_df.index,
        suffix=" EV V2G store",
        bus=ev_v2g_df.index + " EV V2G bus",
        e_nom=ev_v2g_storage_capacity.e_nom,
        e_cyclic=True,
        carrier="ev V2G",
    )


def _get_dsr_profile(
    n: pypsa.Network, df: pd.DataFrame, dsr_hours: list[int], key: str
) -> pd.DataFrame:
    """
    Create DSR profile to set as e_max_pu for DSR store

    Parameters
    ----------
        n : pypsa.Network
            Network to finalize
        df : pd.DataFrame
            DataFrame containing flexibility p_nom data indexed by bus
        dsr_hours : list[int]
            Hours during which demand-side management can occur
        key : str
            Sector key (e.g., 'residential', 'iandc', 'iandc_heat', 'ev')
    """

    dsr_profile = pd.DataFrame(index=n.snapshots, columns=df.index, data=0.0)
    if dsr_hours[0] < dsr_hours[1]:
        mask = (dsr_profile.index.hour >= dsr_hours[0]) & (
            dsr_profile.index.hour <= dsr_hours[1]
        )
    else:
        mask = (dsr_profile.index.hour <= dsr_hours[1]) | (
            dsr_profile.index.hour >= dsr_hours[0]
        )
    dsr_profile.loc[mask] = 1.0

    # Calculate time difference between each neighbouring country and GB
    time_shift = {
        x: time_difference_hours(x) for x in n.buses.country.unique().tolist()
    }

    # Shift European neighbour DSR by 'x' hours to account for time zone difference
    for country, shift_hours in time_shift.items():
        country_columns = dsr_profile.filter(like=country).columns
        dsr_profile.loc[:, country_columns] = dsr_profile[country_columns].shift(
            shift_hours, fill_value=0.0
        )

    logger.info(f"Created DSR profile for {key} sector with DSR hours {dsr_hours}")

    return dsr_profile


def _add_dsr_pypsa_components(
    n: pypsa.Network,
    df: pd.DataFrame,
    key: str,
    storage_capacity: pd.DataFrame,
    e_max_pu: pd.DataFrame,
    p_max_pu: pd.DataFrame,
    load_bus_suffixes: dict[str, str],
    flex_carrier_suffixes: dict[str, str],
):
    """
    Add DSR components for a given sector to PyPSA network

    Parameters
    ----------
        n : pypsa.Network
            Network to finalize
        df : pd.DataFrame
            DataFrame containing flexibility p_nom data indexed by bus
        key : str
            Sector key (e.g., 'residential', 'iandc', 'iandc_heat', "ev")
        storage_capacity : pd.DataFrame
            DataFrame containing storage capacity of the store in MWh indexed by time and bus
        e_max_pu : pd.DataFrame
            DataFrame containing maximum state of charge as per unit of the store indexed by time and bus
        p_max_pu : pd.DataFrame
            DataFrame containing maximum power availability as per unit of the link indexed by time and bus
        load_bus_suffixes: dict[str, str]
            Suffixes to append to load buses
        flex_carrier_suffixes : dict[str, str]
            Suffixes to append to flexibility carriers for each flexibility type
    """
    load_bus_suffix = load_bus_suffixes[key]
    dsr_carrier_suffix = flex_carrier_suffixes[key]

    # Add the store carrier to the PyPSA network
    n.add("Carrier", dsr_carrier_suffix.strip())

    # Add the DSR shift and reverse carriers to the PyPSA network
    n.add("Carrier", f"{dsr_carrier_suffix.strip()} shift")

    n.add("Carrier", f"{dsr_carrier_suffix.strip()} reverse")

    # Create storage buses, links and stores
    # Add the storage bus to the PyPSA network
    n.add(
        "Bus",
        df.index,
        suffix=dsr_carrier_suffix,
        carrier=dsr_carrier_suffix.strip(),
        x=n.buses.loc[df.index].x,
        y=n.buses.loc[df.index].y,
        country=n.buses.loc[df.index].country,
    )

    # Add the DSR link from AC bus to DSR bus to the PyPSA network
    n.add(
        "Link",
        df.index,
        suffix=dsr_carrier_suffix,
        bus0=df.index + load_bus_suffix,
        bus1=df.index + dsr_carrier_suffix,
        p_nom=df.p_nom.abs(),
        p_max_pu=p_max_pu,
        efficiency=1.0,
        carrier=f"{dsr_carrier_suffix.strip()} shift",
    )

    # Add the DSR link from DSR bus to AC bus to the PyPSA network
    n.add(
        "Link",
        df.index,
        suffix=f"{dsr_carrier_suffix} reverse",
        bus0=df.index + dsr_carrier_suffix,
        bus1=df.index + load_bus_suffix,
        p_nom=df.p_nom.abs(),
        p_max_pu=p_max_pu,
        efficiency=1.0,
        carrier=f"{dsr_carrier_suffix.strip()} reverse",
    )

    # Add the DSR store to the PyPSA network
    n.add(
        "Store",
        df.index,
        suffix=dsr_carrier_suffix,
        bus=df.index + dsr_carrier_suffix,
        e_nom=storage_capacity,
        e_cyclic=True,
        carrier=dsr_carrier_suffix.strip(),
        e_max_pu=e_max_pu,
    )

    logger.info(f"Added PyPSA components for {key} sector to perform DSR")


def add_DSR(
    n: pypsa.Network,
    year: int,
    dsr: dict[str, str],
    dsr_hours_dict: dict[str, list],
    ev_availability_profile: pd.DataFrame,
    load_bus_suffixes: dict[str, str],
    flex_carrier_suffixes: dict[str, str],
):
    """
    Add DSR components for residential, i&c and i&c heat sectors to PyPSA network

    Parameters
    ----------
        n : pypsa.Network
            Network to finalize
        year: int
            Year used in the modelling
        dsr: dict[str, str]
            Dictionary containing DSR flexibility data for baseline and electrified heat
        dsr_hours_dict: dict[str,list]
            Hours during which demand-side management can occur
        ev_availability_profile : pd.DataFrame
            DataFrame containing EV availability profile indexed by time and bus
        load_bus_suffixes: dict[str, str]
            Suffixes to append to load buses for each flexibility type
        flex_carrier_suffixes : dict[str, str]
            Suffixes to append to flexibility carriers for each flexibility type

    """
    # Iterate through each demand key in the DSR dictionary
    for file, path in dsr.items():
        dsr_type = re.match("regional_(.*)_dsr_inc_eur", file).groups()[0]
        df_dsr = _load_regional_data(path, year)

        dsr_hours = dsr_hours_dict[f"{dsr_type}_dsr"]

        # Calculate DSR duration in hours
        if dsr_hours[0] < dsr_hours[1]:
            # e.g., dsr_hours = [17,20] -> indicates 5pm of day 1 to 8 pm of day 1
            dsr_duration = dsr_hours[1] - dsr_hours[0] + 1
        else:
            # e.g., dsr_hours = [8,6] -> indicates 8am of day 1 to 6am of day 2
            dsr_duration = dsr_hours[1] + (24 - dsr_hours[0]) + 1
        logger.info(f"DSR duration for {dsr_type} is {dsr_duration} hours per day")

        # Create DSR profile, storage capacity, e_min_pu and e_max_pu based on demand type
        dsr_profile = _get_dsr_profile(n, df_dsr, dsr_hours, dsr_type)
        storage_capacity = df_dsr.p_nom.abs() * (dsr_duration)
        e_max_pu = dsr_profile.loc[:, df_dsr.index]
        p_max_pu = (
            1.0 if dsr_type != "ev" else ev_availability_profile.loc[:, df_dsr.index]
        )

        _add_dsr_pypsa_components(
            n,
            df_dsr,
            dsr_type,
            storage_capacity,
            e_max_pu,
            p_max_pu,
            load_bus_suffixes,
            flex_carrier_suffixes,
        )


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


def attach_dc_interconnectors(
    n: pypsa.Network,
    interconnectors_path: str,
    year: int,
    interconnectors_availability_path: str,
) -> None:
    """
    Attach DC interconnector links to the network with fixed capacities.

    Removes existing DC links between GB and neighboring countries, then adds
    new links based on the interconnectors CSV file for the specified year.

    Args:
        n (pypsa.Network): The PyPSA network
        interconnectors_path (str): Path to interconnectors CSV file
        year (int): Year for which to load interconnector capacities
        interconnectors_availability_path (str): Path to interconnectors availability CSV file
    """
    # Load interconnector data
    interconnectors = pd.read_csv(
        interconnectors_path,
        dtype={"bus0": str, "bus1": str},
        index_col=["project", "year"],
    )
    interconnectors_this_year = interconnectors.xs(year, level="year")

    if interconnectors_this_year.empty:
        logger.warning(f"No interconnector data found for year {year}")
        return

    # Find DC links connecting GB to neighbors (bidirectional)
    existing_dc_links = n.links[
        (n.links.carrier == "DC")
        & (
            (n.links.bus0.str.startswith("GB") & ~n.links.bus1.str.startswith("GB"))
            | (n.links.bus1.str.startswith("GB") & ~n.links.bus0.str.startswith("GB"))
        )
    ]

    # Remove existing DC interconnectors
    if not existing_dc_links.empty:
        removed_links = existing_dc_links.index.tolist()
        n.remove("Link", removed_links)
        logger.info(
            f"Removed {len(removed_links)} existing DC interconnector links: {removed_links}"
        )
    else:
        logger.info("No existing DC interconnector links found to remove")
    availability_df = pd.read_csv(
        interconnectors_availability_path, index_col=0
    ).squeeze()
    hourly_availability_df = _monthly_to_hourly_profile(availability_df, n.snapshots)
    p_max_pu = pd.concat(
        [hourly_availability_df.rename(i) for i in interconnectors_this_year.index],
        axis=1,
    )
    # Add the links
    # Replace the default `p_min_pu` (-1, used to enforce bidirectionality) with the negative of `p_max_pu`.
    n.add(
        "Link",
        interconnectors_this_year.index,
        **interconnectors_this_year.drop(columns="p_min_pu"),
        p_max_pu=p_max_pu,
        p_min_pu=-1 * p_max_pu,
    )

    logger.info(
        f"Added {len(interconnectors_this_year)} DC interconnector links with total capacity "
        f"{interconnectors_this_year['p_nom'].sum():.1f} MW for year {year}"
    )


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

            capital_cost = costs.at[car, "capital_cost"]

            buses = ds.indexes["bus_bin"].get_level_values("bus")
            bus_bins = ds.indexes["bus_bin"].map(flatten)

            p_max_pu = ds["profile"].to_pandas()
            p_max_pu.columns = p_max_pu.columns.map(flatten)

            if not ppl.query("carrier == @car").empty:
                caps = ppl.query("carrier == @car").groupby("bus").p_nom.sum()
                caps = caps.reindex(buses).fillna(0)
                caps = pd.Series(data=caps.values, index=bus_bins)
            else:
                caps = pd.Series(index=bus_bins).fillna(0)

            n.add(
                "Generator",
                bus_bins,
                suffix=" " + car,
                bus=buses,
                carrier=car,
                p_nom=caps,
                p_nom_min=0,
                p_nom_extendable=car in extendable_carriers["Generator"],
                p_nom_max=np.inf,
                marginal_cost=costs.at[car, "marginal_cost"],
                capital_cost=capital_cost,
                efficiency=costs.at[car, "efficiency"],
                p_max_pu=p_max_pu,
                lifetime=costs.at[car, "lifetime"],
            )


def add_battery_storage(
    n: pypsa.Network,
    ppl: pd.DataFrame,
    battery_e_nom_path: str,
    year: int,
) -> None:
    """
    Add battery storage to the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to attach the battery storage to.
    battery_e_nom_path : str
        Path to the battery storage capacity CSV file.
    ppl : pd.DataFrame
        DataFrame containing the power plant data.
    year : int
        Year for which to load battery storage capacities.
    """
    battery_e_nom = pd.read_csv(battery_e_nom_path, index_col="year").xs(
        year,
    )
    ppl_battery = ppl[ppl.carrier == "battery"]
    all_data_battery = (
        ppl_battery.reset_index().merge(battery_e_nom, on="bus").set_index("name")
    )
    if all_data_battery.empty:
        logger.info(f"No battery storage data found for year {year}")
        return

    max_hours = all_data_battery.e_nom / all_data_battery.p_nom
    n.add("Carrier", "Battery Storage")
    n.add(
        "StorageUnit",
        all_data_battery.index,
        bus=all_data_battery.bus,
        carrier="Battery Storage",
        p_nom=all_data_battery.p_nom,
        p_nom_extendable=False,
        marginal_cost=all_data_battery.marginal_cost,
        capital_cost=0,
        lifetime=all_data_battery.lifetime,
        efficiency_store=all_data_battery.efficiency**0.5,
        efficiency_dispatch=all_data_battery.efficiency**0.5,
        max_hours=max_hours,
        cyclic_state_of_charge=True,
    )


def add_H2(
    n: pypsa.Network,
    ppl: pd.DataFrame,
    year: int,
    regional_H2_demand_annual_inc_eur: str,
    regional_non_networked_electrolysis_demand_annual_inc_eur: str,
    regional_H2_storage_capacity_inc_eur_inc_tech_data: str,
    regional_grid_electrolysis_capacities_inc_eur_inc_tech_data: str,
) -> None:
    demand = _load_regional_data(regional_H2_demand_annual_inc_eur, year)
    demand_fixed = _load_regional_data(
        regional_non_networked_electrolysis_demand_annual_inc_eur, year
    )
    storage_caps = _load_powerplants(
        regional_H2_storage_capacity_inc_eur_inc_tech_data, year
    )
    electrolysis_caps = _load_powerplants(
        regional_grid_electrolysis_capacities_inc_eur_inc_tech_data, year
    )

    n.add("Carrier", "H2")
    all_nodes = n.buses[n.buses.carrier == "AC"].index
    n.add(
        "Bus",
        all_nodes,
        suffix=" H2",
        carrier="H2",
        unit="MWh_LHV",
        country=n.buses.loc[all_nodes].country,
    )

    n.add(
        "Load",
        demand.index,
        suffix=" H2 demand",
        bus=demand.index + " H2",
        carrier="H2",
        p_set=demand.p_set / n.snapshot_weightings.objective.sum(),
    )
    n.add(
        "Load",
        demand_fixed.index,
        suffix=" Electricity for off-grid electrolysis",
        bus=demand_fixed.index,
        p_set=demand_fixed.p_set / n.snapshot_weightings.objective.sum(),
    )

    n.add("Carrier", "H2 Electrolysis")
    n.add(
        "Link",
        electrolysis_caps.index,
        bus1=electrolysis_caps.bus + " H2",
        bus0=electrolysis_caps.bus,
        p_nom=electrolysis_caps.p_nom,
        carrier="H2 Electrolysis",
        efficiency=electrolysis_caps.efficiency,
        marginal_cost=electrolysis_caps.marginal_cost,
        capital_cost=0,
        lifetime=electrolysis_caps.lifetime,
    )
    if not (fuel_cells := ppl[ppl.carrier == "hydrogen fuel cell"]).empty:
        n.add("Carrier", "H2 fuel cell")
        n.add(
            "Link",
            fuel_cells.index,
            bus0=fuel_cells.bus + " H2",
            bus1=fuel_cells.bus,
            p_nom=fuel_cells.p_nom,
            carrier="H2 Fuel Cell",
            marginal_cost=fuel_cells.marginal_cost,
            efficiency=fuel_cells.efficiency,
            # NB: fixed cost is per MWel
            capital_cost=0,
            lifetime=fuel_cells.lifetime,
        )
    if not (turbines := ppl[ppl.carrier == "hydrogen turbine"]).empty:
        n.add("Carrier", "H2 Turbine")
        n.add(
            "Link",
            turbines.index,
            bus0=turbines.bus + " H2",
            bus1=turbines.bus,
            carrier="H2 Turbine",
            p_nom=turbines.p_nom,
            marginal_cost=turbines.marginal_cost,
            efficiency=turbines.efficiency,
            capital_cost=0,
            lifetime=turbines.lifetime,
        )
    n.add("Carrier", "H2 Store")
    n.add(
        "Store",
        storage_caps.index,
        bus=storage_caps.bus + " H2",
        e_nom=storage_caps.e_nom,
        e_cyclic=True,
        carrier="H2 Store",
        marginal_cost=storage_caps.marginal_cost,
        capital_cost=0,
        lifetime=storage_caps.lifetime,
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


def _monthly_to_hourly_profile(
    monthly_profile: pd.Series,
    hourly_index: pd.DatetimeIndex,
) -> pd.Series:
    """
    Convert a monthly profile to an hourly profile by repeating each month's value for its number of hours.

    Args:
        hourly_index (pd.DatetimeIndex): DatetimeIndex with hourly timestamps for the entire year.
        monthly_profile (pd.Series): Series with 12 values representing monthly data.

    Returns:
        pd.Series: Series with hourly data for the entire year.
    """
    hourly_profile = pd.Series(
        monthly_profile.reindex(hourly_index.month).values, index=hourly_index
    )
    return hourly_profile


def _add_generator_availability(
    n: pypsa.Network,
    generator_availability_path: str,
) -> None:
    """
    Add generator availability profiles to the network.

    Args:
        n (pypsa.Network): The PyPSA network to modify.
        generator_availability_path (str): Path to generator availability CSV file.
    """
    carrier_availability = pd.read_csv(
        generator_availability_path, index_col=["carrier", "month"]
    ).squeeze()
    gen_availability = (
        carrier_availability.reset_index()
        .merge(n.generators.reset_index(), on="carrier")
        .pivot(index="month", columns="name", values="availability_fraction")
    )
    p_max_pu = gen_availability.apply(
        _monthly_to_hourly_profile, hourly_index=n.snapshots
    )
    _add_timeseries_data_to_network(
        n.generators_t, p_max_pu, "p_max_pu", conflicts="mul"
    )


def add_load_shedding(n: pypsa.Network, voll: float) -> None:
    """
    Add a load shedding generator to the network.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to attach the load shedding to.
    voll : float
        Value of lost load (VOLL) in EUR/MWh. If zero, load shedding is not added.
    """
    if voll > 0:
        n.add("Carrier", "Load Shedding")
        buses = n.buses.query("carrier=='AC'").index
        n.add(
            "Generator",
            buses,
            suffix=" Load Shedding",
            marginal_cost=voll,
            carrier="Load Shedding",
            bus=buses,
            p_nom=np.inf,
        )
        logger.info("Added load shedding generators to all AC buses")
    else:
        logger.info("VOLL is zero or False; load shedding not added to the network")


def aggregate_time(n: pypsa.Network, time_aggregation: list[dict]) -> pypsa.Network:
    """
    Aggregate time steps in the network using the specified time aggregation method.

    Parameters
    ----------
    n : pypsa.Network
        The PyPSA network to aggregate.
    time_aggregation : list of dict
        List of dictionaries specifying the time aggregation method and parameters.

    Returns
    -------
    pypsa.Network
        The aggregated PyPSA network.
    """
    if not time_aggregation:
        logger.info("No time aggregation specified; returning original network")
        return n
    m = n.copy()
    for agg in time_aggregation:
        logger.info(f"Aggregating time steps using method: {agg['method']}")
        m = getattr(m.cluster.temporal, agg["method"])(**agg["parameters"])
    logger.info(
        f"Time aggregation complete. Original snapshots: {len(n.snapshots)}, Aggregated snapshots: {len(m.snapshots)}"
    )
    return m


def compose_network(
    network_path: str,
    output_path: str,
    costs_path: str,
    powerplants_path: str,
    hydro_capacities_path: str | None,
    chp_p_min_pu_path: str,
    interconnectors_path: str,
    interconnectors_availability_path: str,
    generator_availability_path: str,
    battery_e_nom_path: str,
    renewable_profiles: dict[str, str],
    countries: list[str],
    costs_config: dict[str, Any],
    voll: float,
    electricity_config: dict[str, Any],
    renewable_config: dict[str, Any],
    demands: dict[str, str],
    ev_data: dict[str, str],
    dsr: dict[str, str],
    H2_data: dict[str, Any],
    enable_chp: bool,
    dsr_hours_dict: dict[str, list],
    load_bus_suffixes: dict[str, str],
    flex_carrier_suffixes: dict[str, str],
    time_aggregation: list[dict],
    year: int,
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
    chp_p_min_pu_path : str
        Path to CHP minimum operation profile CSV file
    interconnectors_path : str
        Path to interconnectors CSV file with DC link capacities
    renewable_profiles : dict
        Mapping of carrier names to profile file paths
    heat_demand_path : str
        Path to hourly heat demand NetCDF file for CHP constraints
    countries : list[str]
        List of country codes to include
    costs_config : dict
        Costs configuration dictionary
    voll : float
        Value of lost load in £/MWh
    electricity_config : dict
        Electricity configuration dictionary
    clustering_config : dict
        Clustering configuration dictionary
    renewable_config : dict
        Renewable configuration dictionary
    demands: dict[str, str]
        Dictionary mapping demand types to paths for the demand data
    ev_data : dict[str, str]
        Dictionary containing EV flexibility data
    dsr : dict[str, str]
        Dictionary containing DSR flexibility data for baseline and electrified heat
    dsr_hours_dict: dict[str, list]
        DSR hours for each demand type
    load_bus_suffixes : dict[str, str]
        Suffixes to append to load buses for each demand type
    flex_carrier_suffixes : dict[str, str]
        Suffixes to append to flexibility carriers for each demand type
    enable_chp : bool
        Whether to enable CHP constraints
    year: int
        Modelling year
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
        renewable_carriers=set(electricity_config["renewable_carriers"]),
        conventional_params={},
        conventional_inputs={},
        unit_commitment=None,
        fuel_price=None,
    )

    # Add simplified CHP constraints if enabled
    if enable_chp:
        logger.info("Adding simplified CHP constraints based on heat demand.")
        chp_p_min_pu = pd.read_csv(
            chp_p_min_pu_path, index_col="snapshot", parse_dates=True
        )
        attach_chp_constraints(network, chp_p_min_pu)

    add_load(network, demands, load_bus_suffixes)

    ev_availability_profile = pd.read_csv(
        ev_data["avail_profile_s_clustered"], index_col=0, parse_dates=True
    )

    add_DSR(
        network,
        year,
        dsr,
        dsr_hours_dict,
        ev_availability_profile,
        load_bus_suffixes,
        flex_carrier_suffixes,
    )

    add_EV_V2G(
        network,
        year,
        ev_data["regional_ev_v2g_inc_eur"],
        ev_data["regional_ev_v2g_storage_inc_eur"],
        ev_availability_profile,
    )

    add_H2(network, ppl, year, **H2_data)

    attach_dc_interconnectors(
        network, interconnectors_path, year, interconnectors_availability_path
    )

    add_battery_storage(network, ppl, battery_e_nom_path, year)
    _add_generator_availability(network, generator_availability_path)
    add_load_shedding(network, voll)

    finalise_composed_network(network, context)

    network_agg = aggregate_time(network, time_aggregation)
    network_agg.export_to_netcdf(output_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("compose_network", clusters="clustered", year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    log_suffix = "-" + "_".join(snakemake.wildcards) if snakemake.wildcards else ""
    logger = logging.getLogger(Path(__file__).stem + log_suffix)

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
        interconnectors_path=snakemake.input.interconnectors_p_nom,
        interconnectors_availability_path=snakemake.input.interconnectors_availability,
        generator_availability_path=snakemake.input.generator_availability,
        battery_e_nom_path=snakemake.input.battery_e_nom,
        countries=snakemake.params.countries,
        costs_config=snakemake.params.costs_config,
        voll=snakemake.params.voll,
        electricity_config=snakemake.params.electricity,
        renewable_config=snakemake.params.renewable,
        demands=_input_list_to_dict(snakemake.input.demands, parent=True),
        ev_data=_input_list_to_dict(snakemake.input.ev_data),
        dsr=_input_list_to_dict(snakemake.input.dsr),
        H2_data=_input_list_to_dict(snakemake.input.H2_data),
        enable_chp=snakemake.params.enable_chp,
        dsr_hours_dict=snakemake.params.dsr_hours_dict,
        load_bus_suffixes=snakemake.params.load_bus_suffixes,
        flex_carrier_suffixes=snakemake.params.flex_carrier_suffixes,
        time_aggregation=snakemake.params.time_aggregation,
        year=int(snakemake.wildcards.year),
    )

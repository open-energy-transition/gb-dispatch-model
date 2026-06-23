# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
EV demand shape data processor.

This script processes EV demand shape data from PyPSA-Eur.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import (
    configure_logging,
    generate_periodic_profiles,
    get_snapshots,
    set_scenario_config,
)
from scripts.prepare_sector_network import cycling_shift

logger = logging.getLogger(__name__)


def parse_ev_demand_shape(
    traffic_rate_profile: pd.DataFrame,
    plug_in_offset: int,
    charging_duration: int,
) -> pd.DataFrame:
    """
    Parse EV demand shape from traffic rate profiles by applying plug-in timing and charging duration.

    Args:
        traffic_rate_profile (pd.DataFrame): DataFrame containing normalized traffic rate profiles
                                           indexed by time with columns for each node/bus
        plug_in_offset (int): Time offset in hours between traffic activity and EV plug-in events
        charging_duration (int): Duration of EV charging session in hours

    Returns:
        pd.DataFrame: DataFrame containing unmanaged EV charging demand shape indexed by time
                     with columns for each bus. Values represent normalized charging load profiles.

    Processing steps:
        1. Apply time offset to traffic profiles to create plug-in rate profiles
        2. Compute unmanaged EV demand shape using charging duration convolution
    """

    # Define plug in rate
    plug_in_rate = cycling_shift(traffic_rate_profile, steps=plug_in_offset)

    # Compute unmanaged ev demand shape
    unmanaged_ev_demand_shape = compute_ev_demand_shape(plug_in_rate, charging_duration)

    # Rename column name as bus
    unmanaged_ev_demand_shape.columns.name = "bus"

    return unmanaged_ev_demand_shape


def build_traffic_rate_profile(
    traffic_path: str, nodes: list, snapshots: pd.DatetimeIndex
) -> pd.DataFrame:
    """
    Build traffic rate profile for EV demand modeling using weekly traffic count data.

    Args:
        traffic_path (str): Filepath to the traffic count CSV file containing
                           weekly traffic counts averaged over multiple years
        nodes (list): List of PyPSA bus nodes for which to generate traffic profiles
        snapshots (pd.DatetimeIndex): DatetimeIndex representing the temporal
                                     resolution of the model (hourly snapshots)

    Returns:
        pd.DataFrame: DataFrame containing normalized traffic rate profiles indexed
                     by snapshots with columns for each node. Values represent
                     the relative traffic intensity at each time period.

    Processing steps:
        1. Load weekly traffic count data from CSV file
        2. Generate periodic profiles for all nodes using weekly pattern
        3. Normalize profiles so total sum equals 1
    """

    # averaged weekly counts from the year 2010-2015
    traffic = pd.read_csv(traffic_path, skiprows=2, usecols=["count"]).squeeze(
        "columns"
    )

    # create annual profile take account time zone + summer time
    transport_shape = generate_periodic_profiles(
        dt_index=snapshots,
        nodes=nodes,
        weekly_profile=traffic.values,
    )
    transport_shape = transport_shape / transport_shape.sum()

    return transport_shape


def compute_ev_demand_shape(df: pd.DataFrame, charging_duration: int) -> pd.DataFrame:
    """
    Compute EV charging demand shape by convolving plug-in rates with charging duration.

    Args:
        df (pd.DataFrame): DataFrame containing plug-in rate profiles indexed by time
                          with columns for each node/bus
        charging_duration (int): Duration of EV charging session in hours

    Returns:
        pd.DataFrame: DataFrame with unmanaged EV charging demand profiles indexed
                     by time with same column structure as input. Values represent
                     the charging load at each time period.

    Processing steps:
        1. Create temporary data at year start to handle wrap-around effects
        2. Apply rolling sum over charging duration to simulate continuous charging
        3. Normalize the resulting profiles so total sum equals 1
    """
    # To wrap the year, we have to mock data at the start of the timeseries and then strip it out later.
    temp_df = df.iloc[-charging_duration:]
    temp_df.index = temp_df.index - pd.DateOffset(years=1)
    rolling_sum_df = (
        pd.concat([temp_df, df])
        .rolling(charging_duration)
        .sum()
        .iloc[charging_duration:]
    )

    # Normalize the convoluted profiles
    rolling_sum_df = rolling_sum_df / rolling_sum_df.sum()

    return rolling_sum_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, clusters="clustered")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the input paths
    traffic_data_path = snakemake.input.traffic_data_KFZ

    # Load the parameters
    plug_in_offset = snakemake.params.plug_in_offset
    charging_duration = snakemake.params.charging_duration

    # Define nodes and snapshots
    pop_layout = pd.read_csv(snakemake.input.clustered_pop_layout, index_col=0)
    nodes = pop_layout.index
    snapshots = get_snapshots(
        snakemake.params.snapshots, snakemake.params.drop_leap_day, tz="UTC"
    )

    # Build traffic rate profile
    traffic_rate_profile = build_traffic_rate_profile(
        traffic_data_path,
        nodes=nodes,
        snapshots=snapshots,
    )

    # Parse unmanaged ev charging demand shape
    ev_charging_demand_shape = parse_ev_demand_shape(
        traffic_rate_profile,
        plug_in_offset,
        charging_duration,
    )

    # Write EV demand shape dataframe to csv file
    ev_charging_demand_shape.to_csv(snakemake.output.demand_shape)
    logger.info(f"EV demand shape data saved to {snakemake.output.demand_shape}")

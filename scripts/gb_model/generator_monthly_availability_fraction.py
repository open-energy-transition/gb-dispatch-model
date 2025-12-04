# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Monthly asset unavailability profile generator.

This is a script to calculate monthly unavailability for generation assets for which we have outage data in Great Britain.
We do not attempt to calculate unavailability curves regionally as regional data on maximum capacities per carrier are not robust enough.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def prep_outage_data(
    input_path: str,
    max_unavailable_days: int,
    resource_type_mapping: dict,
    carrier_mapping: dict,
) -> pd.DataFrame:
    """
    Prepare outage data for analysis by renaming columns, filtering by duration, and mapping to carriers.

    Args:
        input_path (str): Path to the input CSV file containing outage data.
        max_unavailable_days (int): Maximum number of days a generator can be unavailable to be included.
        resource_type_mapping (dict): Mapping for resource types to Fueltype and Technology.
        carrier_mapping (dict): Mapping for Fueltype and Technology to carrier names.

    Returns:
        pd.DataFrame: A DataFrame containing the prepared outage data.
    """
    outages = pd.read_csv(input_path, parse_dates=["start_time", "end_time"])
    outages = outages[
        (outages.end_time - outages.start_time).dt.days <= max_unavailable_days
    ]

    for mapping in ["Fueltype", "Technology"]:
        outages[mapping] = outages["resource_type"].map(resource_type_mapping[mapping])
    outages["carrier"] = (
        outages["Fueltype"].fillna(outages["Technology"]).map(carrier_mapping)
    )

    outages = outages.dropna(subset=["carrier"])
    if outages.empty:
        logger.warning(f"No outages found in {input_path} after filtering")
        return outages

    return outages


def _start_end_to_month_series(series: pd.Series) -> pd.Series:
    """
    Convert start and end time to a series of months.

    Args:
        series (pd.Series): A Series containing start and end times.

    Returns:
        pd.Series: A Series indexed by month with the maximum unavailable MW for each month.
    """
    full_date_range = pd.date_range(
        start=f"{snakemake.params.start_date} 00:00",
        end=f"{snakemake.params.end_date} 23:59",
        freq="min",
        tz=series["start_time"].tz,
    )
    new_df = pd.Series(0, index=full_date_range, dtype=float)
    new_df.loc[
        slice(series["start_time"].round("min"), series["end_time"].round("min"))
    ] = series["max_unavailable_mw"]
    monthly_df = new_df.groupby(new_df.index.month).mean()
    return monthly_df


def monthly_outages(df: pd.DataFrame, carrier_max_cap: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly outages as a fraction of total national capacity for a given carrier.

    Args:
        df (pd.DataFrame): DataFrame containing outage data.
        carrier_max_cap (pd.Series): Series containing maximum capacity for each carrier.

    Returns:
        pd.DataFrame: DataFrame containing monthly outages as a fraction of total capacity.
    """
    monthly_outages = df.apply(_start_end_to_month_series, axis=1)
    total_cap = carrier_max_cap.loc[df.name]
    availability_fraction = (total_cap - monthly_outages.sum()) / total_cap
    logger.info(
        f"Calculated monthly {df.name} availability from {len(df)} outage events as a fraction of total capacity. "
        f"Average annual unavailability: {availability_fraction.mean()}"
    )
    return availability_fraction


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    powerplants = pd.read_csv(snakemake.input.powerplants)
    all_outages = pd.concat(
        [
            prep_outage_data(
                snakemake.input[i],
                snakemake.params.max_unavailable_days,
                snakemake.params.resource_type_mapping,
                snakemake.params.carrier_mapping,
            )
            for i in ["planned", "forced"]
        ]
    )
    logger.info(f"Loaded {len(all_outages)} outage records")

    powerplants_carriers = (
        powerplants["Fueltype"]
        .map(snakemake.params.carrier_mapping)
        .fillna(powerplants["Technology"].map(snakemake.params.carrier_mapping))
    )
    carrier_max_cap = (
        powerplants[powerplants.Country == "GB"]
        .groupby(powerplants_carriers)
        .Capacity.sum()
    )

    grouped_outages = all_outages.groupby("carrier").apply(
        monthly_outages,
        include_groups=False,
        carrier_max_cap=carrier_max_cap,
    )

    grouped_outages_tdf = (
        grouped_outages.rename_axis(columns="month")
        .stack()
        .rename("availability_fraction")
        .reset_index()
    )
    grouped_outages_tdf.to_csv(snakemake.output.csv, index=False)

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
from scripts.gb_model.generators import create_powerplants_table

logger = logging.getLogger(__name__)


def prep_outage_data(
    input_path: str,
    max_unavailable_days: int,
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
    outages = pd.read_csv(input_path, dtype={"nominal_power": int, "avail_qty": float})
    for dt_col in ["start", "end"]:
        outages[dt_col] = pd.to_datetime(outages[dt_col], utc=True)

    outages = outages[(outages.end - outages.start).dt.days <= max_unavailable_days]

    outages["carrier"] = outages["plant_type"].map(carrier_mapping)

    if (isna := outages["carrier"].isnull()).any():
        missing_carriers = outages[isna]["plant_type"].unique()
        logger.warning(
            f"Some plant types in {input_path} could not be mapped to carriers: {missing_carriers}"
        )
    outages = outages.dropna(subset=["carrier"])

    if outages.empty:
        logger.warning(f"No outages found in {input_path} after filtering")

    outages["max_unavailable_mw"] = outages["nominal_power"] - outages["avail_qty"]

    return outages


def _add_outages_to_ts(series: pd.Series, sum_df: pd.Series) -> None:
    """
    Add outage data to a time series based on start and end times.

    Args:
        series (pd.Series): A Series containing start and end times.
        sum_df (pd.Series): An 1-minute resolution timeseries to accumulate the maximum unavailable MW.
    """

    sum_df.loc[slice(series["start"].round("min"), series["end"].round("min"))] += (
        series["max_unavailable_mw"]
    )


def monthly_outages(df: pd.DataFrame, carrier_max_cap: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly outages as a fraction of total national capacity for a given carrier.

    Args:
        df (pd.DataFrame): DataFrame containing outage data.
        carrier_max_cap (pd.Series): Series containing maximum capacity for each carrier.

    Returns:
        pd.DataFrame: DataFrame containing monthly outages as a fraction of total capacity.
    """
    full_date_range = pd.date_range(
        start=f"{snakemake.params.start_date} 00:00",
        end=f"{snakemake.params.end_date} 23:59",
        freq="min",
        tz=df["start"].dt.tz,
    )
    sum_df = pd.Series(0, index=full_date_range, dtype=float)
    df.apply(_add_outages_to_ts, axis=1, sum_df=sum_df)
    total_cap = carrier_max_cap.loc[df.name]
    availability_fraction = (
        total_cap - sum_df.groupby(sum_df.index.month).mean()
    ) / total_cap
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

    all_outages = prep_outage_data(
        snakemake.input.outages,
        snakemake.params.max_unavailable_days,
        snakemake.params.entsoe_carrier_mapping,
    )
    logger.info(f"Loaded {len(all_outages)} outage records")

    df_dukes = pd.read_csv(snakemake.input.dukes_data)
    dukes_config = snakemake.params.dukes_config
    default_set = snakemake.params.default_set
    df_capacity_gb_dukes = create_powerplants_table.capacity_table(
        df_dukes, dukes_config, default_set
    )
    carrier_max_cap = df_capacity_gb_dukes.groupby("carrier")["p_nom"].sum()
    missing_carriers = set(all_outages["carrier"]) - set(carrier_max_cap.index)
    if missing_carriers:
        logger.warning(
            f"The following carriers have outage data but no capacity data and will be skipped: {missing_carriers}"
        )
    all_outages_filtered = all_outages[
        all_outages["carrier"].isin(carrier_max_cap.index)
    ]
    grouped_outages = all_outages_filtered.groupby("carrier").apply(
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

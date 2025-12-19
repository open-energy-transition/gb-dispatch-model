# SPDX-FileCopyrightText:  gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT


"""
Low carbon contract strike price processor
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def process_strike_prices(
    df: pd.DataFrame, carrier_mapping: dict, end_year: int
) -> pd.DataFrame:
    """
    Process low carbon contract strike prices to obtain strike prices per technology and year.

    Args:
        df (pd.DataFrame): Low carbon contracts register data
        carrier_mapping (dict): Mapping of technology names to PyPSA carriers
        end_year (int): Final year to process
    Returns:
        pd.DataFrame: DataFrame containing average strike prices indexed by carrier
    """
    df_filtered = df[
        (df.Settlement_Date.dt.year < end_year)
        & (df.Technology.isin(carrier_mapping.keys()))
    ]
    df_strike_price = (
        df_filtered.replace({"Technology": carrier_mapping})
        .groupby("Technology")["Strike_Price_GBP_Per_MWh"]
        .mean()
        .to_frame("strike_price_GBP_per_MWh")
        .rename_axis(index="carrier")
    )
    logger.info("Processed average strike prices:\n%s", df_strike_price)
    return df_strike_price


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    df = pd.read_csv(snakemake.input.register, parse_dates=["Settlement_Date"])
    df_strike_prices = process_strike_prices(
        df, snakemake.params.carrier_mapping, end_year=int(snakemake.params.end_year)
    )
    df_strike_prices.to_csv(snakemake.output.csv)

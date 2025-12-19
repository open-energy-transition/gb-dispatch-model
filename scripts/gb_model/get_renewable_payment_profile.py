# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Compute renewable payment profile as the difference between market rate and strike prices.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def compute_payment_profile(
    result: pypsa.Network, strike_prices: dict[str, float]
) -> pd.DataFrame:
    """
    Compute difference in electricity market price and strike price for renewable generators

    Parameters
    ----------
    result: pypsa.Network
        Unconstrained optimisation result
    strike_prices: dict[str, float]
        Dictionary of strike prices for each renewable generator
    """

    generator_columns = result.generators.query(
        "carrier in @strike_prices.keys()"
    ).index
    price_profile = pd.DataFrame(
        index=result.snapshots, columns=generator_columns, data=0.0
    )
    for car in strike_prices.keys():
        carrier_generators = result.generators.query("carrier in @car")
        if carrier_generators.empty:
            logger.info(
                "No generators found to compute renewable payment profile for the carrier {car}"
            )
            continue
        strike_price = strike_prices[car]
        buses = carrier_generators.bus.tolist()
        marginal_cost_columns = carrier_generators.index.tolist()
        price_difference_profile = strike_price - result.buses_t.marginal_price[buses]
        price_difference_profile.columns = marginal_cost_columns
        price_profile[marginal_cost_columns] = price_difference_profile

    return price_profile


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load input networks and parameters
    unconstrained_result = pypsa.Network(snakemake.input.unconstrained_result)
    strike_prices = pd.read_csv(
        snakemake.input.strike_prices, index_col="carrier"
    ).to_dict()["strike_price_GBP_per_MWh"]

    renewable_price_profile = compute_payment_profile(
        unconstrained_result, strike_prices
    )

    renewable_price_profile.to_csv(snakemake.output.csv)
    logger.info(f"Exported renewable price profile to {snakemake.output.csv}")

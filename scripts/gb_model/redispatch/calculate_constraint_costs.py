# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Calculate constraint costs across all redispatch years
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

LOAD_SHEDDING_REGEX = "Load Shedding"


def constraint_cost(networks: list[pypsa.Network], extra_years: int) -> float:
    """
    Calculate constraint costs across all redispatch years

    Parameters
    ----------
    networks: list[pypsa.Network]
        List of solved networks for each redispatch year
    extra_years: int
        Number of extra years included in the redispatch model

    Returns
    -------
    float
        Total constraint costs across all redispatch years
    """
    year_constraint_costs = {}
    for network in networks:
        redispatch_carriers = network.carriers.filter(
            regex=r" ramp (up|down)$", axis=0
        ).index.tolist() + ["Load Shedding"]
        constraint_costs = network.statistics.opex(
            comps="Generator", aggregate_time="sum", carrier=redispatch_carriers
        )

        year = network.meta["wildcards"]["year"]
        logger.info(
            constraint_costs.to_frame(f"Constraint costs for redispatch year {year}:")
            .style.format("{:,.0f} GBP")
            .to_string()
            + f"\nTotal: {constraint_costs.sum():,.0f} GBP"
        )
        year_constraint_costs[year] = constraint_costs.sum()

    final_year = max(int(i) for i in year_constraint_costs.keys())
    total_constraint_cost = (
        sum(year_constraint_costs.values())
        + year_constraint_costs[str(final_year)] * extra_years
    )
    logger.info(
        "Total constraint costs across all redispatch years "
        f"(including {extra_years} extra years at year {final_year}): {total_constraint_cost:,.0f} GBP"
    )
    return total_constraint_cost


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load input networks and parameters
    networks = [pypsa.Network(f) for f in snakemake.input.networks]
    extra_years = snakemake.params.constraint_cost_extra_years

    total_constraint_cost = constraint_cost(networks, extra_years)

    pd.Series({"constraint_cost": total_constraint_cost}).to_csv(
        snakemake.output.csv, header=False
    )

# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Prepare network for constrained optimization.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def fix_dispatch(constrained_network, unconstrained_result):
    """
    Fix dispatch of generators and storage units based on the result of unconstrained optimization

    Parameters
    ----------
    constrained_network: pypsa.Network
        Constrained network to finalize
    unconstrained_result: pypsa.Network
        Result of the unconstrained optimization
    """

    for comp in unconstrained_result.components:
        if comp.name not in ["Generator", "StorageUnit"]:
            continue
        p_fix = comp.dynamic.p / comp.static.p_nom

        # Filter only GB plants
        p_fix = p_fix.filter(like="GB")

        constrained_network.components[comp.name].dynamic.p_max_pu = p_fix
        constrained_network.components[comp.name].dynamic.p_min_pu = p_fix

    logger.info("Fixed the dispatch of generators")


def _apply_multiplier(
    df: pd.DataFrame,
    multiplier: dict[str, float],
    renewable_payment_profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply bid/offer multiplier and strike prices

    Parameters
    ----------
    df: pd.DataFrame
        Generator dataframe
    multiplier: dict[str, float]
        Mapping of conventional carrier to multiplier
    renewable_payment_profile: pd.DataFrame
        Renewable payment profile for each renewable generator
    """
    df = df.assign(multiplier=df["carrier"].map(multiplier)).fillna(1)

    df["marginal_cost"] *= df["multiplier"]

    intersecting_columns = renewable_payment_profile.columns.intersection(df.index)
    marginal_cost_profile = renewable_payment_profile[intersecting_columns]
    renewable_carriers = df.loc[intersecting_columns].carrier.unique()

    undefined_multipliers = set(df["carrier"].unique()) - (
        set(multiplier.keys()) | set(renewable_carriers)
    )
    logger.warning(
        f"Neither bid/offer multiplier nor strike price provided for the carriers: {undefined_multipliers}"
    )

    return df, marginal_cost_profile


def create_up_down_plants(
    constrained_network: pypsa.Network,
    unconstrained_result: pypsa.Network,
    bids_and_offers: dict[str, float],
    renewable_payment_profile: pd.DataFrame,
):
    """
    Add generators and storage units components that mimic increase / decrease in dispatch

    Parameters
    ----------
    constrained_network: pypsa.Network
        Constrained network to finalize
    unconstrained_result: pypsa.Network
        Result of the unconstrained optimization
    bids_and_offers: dict[str, float]
        Bid and offer multipliers for conventional carriers
    renewable_payment_profile: pd.DataFrame
        Dataframe of the renewable price profile
    """
    gb_buses = unconstrained_result.buses.query("country == 'GB'").index  # noqa: F841

    for comp in constrained_network.components:
        if comp.name not in ["Generator", "StorageUnit"]:
            continue

        g_up = comp.static.copy()
        g_down = comp.static.copy()

        # Filter GB plants
        g_up = g_up.query("bus in @gb_buses")
        g_down = g_down.query("bus in @gb_buses")

        # Compute dispatch limits for the up and down generators
        result_component = unconstrained_result.components[comp.name]
        up_limit = (
            unconstrained_result.get_switchable_as_dense(comp.name, "p_max_pu")
            * result_component.static.p_nom
            - result_component.dynamic.p
        ).clip(0) / result_component.static.p_nom
        down_limit = -result_component.dynamic.p / result_component.static.p_nom

        # Add bid/offer multipliers for conventional generators
        g_up, marginal_cost_up = _apply_multiplier(
            g_up, bids_and_offers["offer_multiplier"], renewable_payment_profile
        )
        g_down, marginal_cost_down = _apply_multiplier(
            g_down, bids_and_offers["bid_multiplier"], renewable_payment_profile
        )

        # Add generators that can increase dispatch
        constrained_network.add(
            comp.name,
            g_up.index,
            suffix=" ramp up",
            p_max_pu=up_limit.loc[:, g_up.index],
            **g_up.drop("p_max_pu", axis=1),
        )

        # Add generators that can decrease dispatch
        constrained_network.add(
            comp.name,
            g_down.index,
            suffix=" ramp down",
            p_min_pu=down_limit.loc[:, g_down.index],
            p_max_pu=0,
            **g_down.drop(["p_max_pu", "p_min_pu"], axis=1),
        )

        if not marginal_cost_down.empty:
            constrained_network.components[comp.name].dynamic.marginal_cost = pd.concat(
                [
                    marginal_cost_up.add_suffix(" ramp up"),
                    marginal_cost_down.add_suffix(" ramp down"),
                ],
                axis=1,
            )

        logger.info("Added generators that can mimic increase and decrease in dispatch")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load input networks and parameters
    network = pypsa.Network(snakemake.input.network)
    unconstrained_result = pypsa.Network(snakemake.input.unconstrained_result)
    bids_and_offers = snakemake.params.bids_and_offers
    renewable_payment_profile = pd.read_csv(
        snakemake.input.renewable_payment_profile,
        index_col="snapshot",
        parse_dates=True,
    )

    fix_dispatch(network, unconstrained_result)

    create_up_down_plants(
        network, unconstrained_result, bids_and_offers, renewable_payment_profile
    )

    network.export_to_netcdf(snakemake.output.network)
    logger.info(f"Exported network to {snakemake.output.network}")

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


def fix_dispatch(
    constrained_network: pypsa.Network, unconstrained_result: pypsa.Network
):
    """
    Fix dispatch of generators and storage units based on the result of unconstrained optimization

    Parameters
    ----------
    constrained_network: pypsa.Network
        Constrained network to finalize
    unconstrained_result: pypsa.Network
        Result of the unconstrained optimization
    """

    def _process_p_fix(dispatch_t: pd.DataFrame, p_nom: pd.DataFrame):
        p_fix = (dispatch_t / p_nom).round(5).fillna(0)
        p_fix = p_fix.drop(columns=p_fix.filter(like="load").columns)

        return p_fix

    for comp in unconstrained_result.components:
        if comp.name not in ["Generator", "StorageUnit", "Link"]:
            continue

        if comp.name == "Generator":
            p_max_fix = p_min_fix = _process_p_fix(comp.dynamic.p, comp.static.p_nom)
        elif comp.name == "Link":
            # Dispatch of intra gb DC links are not be fixed
            mask = comp.static.bus0.str.startswith(
                "GB"
            ) & comp.static.bus1.str.startswith("GB")
            intra_gb_dc_links = comp.static[mask].query("carrier == 'DC'").index

            p_max_fix = p_min_fix = _process_p_fix(
                comp.dynamic.p0, comp.static.p_nom
            ).drop(intra_gb_dc_links, axis=1)
        elif comp.name == "StorageUnit":
            # For storage units: the decision variables are `p_dispatch` and `p_store`.
            # p = p_dispatch - p_store
            # Refer https://docs.pypsa.org/latest/user-guide/optimization/storage/#storage-units
            p_max_fix = _process_p_fix(comp.dynamic.p_dispatch, comp.static.p_nom)
            p_min_fix = _process_p_fix(-1 * comp.dynamic.p_store, comp.static.p_nom)

        constrained_network.components[comp.name].dynamic.p_max_pu = p_max_fix
        constrained_network.components[comp.name].dynamic.p_min_pu = p_min_fix

        logger.info(f"Fixed the dispatch of {comp.name}")


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
    interconnector_bid_offer_profile: pd.DataFrame,
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
    interconnector_bid_offer_profile: pd.DataFrame
        Interconnectors bid/offer profile for each interconnector
    """
    gb_buses = unconstrained_result.buses.query("country == 'GB'").index  # noqa: F841

    for comp in constrained_network.components:
        if comp.name not in ["Generator", "StorageUnit", "Link"]:
            continue

        g_up = comp.static.copy()
        g_down = comp.static.copy()

        if comp.name != "Link":
            # Filter GB plants
            g_up = g_up.query("bus in @gb_buses and p_nom != 0")
            g_down = g_down.query("bus in @gb_buses and p_nom != 0")

        # Compute dispatch limits for the up and down generators
        result_component = unconstrained_result.components[comp.name]
        dynamic_p = (
            result_component.dynamic.p0
            if comp.name == "Link"
            else result_component.dynamic.p
        )

        up_limit = (
            unconstrained_result.get_switchable_as_dense(comp.name, "p_max_pu")
            * result_component.static.p_nom
            - dynamic_p
        ).clip(0) / result_component.static.p_nom
        if comp.name == "Generator":
            down_limit = -dynamic_p / result_component.static.p_nom
        else:
            down_limit = (
                unconstrained_result.get_switchable_as_dense(comp.name, "p_min_pu")
                * result_component.static.p_nom
                - dynamic_p
            ) / result_component.static.p_nom

        if comp.name != "Link":
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

        if comp.name == "Link":
            interconnector_bid_offer_profile.columns = (
                interconnector_bid_offer_profile.columns.str.replace(
                    "bid", "ramp down"
                ).str.replace("offer", "ramp up")
            )
            constrained_network.components[
                comp.name
            ].dynamic.marginal_cost = interconnector_bid_offer_profile
        else:
            if not marginal_cost_down.empty:
                constrained_network.components[
                    comp.name
                ].dynamic.marginal_cost = pd.concat(
                    [
                        marginal_cost_up.add_suffix(" ramp up"),
                        marginal_cost_down.add_suffix(" ramp down"),
                    ],
                    axis=1,
                )

        logger.info(
            f"Added {comp.name} that can mimic increase and decrease in dispatch"
        )


def read_csv(filepath):
    return pd.read_csv(filepath, index_col="snapshot", parse_dates=True)


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
    renewable_payment_profile = read_csv(snakemake.input.renewable_payment_profile)
    interconnector_bid_offer_profile = read_csv(
        snakemake.input.interconnector_bid_offer
    )

    fix_dispatch(network, unconstrained_result)

    create_up_down_plants(
        network,
        unconstrained_result,
        bids_and_offers,
        renewable_payment_profile,
        interconnector_bid_offer_profile,
    )

    network.export_to_netcdf(snakemake.output.network)
    logger.info(f"Exported network to {snakemake.output.network}")

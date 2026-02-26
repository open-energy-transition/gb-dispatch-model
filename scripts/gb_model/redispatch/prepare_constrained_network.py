# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Prepare network for constrained optimization.
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import filter_interconnectors
from scripts.gb_model.dispatch.prepare_unconstrained_network import (
    copperplate_gb,
)

logger = logging.getLogger(__name__)

LOAD_SHEDDING_REGEX = "Load Shedding"


def fix_dispatch(
    constrained_network: pypsa.Network,
    unconstrained_result: pypsa.Network,
    gb_buses: pd.Index,
):
    """
    Fix dispatch of generators and storage units based on the result of unconstrained optimization

    Parameters
    ----------
    constrained_network: pypsa.Network
        Constrained network to finalize
    unconstrained_result: pypsa.Network
        Result of the unconstrained optimization
    gb_buses: pd.Index
        Index of GB buses
    """

    def _process_p_fix(dispatch_t: pd.DataFrame, p_nom: pd.DataFrame):
        p_fix = (dispatch_t / p_nom).round(5).fillna(0)
        p_fix = p_fix.loc[:, ~p_fix.columns.str.contains(LOAD_SHEDDING_REGEX)]

        return p_fix

    for comp in unconstrained_result.components:
        if comp.name not in ["Generator", "StorageUnit", "Link"]:
            continue

        if comp.name == "Generator":
            p_max_fix = p_min_fix = _process_p_fix(comp.dynamic.p, comp.static.p_nom)
        elif comp.name == "Link":
            # Only dispatch of interconnector links are to be fixed
            interconnector_links = comp.static.query(
                "carrier == 'DC' and bus0 in @gb_buses and bus1 not in @gb_buses"
            ).index

            p_max_fix = p_min_fix = _process_p_fix(comp.dynamic.p0, comp.static.p_nom)[
                interconnector_links
            ]
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
    renewable_strike_prices: pd.Series,
    direction: Literal["bid", "offer"],
) -> pd.Series:
    """
    Apply bid/offer multiplier and strike prices

    Parameters
    ----------
    df: pd.DataFrame
        Generator dataframe
    multiplier: dict[str, float]
        Mapping of conventional carrier to multiplier
    renewable_strike_prices: pd.Series
        Renewable CfD strike prices for each renewable generator
    direction: Literal["bid", "offer"]
        Direction of the multiplier, either "bid" or "offer"
    """
    new_marginal_costs = (
        (df["carrier"].map(renewable_strike_prices) - df["marginal_cost"])
        # if strike price is lower than marginal cost, then we apply zero charge for bids/offers
        .clip(lower=0)
        .mul(-1 if direction == "bid" else 1)
        .fillna(df["marginal_cost"] * df["carrier"].map(multiplier).fillna(1))
    )
    assert not (isna := new_marginal_costs.isna()).any(), (
        f"Some marginal costs are NaN after applying multipliers and strike prices: {new_marginal_costs[isna].index.tolist()}"
    )

    undefined_multipliers = set(df["carrier"].unique()) - (
        set(multiplier.keys()) | set(renewable_strike_prices.index)
    )
    logger.warning(
        f"Neither bid/offer multiplier nor strike price provided for the carriers: {undefined_multipliers}"
    )

    return new_marginal_costs


def create_up_down_plants(
    constrained_network: pypsa.Network,
    unconstrained_result: pypsa.Network,
    bids_and_offers: dict[str, dict[str, float]],
    renewable_strike_prices: pd.Series,
    interconnector_bid_offer_profile: pd.DataFrame,
    gb_buses: pd.Index,
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
    renewable_strike_prices: pd.DataFrame
        Dataframe of the renewable CfD strike prices
    interconnector_bid_offer_profile: pd.DataFrame
        Interconnectors bid/offer profile for each interconnector
    gb_buses: pd.Index
        Index of GB buses
    """
    for comp in constrained_network.components:
        if comp.name not in ["Generator", "StorageUnit", "Link"]:
            continue

        constrained_network.add(
            "Carrier", [f"{comp.name} ramp up", f"{comp.name} ramp down"]
        )

        g_up = comp.static.loc[~comp.static.index.str.contains(LOAD_SHEDDING_REGEX)]
        g_down = comp.static.loc[~comp.static.index.str.contains(LOAD_SHEDDING_REGEX)]

        if comp.name != "Link":
            # Filter GB plants
            g_up = g_up.query("bus in @gb_buses and p_nom != 0")
            g_down = g_down.query("bus in @gb_buses and p_nom != 0")
        else:
            g_up = g_up.query(
                "bus0 in @gb_buses and bus1 not in @gb_buses and p_nom != 0"
            )
            g_down = g_down.query(
                "bus0 in @gb_buses and bus1 not in @gb_buses and p_nom != 0"
            )

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

        prices = {}
        for direction, df in [("offer", g_up), ("bid", g_down)]:
            if comp.name != "Link":
                # Add bid/offer multipliers for conventional generators
                prices[direction] = _apply_multiplier(
                    df,
                    bids_and_offers[f"{direction}_multiplier"],
                    renewable_strike_prices,
                    direction,
                )
            else:
                prices[direction] = interconnector_bid_offer_profile.filter(
                    regex=f".* {direction}$"
                ).rename(columns=lambda x: x.replace(" " + direction, ""))

        # Add generators that can increase dispatch
        constrained_network.add(
            "Generator",
            g_up.index,
            suffix=" ramp up",
            carrier=f"{comp.name} ramp up",
            p_min_pu=0,
            p_max_pu=up_limit.loc[:, g_up.index],
            marginal_cost=prices["offer"],
            p_nom=g_up.p_nom,
            bus=g_down.bus if comp.name not in ["Link", "Line"] else g_down.bus0,
        )

        # Add generators that can decrease dispatch
        constrained_network.add(
            "Generator",
            g_down.index,
            suffix=" ramp down",
            carrier=f"{comp.name} ramp down",
            p_min_pu=down_limit.loc[:, g_down.index],
            p_max_pu=0,
            marginal_cost=prices["bid"],
            p_nom=g_down.p_nom,
            bus=g_down.bus if comp.name not in ["Link", "Line"] else g_down.bus0,
        )

        logger.info(
            f"Added {comp.name} for carriers {g_up.carrier.unique()} that can mimic increase and decrease in dispatch"
        )


def drop_existing_eur_buses(network: pypsa.Network):
    """
    Drop existing eur buses from the network

    Parameters
    ----------
    network: pypsa.Network
        Network to finalize
    """

    eur_buses = network.buses.query("country != 'GB'").index
    network.remove("Bus", eur_buses)

    for comp in network.components[["Generator", "StorageUnit", "Store", "Load"]]:
        network.remove(comp.name, comp.static.query("bus in @eur_buses").index)

    for comp in network.components[["Link", "Line"]]:
        # Drop HVDC links / AC lines that connect two eur buses
        network.remove(
            comp.name,
            comp.static.query("bus0 in @eur_buses and bus1 in @eur_buses").index,
        )

    logger.info(
        f"Dropped generators, storage units, links and loads connected to {eur_buses} from the network"
    )


def add_single_eur_bus(network: pypsa.Network, unconstrained_result: pypsa.Network):
    """
    Add a single EUR bus to simplify the network structure

    Parameters
    ----------
    network: pypsa.Network
        Network to finalize
    unconstrained_result: pypsa.Network
        Result of the unconstrained optimization
    """

    network.add("Bus", "EUR", country="EUR")

    network.add(
        "Store",
        "EUR store",
        bus="EUR",
        e_nom=1e9,  # Large capacity to avoid energy constraints,
    )

    # Change bus1 of all interconnectors to EUR
    interconnectors = filter_interconnectors(
        network.links, "carrier in ['DC', 'ramp up', 'ramp down']"
    )
    network.links.loc[interconnectors.index, "bus1"] = "EUR"

    logger.info(
        "Added single EUR bus with a store and connected all interconnectors to it"
    )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load input networks and parameters
    network = pypsa.Network(snakemake.input.network)
    unconstrained_result = pypsa.Network(snakemake.input.unconstrained_result)
    bids_and_offers = pd.read_csv(
        snakemake.input.bids_and_offers, index_col="carrier"
    ).to_dict()
    renewable_strike_prices = pd.read_csv(
        snakemake.input.renewable_strike_prices, index_col="carrier"
    ).squeeze()
    interconnector_bid_offer_profile = pd.read_csv(
        snakemake.input.interconnector_bid_offer, index_col="snapshot", parse_dates=True
    )
    gb_buses = network.buses.query("country == 'GB'").index

    fix_dispatch(network, unconstrained_result, gb_buses)

    create_up_down_plants(
        network,
        unconstrained_result,
        bids_and_offers,
        renewable_strike_prices,
        interconnector_bid_offer_profile,
        gb_buses,
    )

    drop_existing_eur_buses(network)

    add_single_eur_bus(network, unconstrained_result)

    if snakemake.params["unconstrain_lines_and_links"]:
        # Set line capacities to infinity, so only boundary capabilities bound the optimization instead of line capacities.
        copperplate_gb(network)

    network.export_to_netcdf(snakemake.output.network)
    logger.info(f"Exported network to {snakemake.output.network}")

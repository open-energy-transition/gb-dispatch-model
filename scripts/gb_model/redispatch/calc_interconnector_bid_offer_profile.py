# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Calculate interconnector bids and offer prices
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import (
    filter_interconnectors,
    get_gb_neighbour_countries,
    marginal_costs_bus,
)

logger = logging.getLogger(__name__)


def compute_interconnector_fee(
    marginal_price_profile: pd.DataFrame,
    unconstrained_result: pypsa.Network,
    interconnectors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute the spread between the marginal costs of GB and eur node

    Parameters
    ----------
    marginal_price_profile: pd.DataFrame
        Dataframe of marginal costs at each bus
    unconstrained_result: pypsa.Network
        pypsa network with the results of the unconstrained optimization
    interconnectors: pd.DataFrame
        Dataframe of interconnectors between GB and eur bus
    """

    fee_profile = pd.DataFrame(
        index=unconstrained_result.snapshots, columns=interconnectors.index
    )

    for connector, row in interconnectors.iterrows():
        #  Replace the load shedding generator costs with the otherwise most expensive generator

        p_gb = marginal_price_profile[row.bus0]
        p_eur = marginal_price_profile[row.bus1]
        # Price spread calculated as the absolute difference
        fee_profile[connector] = (p_gb - p_eur).abs()

    logger.info("Calculated interconnector fee")
    return fee_profile


def get_price_spread(
    generator_marginal_prices: pd.DataFrame,
    marginal_electricity_price: float,
) -> str:
    """
    To obtain price spread between generators at the eur node and it's marginal electricity price

    Parameters
    ----------
    generator_marginal_prices: pd.DataFrame
        Dataframe of marginal costs for each carrier at the eur node
    marginal_electricity_price: float
        Marginal cost of electricity at the EUR node for the interconnector
    """

    # Calculate price difference between generator marginal costs and GB marginal electricity price
    price_spread = (
        (generator_marginal_prices - marginal_electricity_price).abs().sort_values()
    )

    least_spread_carrier = price_spread.index[0]
    least_spread = price_spread.iloc[0]
    return least_spread, least_spread_carrier


def get_eur_marginal_generator(
    marginal_price_profile: pd.DataFrame,
    unconstrained_result: pypsa.Network,
    bids_and_offers_multipliers: dict[dict[str, float]],
) -> pd.DataFrame:
    """
    Obtain the marginal generator at eur node at each timestamp

    Parameters
    ----------
    marginal_price_profile: pd.DataFrame
        Bus shadow price profile indexed by time for each eur bus
    unconstrained_result: pypsa.Network
        Unconstrained optimization model result
    bids_and_offers_multipliers: dict[dict[str, float]]
        Bid and offer multiplier to be applied for conventional generation
    """
    gb_neighbours = get_gb_neighbour_countries(unconstrained_result)
    columns = [f"{x} bid" for x in gb_neighbours] + [
        f"{x} offer" for x in gb_neighbours
    ]
    eur_marginal_gen_profile = pd.DataFrame(
        index=unconstrained_result.snapshots, columns=columns
    )

    bid_multiplier = bids_and_offers_multipliers["bid_multiplier"]
    offer_multiplier = bids_and_offers_multipliers["offer_multiplier"]
    eur_buses = unconstrained_result.buses.query(
        "country != 'GB' and carrier == 'AC'"
    ).index

    # Function to identify marginal carrier
    def _identify_marginal_carrier(electricity_price):
        # Check if electricity price falls in marginal price range of any of the generators
        marginal_carrier_rows = marginal_price_range.loc[
            (electricity_price >= marginal_price_range["min"])
            & (electricity_price <= marginal_price_range["max"])
        ]

        # If electricity price does not match any of the generator marginal price ranges
        if marginal_carrier_rows.empty:
            # Get rows where electricity price lies in between the marginal price ranges of two generators
            # i.e., exceeds the maximum of one row and lesser than the minimum of next row
            marginal_carrier_rows = pd.concat(
                [
                    # last row where marginal electricity price exceeds maximum of generator price range
                    marginal_price_range[
                        (marginal_price_range["max"] < electricity_price)
                    ].iloc[-1],
                    # first row where marginal electricity price is lesser than minimum of generator price range
                    marginal_price_range[
                        (marginal_price_range["min"] > electricity_price)
                    ].iloc[0],
                ],
                axis=1,
            ).T

            # The marginal generator is set as the generator which has the least deviation from
            # the extremities of the two rows selected in the previous step
            if (
                electricity_price - marginal_carrier_rows.iloc[0]["max"]
                > marginal_carrier_rows.iloc[1]["min"] - electricity_price
            ):
                # Minimum of generator marginal price range is closer to marginal electricity price
                marginal_carrier_row = marginal_carrier_rows.iloc[1]
            else:
                # Maximum of generator marginal price range in closer to marginal electricity price
                marginal_carrier_row = marginal_carrier_rows.iloc[0]
        else:
            # Row where marginal electricity price is within generator marginal price range
            marginal_carrier_row = marginal_carrier_rows.iloc[0]
        # Get the marginal carrier name
        marginal_carrier = marginal_carrier_row.name
        if marginal_carrier in bid_multiplier.keys():
            bid_cost = electricity_price * bid_multiplier[marginal_carrier]
            offer_cost = electricity_price * offer_multiplier[marginal_carrier]
        else:
            bid_cost = offer_cost = electricity_price

        return bid_cost, offer_cost

    for bus in gb_neighbours:
        generator_marginal_prices = pd.concat(
            [marginal_costs_bus(b, unconstrained_result) for b in eur_buses]
        )

        # Price range of each carrier in EUR (min and max values)
        marginal_price_range = (
            generator_marginal_prices.round(3)
            .groupby("carrier")
            .agg(["min", "max"])
            .sort_values(by="min")
        )

        marginal_electricity_price = marginal_price_profile[bus].round(3)
        eur_marginal_gen_profile[[f"{bus} bid", f"{bus} offer"]] = (
            marginal_electricity_price.apply(
                lambda x: _identify_marginal_carrier(x)
            ).tolist()
        )
        logger.info(
            f"Calculated marginal generator profile and it's associate bid/offer at {bus}"
        )

    return eur_marginal_gen_profile


def calculate_interconnector_loss(
    unconstrained_result: pypsa.Network, interconnectors: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate capacity weighted average of interconnector losses at each foreign market

    Parameters
    ----------
    unconstrained_result: pypsa.Network
        Result of the unconstrained optimization
    interconnectors: pd.DataFrame
        Dataframe of interconnectors and their pypsa parameters
    """
    interconnector_grouping = (
        interconnectors.index.to_series().groupby(interconnectors["bus1"]).agg(list)
    )
    loss_profile = pd.DataFrame(
        index=unconstrained_result.snapshots, columns=interconnector_grouping.index
    )
    for bus in interconnector_grouping.index:
        group = interconnector_grouping.loc[bus]
        # Capacity weighted average of interconnectors connected to each eur node
        loss_profile[bus] = (
            (
                unconstrained_result.links_t.p1[group]
                - unconstrained_result.links_t.p0[group] * -1
            )
            .abs()  # interconnector loss between GB and eur node
            .mul(interconnectors.loc[group].p_nom)
            .sum(axis=1)
            .div(interconnectors.loc[group].p_nom.sum())
        )

    if loss_profile.isna().any().any():
        logger.info(
            f"NaNs found in the loss profile at buses {loss_profile.columns[loss_profile.isna().any()].tolist()}"
        )
        loss_profile = loss_profile.fillna(0)

    logger.info("Calculated the interconnector loss profile")
    return loss_profile


def calc_interconnector_bids_and_offers(
    unconstrained_result: pypsa.Network,
    interconnectors: pd.DataFrame,
    eur_marginal_gen_profile: pd.DataFrame,
    interconnector_fee_profile: pd.DataFrame,
    loss_profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the bids and offers for import, export and floating condition of each interconnector

    Parameters
    ----------
        unconstrained_result: pypsa.Network
            Result of unconstrained optimization
        interconnectors: pd.DataFrame
            Dataframe of interconnectors from the pypsa network.links
        eur_marginal_gen_profile: pd.DataFrame
            bid / offer for the marginal generator at each eur bus
        interconnector_fee_profile: pd.DataFrame
            Spread in marginal prices for each interconnector
        loss_profile: pd.DataFrame
            Weighted capacity average of losses of interconnectors connected to each eur bus
    """
    profile_dict = {}

    for connector, row in interconnectors.iterrows():
        gb_bus = row["bus0"]
        eur_bus = row["bus1"]

        fee = interconnector_fee_profile[connector]
        # Reset bus shadow price if load generator set the LMP
        p_gb = unconstrained_result.buses_t.marginal_price[gb_bus]
        p_eur = unconstrained_result.buses_t.marginal_price[eur_bus]
        bid = eur_marginal_gen_profile[f"{eur_bus} bid"]
        offer = eur_marginal_gen_profile[f"{eur_bus} offer"]
        loss = loss_profile[eur_bus]

        # Import bid profile
        profile_dict[(connector, "import_bid")] = (
            p_gb - p_eur * (1 + loss) - bid * (1 + loss)
        )

        # Import offer profile / Import float profile
        profile_dict[(connector, "import_offer")] = profile_dict[
            (connector, "float_import")
        ] = fee + offer * (1 + loss)

        # Export float profile / Export bid profile
        profile_dict[(connector, "float_export")] = profile_dict[
            (connector, "export_bid")
        ] = fee - bid * (1 - loss)

        # Export offer profile
        profile_dict[(connector, "export_offer")] = (
            p_eur * (1 - loss) - p_gb + offer * (1 - loss)
        )

    profile_df = pd.DataFrame(profile_dict)

    return profile_df


def assign_bid_offer(
    unconstrained_result: pypsa.Network,
    interconnectors: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate bid/offer for each interconnector based on the status of the interconnector at each time step

    Parameters
    ----------
    unconstrained_result: pypsa.Network
        Result of the unconstrained optimization
    interconnectors: pd.DataFrame
        List of interconnectors between EU and GB
    profiles: pd.DataFrame
        six different bid/offer profiles for each interconnector that can be assigned to it based on it's status
    """

    gb_power = unconstrained_result.links_t.p0[interconnectors.index]
    conditions = [
        gb_power < 0,  # interconnector importing
        gb_power == 0,  # interconnector float
        gb_power > 0,  # interconnector exporting
    ]

    def _filter_profiles(df, key):
        df_filtered = df[df.columns[df.columns.get_level_values(1) == key]]
        df_filtered.columns = df_filtered.columns.droplevel(1)
        return df_filtered

    bid_profiles = [
        _filter_profiles(profiles, "import_bid"),
        _filter_profiles(profiles, "float_export"),
        _filter_profiles(profiles, "export_bid"),
    ]
    offer_profiles = [
        _filter_profiles(profiles, "import_offer"),
        _filter_profiles(profiles, "float_import"),
        _filter_profiles(profiles, "export_offer"),
    ]
    # for connector in interconnectors:
    bid_profiles = pd.DataFrame(
        np.select(conditions, bid_profiles), columns=interconnectors.index
    ).add_suffix(" bid")
    offer_profiles = pd.DataFrame(
        np.select(conditions, offer_profiles), columns=interconnectors.index
    ).add_suffix(" offer")

    interconnector_profile = pd.concat([bid_profiles, offer_profiles], axis=1)
    interconnector_profile.index = unconstrained_result.snapshots
    logger.info(
        "Assigned the bid/offer to each interconnector based on the status of the interconnector"
    )

    return interconnector_profile


def compose_data(
    unconstrained_result: pypsa.Network,
    bids_and_offers_multipliers_path: str,
):
    """
    Main composition function to process the data

    Parameters
    ----------
    unconstrained_result: pypsa.Network
        Unconstrained optimization network
    bids_and_offers_multipliers: str
        Path to Bids and offers multipliers CSV data
    """
    # Bids and offers multipliers
    bids_and_offers_multipliers = pd.read_csv(
        bids_and_offers_multipliers_path, index_col="carrier"
    ).to_dict()

    # Get list of interconnectors between GB and Eur
    interconnectors = filter_interconnectors(
        unconstrained_result.links, "carrier == 'DC' and p_nom != 0"
    )

    # Compute shadow prices at buses
    ac_buses = unconstrained_result.buses.query("carrier == 'AC'").index
    marginal_price_profile = unconstrained_result.buses_t.marginal_price[ac_buses]

    # Compute interconnector fee
    interconnector_fee_profile = compute_interconnector_fee(
        marginal_price_profile,
        unconstrained_result,
        interconnectors,
    )

    # Compute marginal generator at the eur bus
    eur_marginal_gen_profile = get_eur_marginal_generator(
        marginal_price_profile,
        unconstrained_result,
        bids_and_offers_multipliers,
    )

    # Compute interconnector losses
    loss_profile = calculate_interconnector_loss(unconstrained_result, interconnectors)

    # Compute 6 different bid/offer profiles for each interconnector
    profile_df = calc_interconnector_bids_and_offers(
        unconstrained_result,
        interconnectors,
        eur_marginal_gen_profile,
        interconnector_fee_profile,
        loss_profile,
    )

    # Assign bid/offer from one of the six profiles to interconnector at each time stamp depending on it's status
    interconnector_profile = assign_bid_offer(
        unconstrained_result, interconnectors, profile_df
    )

    return interconnector_profile


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, year=2022)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    interconnector_profile = compose_data(
        unconstrained_result=pypsa.Network(snakemake.input.unconstrained_result),
        bids_and_offers_multipliers_path=snakemake.input.bids_and_offers,
    )

    interconnector_profile.to_csv(snakemake.output.bid_offer_profile)
    logger.info(
        f"Exported interconnector bid/offer profiles to {snakemake.output.bid_offer_profile}"
    )

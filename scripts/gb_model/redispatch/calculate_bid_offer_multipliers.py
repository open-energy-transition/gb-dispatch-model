# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Calculate bid and offer multipliers
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_scenario_name
from scripts.gb_model.generators.assign_costs import (
    DEFAULT_SETS,
    assign_technical_and_costs_defaults,
    load_costs,
    load_fes_carbon_costs,
    load_fes_power_costs,
)

logger = logging.getLogger(__name__)


def calc_bid_offer_multipliers(
    bid_offer_costs_path: list[str], df_cost: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate bid and offer multipliers as a fraction of the average marginal cost for each technology

    Parameters
    ----------
    bid_offer_costs_path: list[str]
        Filepaths of bid and offer costs data for each technology year-wise
    df_costs: pd.DataFrame
        DataFrame containing the average marginal cost for each technology across years

    Returns
    -------
    pd.DataFrame: DataFrame containing the bid and offer multipliers for each technology

    """
    # Read bid offer data
    df_bid_offer = pd.concat(
        [
            pd.read_csv(path, index_col="carrier", parse_dates=["settlementDate"])
            for path in bid_offer_costs_path
        ]
    )

    df_bod_mean = {}

    # BidOfferPairId is an indication of the bandwidth within which a BMunit can increase / decrease it's power output.
    # -ve pairId's indicate bids and +ve pairId's indicate offers
    # The bid / offer price can vary with the pairId - for simplicity an average of the prices over the pair id's is used here
    mask = [pd.Grouper(key="settlementDate", freq="QE"), "carrier"]
    df_bod_mean['bidPrice'] = (
        df_bid_offer.query("bidOfferPairId < 0")
        .groupby(mask)["bidPrice"]
        .mean()
    )
    df_bod_mean['offerPrice'] = (
        df_bid_offer.query("bidOfferPairId > 0")
        .groupby(mask)["offerPrice"]
        .mean()
    )
    df_bod_mean = pd.DataFrame(df_bod_mean)
    df_bod_mean.reset_index(inplace=True)

    df_bod_mean["year"] = df_bod_mean["settlementDate"].dt.year
    df_bod_mean["quarter"] = df_bod_mean["settlementDate"].dt.month.replace(
        {3: 1, 6: 2, 9: 3, 12: 4}
    )
    df_bod_mean = df_bod_mean.reset_index().set_index(["carrier", "year", "quarter"])

    df_cost_filtered = df_cost.query("set in @set", local_dict={"set": DEFAULT_SETS})

    # Join bid offer quarterly dataframe with cost dataframe
    df_multipliers = df_bod_mean.join(
        df_cost_filtered.set_index(["carrier", "year", "quarter"])
    )

    # Compute bid and offer multipliers for each quarter across the years
    df_multipliers["bid_multiplier"] = (
        df_multipliers["bidPrice"].abs() / df_multipliers["marginal_cost"]
    )

    df_multipliers["offer_multiplier"] = (
        df_multipliers["offerPrice"] / df_multipliers["marginal_cost"]
    )

    # Calculate average multiplier for each carrier
    df_multipliers = (
        df_multipliers.reset_index()
        .groupby("carrier")[["bid_multiplier", "offer_multiplier"]]
        .mean()
    )

    logger.info("Calculated bid and offer multipliers for each carrier.")

    return df_multipliers


def add_historical_fuel_prices(
    historical_fuel_path: str,
    years: list[int],
    dukes_config: dict,
    fes_power_costs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Get historical fuel prices

    Parameters
    ----------
    historical_fuel_path: str
        Path to historical fuel prices CSV file
    years: list[int]
        List of years to consider
    dukes_config: dict
        Configuration parameters when reading the historical fuel price data
    fes_power_costs: pd.DataFrame
        DataFrame containing FES power costs to which we want to add the historical fuel prices

    Returns
    -------
        pd.DataFrame: Updated FES power costs DataFrame with historical fuel prices instead of the original fuel costs.
    """

    sheet_config = dukes_config["sheet-config"]
    sheet_name = sheet_config.pop("sheet_name")

    # Get quarterly historical fuel prices
    df = pd.read_excel(historical_fuel_path, sheet_name=sheet_name, **sheet_config)

    # Filter the required years of data
    df = df.query("Year in @y", local_dict={"y": years})
    df["Quarter"] = (
        df["Quarter"].replace(dukes_config["quarter_mapping"]).infer_objects(copy=False)
    )
    new_cols = {
        i: df[k].values for k, v in dukes_config["column_mapping"].items() for i in v
    }
    df_renamed = (
        df.set_index(["Year", "Quarter"])
        .assign(**new_cols)
        .loc[:, new_cols.keys()]
        .replace("..", np.nan)
        .rename_axis(index={"Year": "year", "Quarter": "quarter"}, columns="technology")
        .stack()
        .to_frame("fuel")
    )
    df_renamed *= 10  # Convert pence per kWh to GBP per MWh

    new_fes_power_costs = (
        df_renamed.assign(VOM=np.nan)
        .to_xarray()
        .broadcast_like(fes_power_costs.to_xarray())
        .fillna(fes_power_costs.to_xarray())
        .to_dataframe()
    )
    return new_fes_power_costs


def bid_offer_powerplants(df: pd.DataFrame, bid_offer_years: list[int]) -> pd.DataFrame:
    """
    Create a dummy powerplant dataframe for the years for which we have bid and offer data.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame containing powerplant data with columns 'bus', 'carrier', 'set', 'year', 'p_nom'
    bid_offer_years: list[int]
        List of years for which we have bid and offer data

    Returns
    -------
        pd.DataFrame: `df` with year column changed to the years in `bid_offer_years` and p_nom set to -1 for these years (since we won't be using this column).
    """
    df_bid_offer_years = df.pivot(
        index=["bus", "carrier", "set"], columns="year", values="p_nom"
    ).assign(**{str(year): -1 for year in bid_offer_years})
    df_bid_offer_years.columns = df_bid_offer_years.columns.astype(int)
    df_bid_offer_years = (
        df_bid_offer_years.loc[:, bid_offer_years]
        .stack()
        .to_frame("p_nom")
        .reset_index()
    )
    return df_bid_offer_years


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)
    fes_scenario = get_scenario_name(snakemake)
    costs_config = snakemake.params.costs_config
    bid_offer_years = [int(Path(x).stem) for x in snakemake.input.bid_offer_data]
    df = pd.read_csv(snakemake.input.powerplants)
    # Load costs data
    costs = load_costs(snakemake.input.tech_costs, costs_config)
    fes_power_costs = load_fes_power_costs(
        snakemake.input.fes_power_costs, fes_scenario
    )
    fes_carbon_costs = load_fes_carbon_costs(
        snakemake.input.fes_carbon_costs, fes_scenario
    )
    updated_fes_power_costs = add_historical_fuel_prices(
        historical_fuel_path=snakemake.input.historical_fuel_price,
        years=bid_offer_years,
        dukes_config=snakemake.params.dukes_config,
        fes_power_costs=fes_power_costs,
    )
    df_bid_offer_years = bid_offer_powerplants(df, bid_offer_years)

    # Enrich powerplants with technical/cost parameters
    df_cost = assign_technical_and_costs_defaults(
        df=df_bid_offer_years,
        fes_power_costs=updated_fes_power_costs,
        costs=costs,
        fes_carbon_costs=fes_carbon_costs,
        costs_config=costs_config,
    )

    df_multipliers = calc_bid_offer_multipliers(
        bid_offer_costs_path=snakemake.input.bid_offer_data, df_cost=df_cost
    )

    df_multipliers.to_csv(snakemake.output.csv)
    logger.info(f"Saved bid and offer multipliers to {snakemake.output.csv}")

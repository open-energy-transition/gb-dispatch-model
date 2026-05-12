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
    _calculate_marginal_costs,
    _load_costs,
    _load_fes_carbon_costs,
    _load_fes_power_costs,
)

logger = logging.getLogger(__name__)


def calculate_costs(
    fes_power_costs_path: str,
    fes_carbon_costs_path: str,
    fes_scenario: str,
    tech_costs_path: str,
    costs_config: dict,
    technology_mapping: dict[str, str],
    years: list[int],
    historical_fuel_cost: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate marginal costs for each technology

    Parameters
    ----------
    fes_power_costs_path: str
        Filepath to FES power costs CSV
    fes_carbon_costs_path: str
        Filepath to FES carbon costs CSV
    fes_scenario: str
        FES Scenario name
    tech_costs_path: str
        Filepath to technology costs CSV
    costs_config: dict
        Configuration dictionary for costs
    technology_mapping: dict[str, str]
        Mapping of technology names
    years: list[int]
        List of years to consider
    historical_fuel_cost: pd.DataFrame
        Historical fuel prices index by carrier, year and quarter
    """

    # Load FES power and carbon costs
    fes_power_costs = _load_fes_power_costs(fes_power_costs_path, fes_scenario)
    fes_carbon_costs = _load_fes_carbon_costs(fes_carbon_costs_path, fes_scenario)
    costs = _load_costs(tech_costs_path, costs_config)

    # Map FES technology names to PyPSA carriers
    power_tech_map = {
        v: k.split(" ")[0]
        for k, v in costs_config["fes_VOM_carrier_mapping"].items()
        if "CHP" not in k
    }

    fes_power_costs["carrier"] = (
        fes_power_costs.index.get_level_values("technology")
        .map(power_tech_map)
        .str.upper()
        .map(technology_mapping)
    )

    # Create a multi-index dataframe for technologies and years and merge the FES costs data
    multi_index = pd.MultiIndex.from_product(
        [technology_mapping.values(), years, np.arange(1, 5)],
        names=["carrier", "year", "quarter"],
    )
    df_tech = pd.DataFrame(index=multi_index)

    # Merge FES power costs
    df_tech = df_tech.join(fes_power_costs.reset_index().set_index(["carrier", "year"]))

    # Merge technology costs data
    df_tech = (
        df_tech.reset_index()
        .set_index("carrier")
        .join(costs[["efficiency", "CO2 intensity"]])
    ).reset_index()

    for fuel in historical_fuel_cost.columns:
        # Replace existing fuel costs from FES with DUKES historical fuel prices
        fuel_fuel_keys = [
            k
            for k, v in costs_config["carrier_gap_filling"]["fuel"].items()
            if v == fuel
        ]
        fuel_rows = df_tech.carrier.isin(fuel_fuel_keys)
        df_tech.loc[fuel_rows, "fuel"] = (
            df_tech[fuel_rows][["year", "quarter"]]
            .apply(tuple, axis=1)
            .map(historical_fuel_cost[fuel].to_dict())
        )

    df_tech = _calculate_marginal_costs(
        df_tech,
        costs,
        costs_config,
        fes_carbon_costs,
    )

    logger.info(f"Calculated marginal costs for each technology for the years {years}.")
    return df_tech


def calc_bid_offer_multipliers(
    bid_offer_costs_path: list[str],
    df_cost: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate bid and offer multipliers as a fraction of the average marginal cost for each technology

    Parameters
    ----------
    bid_offer_costs_path: list[str]
        Filepaths of bid and offer costs data for each technology year-wise
    df_costs: pd.DataFrame
        DataFrame containing the average marginal cost for each technology across years
    """
    # Read bid offer data
    df_bid_offer = pd.concat(
        [
            pd.read_csv(path, index_col="carrier", parse_dates=["settlementDate"])
            for path in bid_offer_costs_path
        ]
    )
    df_bid_offer["year"] = df_bid_offer["settlementDate"].dt.year
    df_bid_offer["quarter"] = df_bid_offer["settlementDate"].dt.month.replace(
        {3: 1, 6: 2, 9: 3, 12: 4}
    )
    df_bid_offer = df_bid_offer.reset_index().set_index(["carrier", "year", "quarter"])

    df_cost.set_index(["carrier", "year", "quarter"], inplace=True)

    # Join bid offer quarterly dataframe with cost dataframe
    df_multipliers = df_bid_offer.join(df_cost)

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


def get_historical_fuel_prices(
    historical_fuel_path: str, years: list[int], dukes_config: dict
):
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
    """

    sheet_config = dukes_config["sheet-config"]
    sheet_name = sheet_config.pop("sheet_name")

    # Get quarterly historical fuel prices
    df = pd.read_excel(historical_fuel_path, sheet_name=sheet_name, **sheet_config)

    # Filter the required years of data
    df = df.query("Year in @y", local_dict={"y": years})
    df["Quarter"] = df["Quarter"].replace(dukes_config["quarter_mapping"])
    df.set_index(["Year", "Quarter"], inplace=True)
    df.rename(columns=dukes_config["column_mapping"], inplace=True)
    df = df[["gas", "coal", "oil"]]
    df[df.columns] = df[df.columns].replace("..", np.nan)
    df *= 10  # Convert pence per kWh to GBP per MWh
    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    bid_offer_years = [int(Path(x).stem) for x in snakemake.input.bid_offer_data]
    historical_fuel_cost = get_historical_fuel_prices(
        historical_fuel_path=snakemake.input.historical_fuel_price,
        years=bid_offer_years,
        dukes_config=snakemake.params.dukes_config,
    )

    df_cost = calculate_costs(
        fes_power_costs_path=snakemake.input.fes_power_costs,
        fes_carbon_costs_path=snakemake.input.fes_carbon_costs,
        fes_scenario=get_scenario_name(snakemake),
        tech_costs_path=snakemake.input.tech_costs,
        costs_config=snakemake.params.costs_config,
        technology_mapping=snakemake.params.technology_mapping,
        years=bid_offer_years,
        historical_fuel_cost=historical_fuel_cost,
    )

    df_multipliers = calc_bid_offer_multipliers(
        bid_offer_costs_path=snakemake.input.bid_offer_data, df_cost=df_cost
    )

    df_multipliers.to_csv(snakemake.output.csv)
    logger.info(f"Saved bid and offer multipliers to {snakemake.output.csv}")

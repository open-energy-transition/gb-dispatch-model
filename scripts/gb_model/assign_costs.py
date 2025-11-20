# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
Costs assigner.

This script enriches powerplants CSV data with costs information.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _ensure_column_with_default(
    df: pd.DataFrame, col: str, default: float, units: str = ""
) -> pd.DataFrame:
    """Helper to ensure column exists and has no NaN values."""
    unit_str = f" {units}" if units else ""

    if col not in df.columns:
        logger.warning(f"No {col} column; creating with default {default}{unit_str}")
        df[col] = default
    else:
        missing = df[col].isna().sum()
        if missing > 0:
            logger.warning(
                f"Missing {col} for {missing} rows; using default {default}{unit_str}"
            )
            df[col] = df[col].fillna(default)

    return df


def _load_costs(
    tech_costs_path: str,
    costs_config: dict[str, dict],
) -> pd.DataFrame:
    """Load technology costs data."""
    costs = pd.read_csv(tech_costs_path, index_col=[0, 1])

    # correct units to MW
    costs.loc[costs.unit.str.contains("/kW"), "value"] *= 1e3
    costs.loc[costs.unit.str.contains("/GW"), "value"] /= 1e3
    costs.unit = costs.unit.str.replace("/kW", "/MW")
    costs.unit = costs.unit.str.replace("/GW", "/MW")

    # Convert costs to GBP from EUR or USD
    costs.loc[costs.unit.str.contains("EUR"), "value"] /= costs_config["GBP_to_EUR"]
    costs.loc[costs.unit.str.contains("USD"), "value"] /= costs_config["GBP_to_USD"]
    costs.unit = costs.unit.str.replace("EUR", "GBP")
    costs.unit = costs.unit.str.replace("USD", "GBP")

    # min_count=1 is important to generate NaNs which will be filled with default characteristics later
    costs = costs.value.unstack(level=1).groupby("technology").sum(min_count=1)

    # Keep only relevant cost columns
    costs = costs[costs_config["relevant_cost_columns"]]

    return costs


def _load_fes_power_costs(
    fes_power_costs_path: str,
    fes_scenario: str,
) -> pd.DataFrame:
    """
    Loads FES power cost data, filters by scenario and relevant cost types,
    then pivots to create a DataFrame with multi-index (Sub Type, year)
    and columns for each Cost Type (fuel, VOM).

    Args:
        fes_power_costs_path (str): Path to FES power costs CSV file.
        fes_scenario (str): FES scenario name to filter (e.g., "leading the way").

    Returns:
        pd.DataFrame: Multi-indexed DataFrame with:
            - Index: ["technology", "year"]
            - Columns: ["fuel", "VOM"] (Variable Other Work Costs)
            - Values: Cost data in GBP
    """
    # Load FES power costs
    fes_power_costs = pd.read_csv(fes_power_costs_path)

    # Filter for the selected FES scenario
    fes_power_costs = fes_power_costs[
        (fes_power_costs["Scenario"].str.lower().isin([fes_scenario, "all scenarios"]))
    ]

    # Keep relevant cost types
    fes_power_costs = fes_power_costs[
        fes_power_costs["Cost Type"]
        .str.lower()
        .isin(["variable other work costs", "fuel cost"])
    ]

    # Pivot to create multi-index with Cost Type as columns
    fes_power_costs.loc[:, "technology"] = (
        fes_power_costs["Type"] + "-" + fes_power_costs["Sub Type"]
    )
    fes_power_costs_pivoted = fes_power_costs.pivot_table(
        index=["technology", "year"],
        columns="Cost Type",
        values="data",
    )

    # Rename columns to match expected names
    fes_power_costs_pivoted = fes_power_costs_pivoted.rename(
        columns={
            "Fuel Cost": "fuel",
            "Variable Other Work Costs": "VOM",
        }
    )

    return fes_power_costs_pivoted


def _load_fes_carbon_costs(
    fes_carbon_costs_path: str,
    fes_scenario: str,
) -> pd.DataFrame:
    """
    Load FES carbon costs data.

    Args:
        fes_carbon_costs_path: Path to FES carbon costs CSV
        fes_scenario: FES scenario name (e.g., "leading the way")

    Returns:
        DataFrame with year index and carbon_cost column (£/tCO2)

    Steps:
        1. Load FES carbon costs CSV
        2. Filter by scenario
        3. Select year and data columns, set year as index
        4. Rename data column to carbon_cost
    """
    # Load FES carbon costs
    fes_carbon_costs = pd.read_csv(fes_carbon_costs_path)

    # Filter for the selected FES scenario
    fes_carbon_costs = fes_carbon_costs[
        fes_carbon_costs["Scenario"].str.lower() == fes_scenario
    ]

    # Select relevant columns
    fes_carbon_costs = fes_carbon_costs[["year", "data"]].set_index("year")

    # Rename columns to match expected names
    fes_carbon_costs.rename(columns={"data": "carbon_cost"}, inplace=True)

    return fes_carbon_costs


def _integrate_fes_power_costs(
    df: pd.DataFrame,
    fes_power_costs: pd.DataFrame,
    costs_config: dict[str, dict],
    default_characteristics: dict[str, dict],
) -> pd.DataFrame:
    """
    Integrate FES power costs into the powerplants DataFrame.

    Args:
        df (pd.DataFrame): Powerplants DataFrame with 'carrier', 'set', and 'year' columns.
        fes_power_costs (pd.DataFrame): FES power costs DataFrame with multi-index
            (Sub Type, year) and columns for each Cost Type (fuel, VOM).
        costs_config (dict): Configuration dict containing:
            - fes_costs_carrier_mapping: Mapping from carrier names to FES Sub Type name.
        default_characteristics (dict): Default characteristics for filling missing values.

    Returns:
        pd.DataFrame: Updated powerplants DataFrame with integrated FES power costs.
    """
    # Create a carrier_set column for mapping
    df["carrier_set"] = df["carrier"] + " " + df["set"]

    # Map carrier_set to FES costs technology name to merge VOM data
    df["fes_VOM_techs"] = df["carrier_set"].map(costs_config["fes_VOM_carrier_mapping"])

    # Merge FES VOM data using technology name and year
    df = df.merge(
        fes_power_costs[["VOM"]],
        left_on=["fes_VOM_techs", "year"],
        right_index=True,
        how="left",
        suffixes=("_pypsa", ""),
    )

    # Map carrier_set to FES costs technology name to merge fuel cost data
    df["fes_fuel_techs"] = df["carrier_set"].map(
        costs_config["fes_fuel_carrier_mapping"]
    )

    # Merge FES fuel cost data using technology name and year
    df = df.merge(
        fes_power_costs[["fuel"]],
        left_on=["fes_fuel_techs", "year"],
        right_index=True,
        how="left",
        suffixes=("_pypsa", ""),
    )

    # Fill VOM, fuel costs, efficiency, and CO2 intensity with default characteristics from config where FES data is missing
    for col in ["VOM", "fuel", "efficiency", "CO2 intensity"]:
        if df[col].isna().any():
            logger.info(
                f"Filling {df[col].isna().sum()} missing {col} values with default: {default_characteristics[col]['data']}"
            )
        df[col] = df[col].fillna(default_characteristics[col]["data"])

    # Drop temporary columns
    df = df.drop(
        columns=[
            "carrier_set",
            "VOM_pypsa",
            "fuel_pypsa",
            "fes_VOM_techs",
            "fes_fuel_techs",
        ]
    )

    return df


def assign_technical_and_costs_defaults(
    ppl_path: str,
    tech_costs_path: str,
    fes_power_costs_path: str,
    fes_carbon_costs_path: str,
    default_characteristics: dict[str, dict],
    costs_config: dict[str, dict],
    fes_scenario: str,
) -> pd.DataFrame:
    """
    Enrich powerplants dataframe with cost and technical parameters.

    Args:
        ppl_path: Path to powerplant data CSV file
        tech_costs_path: Path to technology costs CSV file
        fes_power_costs_path: Path to FES power costs CSV file
        fes_carbon_costs_path: Path to FES carbon costs CSV file
        default_characteristics: Default values for technical and cost parameters
        costs_config: Configuration dict containing mappings and conversion rates
        fes_scenario: FES scenario name (e.g., "leading the way")

    Returns:
        Enriched powerplants DataFrame with efficiency, marginal_cost, VOM, fuel,
        CO2 intensity, capital_cost, lifetime, build_year, and unique index

    Steps:
        1. Load technology costs, FES power costs, and FES carbon costs
        2. Join technology costs on carrier
        3. Fill CO2 intensity and fuel costs using carrier_fuel_mapping
        4. Format bus and build_year columns
        5. Integrate FES power costs (VOM and fuel)
        6. Integrate FES carbon costs
        7. Calculate marginal_cost from VOM, fuel, efficiency, CO2 intensity, and carbon_cost
        8. Create unique index (bus carrier-year-idx)
    """
    # Load powerplant data
    df = pd.read_csv(ppl_path, index_col=0)

    # Load costs data
    costs = _load_costs(tech_costs_path, costs_config)
    fes_power_costs = _load_fes_power_costs(fes_power_costs_path, fes_scenario)
    fes_carbon_costs = _load_fes_carbon_costs(fes_carbon_costs_path, fes_scenario)
    logger.info("Loaded technology costs and FES power and carbon costs data")

    # Join cost data
    df = df.join(costs[costs_config["marginal_cost_columns"]], on="carrier")

    # Fill CO2 intensities and fuel costs using carrier_fuel_mapping because fuel names might differ (Eg. CCGT uses gas)
    df["CO2 intensity"] = df["CO2 intensity"].fillna(
        df["carrier"]
        .map(costs_config["carrier_fuel_mapping"])
        .map(costs["CO2 intensity"])
    )
    df["fuel"] = df["fuel"].fillna(
        df["carrier"].map(costs_config["carrier_fuel_mapping"]).map(costs["fuel"])
    )

    # Format bus and build_year columns
    df["bus"] = df["bus"].astype(str)
    df["build_year"] = df["year"].astype(int)

    # Add country columns
    df["country"] = df["bus"].str[:2]

    # Integrate FES power costs
    df = _integrate_fes_power_costs(
        df, fes_power_costs, costs_config, default_characteristics
    )

    # Integrate FES carbon costs
    df = df.join(fes_carbon_costs, on="year")

    # Calculate marginal cost if possible
    if all(col in df.columns for col in ["VOM", "fuel", "efficiency", "carbon_cost"]):
        df["marginal_cost"] = (
            df["VOM"]
            + df["fuel"] / df["efficiency"]
            + df["CO2 intensity"] * df["carbon_cost"] / df["efficiency"]
        )

    # Set capital costs from default_characteristics
    for col in ["capital_cost", "lifetime"]:
        df = _ensure_column_with_default(
            df,
            col,
            default_characteristics[col]["data"],
        )

    # Create unique index: "bus carrier-year-idx"
    df["idx_counter"] = df.groupby(["bus", "carrier", "year"]).cumcount()
    df.index = (
        df["bus"]
        + " "
        + df["carrier"]
        + "-"
        + df["year"].astype(int).astype(str)
        + "-"
        + df["idx_counter"].astype(str)
    )
    df = df.drop(columns=["idx_counter", "carbon_cost"])

    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    tech_costs_path = snakemake.input.tech_costs
    fes_power_costs_path = snakemake.input.fes_power_costs
    fes_carbon_costs_path = snakemake.input.fes_carbon_costs
    ppl_path = snakemake.input.fes_powerplants

    # Load all the params
    default_characteristics = snakemake.params.default_characteristics
    costs_config = snakemake.params.costs_config
    fes_scenario = snakemake.params.fes_scenario

    # Enrich powerplants with technical/cost parameters
    df_powerplants = assign_technical_and_costs_defaults(
        ppl_path=ppl_path,
        tech_costs_path=tech_costs_path,
        fes_power_costs_path=fes_power_costs_path,
        fes_carbon_costs_path=fes_carbon_costs_path,
        default_characteristics=default_characteristics,
        costs_config=costs_config,
        fes_scenario=fes_scenario,
    )
    logger.info("Enriched powerplants with cost and technical parameters")

    # Save with index (contains unique generator IDs)
    df_powerplants.to_csv(snakemake.output.enriched_powerplants, index=True)

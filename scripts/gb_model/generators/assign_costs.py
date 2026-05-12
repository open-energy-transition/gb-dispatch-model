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
from scripts.gb_model._helpers import get_scenario_name

logger = logging.getLogger(__name__)

COST_NAME_MAPPING = {"Fuel Cost": "fuel", "Variable Other Work Costs": "VOM"}
DEFAULT_SETS = {"PP", "Store"}


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
    costs = pd.read_csv(tech_costs_path, index_col=["technology", "parameter"])

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
    costs = costs[costs_config["pypsa_eur_tech_data_columns"]]

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

    if not (
        fes_power_costs_scenario := fes_power_costs.query(
            "Scenario == @scenario", local_dict={"scenario": fes_scenario}
        )
    ).empty:
        fes_power_costs = fes_power_costs_scenario

    # If we don't have a scenario match in the FES cost data (as is the case for any FES years >=2024),
    # We take the mean of all scenarios
    # Only battery storage is affected by this (having up to ~10% difference in VOM costs in different scenarios),
    fes_power_costs_mean = (
        fes_power_costs.groupby(["Type", "Sub Type", "Cost Type", "year"])
        .data.mean()
        .reset_index()
    )

    fes_power_costs_pivoted = (
        fes_power_costs_mean.assign(
            technology=fes_power_costs_mean["Type"]
            + "-"
            + fes_power_costs_mean["Sub Type"]
        )
        .pivot_table(
            index=["technology", "year"],
            columns="Cost Type",
            values="data",
        )
        .rename(columns=COST_NAME_MAPPING)
    )
    return fes_power_costs_pivoted[COST_NAME_MAPPING.values()]


def _load_fes_carbon_costs(
    fes_carbon_costs_path: str,
    fes_scenario: str,
) -> pd.Series:
    """
    Load FES carbon costs data.

    Args:
        fes_carbon_costs_path: Path to FES carbon costs CSV
        fes_scenario: FES scenario name (e.g., "leading the way")

    Returns:
        Series with year index and carbon_cost column (£/tCO2)

    Steps:
        1. Load FES carbon costs CSV
        2. Filter by scenario
        3. Select year and data columns, set year as index
        4. Rename data column to carbon_cost
    """
    # Load FES carbon costs
    fes_carbon_costs = pd.read_csv(fes_carbon_costs_path)

    if not (
        fes_carbon_costs_scenario := fes_carbon_costs.query(
            "Scenario == @scenario", local_dict={"scenario": fes_scenario}
        )
    ).empty:
        fes_carbon_costs = fes_carbon_costs_scenario
    # If we don't have a scenario match in the FES cost data (as is the case for any FES years >=2024),
    # We take the mean of all scenarios
    fes_carbon_costs_mean = fes_carbon_costs.groupby("year").data.mean()
    return fes_carbon_costs_mean.rename("carbon_cost")


def _integrate_fes_power_costs(
    df: pd.DataFrame,
    fes_power_costs: pd.DataFrame,
    costs_config: dict[str, dict],
) -> pd.DataFrame:
    """
    Integrate FES power costs into the powerplants DataFrame.

    Args:
        df (pd.DataFrame): Powerplants DataFrame with 'carrier', 'set', and 'year' columns.
        fes_power_costs (pd.DataFrame): FES power costs DataFrame with multi-index
            (Sub Type, year) and columns for each Cost Type (fuel, VOM).
        costs_config (dict): Configuration dict containing:
            - fes_costs_carrier_set_mapping: Mapping from carrier names to FES Sub Type name.

    Returns:
        pd.DataFrame: Updated powerplants DataFrame with integrated FES power costs.
    """
    names = df["name"]

    for col in ["VOM", "fuel"]:
        techs = names.map(costs_config[f"fes_{col}_carrier_set_mapping"])

        assert not (
            missing := set(techs.dropna()).difference(
                fes_power_costs.index.get_level_values("technology")
            )
        ), (
            f"Some mapped FES technologies for {col} costs are missing in FES power costs data: {missing}"
        )

        df[col] = fes_power_costs[[col]].reindex([techs, df.year]).values

    return df


def _calculate_marginal_costs(
    df: pd.DataFrame,
    costs: pd.DataFrame,
    costs_config: dict,
    fes_carbon_costs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Function to calculate marginal costs

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to finalize
    costs: pd.DataFrame
        Technology costs dataframe
    costs_config: dict
        config dictionary to map technology names and fill default characteristics
    fes_carbon_costs: pd.DataFrame
        Carbon costs indexed by year from FES data

    """
    gap_filling_mapping = costs_config["pypsa_eur_tech_data_carrier_set_mapping"]
    co2_intensity_mapping = costs_config["carrier_fossil_fuel_type"]
    if diff := set(co2_intensity_mapping.values()).difference(costs.index):
        msg = f"Found fossil fuel types not given in PyPSA-Eur technology data table: {diff}"
        raise ValueError(msg)
    if diff := set(gap_filling_mapping.values()).difference(costs.index):
        msg = f"Found carrier set gap filling technologies not given in PyPSA-Eur technology data table: {diff}"
        raise ValueError(msg)
    for param in costs_config["pypsa_eur_tech_data_columns"]:
        if param == "CO2 intensity":
            col, mapper = "carrier", co2_intensity_mapping
        else:
            col, mapper = "name", gap_filling_mapping
        df[param] = (
            df[col]
            .map(mapper)
            .map(costs[param])
            .fillna(costs_config["pypsa_eur_tech_data_defaults"][param])
        )

    # Calculate marginal cost if possible
    # CCS is expected to not be subject to a carbon tax on its fossil fuel intake.
    carbon_tax = (
        df["CO2 intensity"]
        .mul(fes_carbon_costs.reindex(df["year"]).values)
        .where(df["set"] != "CCS")
        .fillna(0)
    )
    df["marginal_cost"] = (
        df["VOM"]
        .fillna(0)
        .add(df["fuel"].add(carbon_tax).fillna(0))
        .div(df["efficiency"])
        .fillna(0)
    )

    return df


def assign_technical_and_costs_defaults(
    ppl_path: str,
    tech_costs_path: str,
    fes_power_costs_path: str,
    fes_carbon_costs_path: str,
    costs_config: dict[str, dict],
    fes_scenario: str,
    data_file: str,
    max_hours_path: str,
    gb_config: dict[str, dict],
) -> pd.DataFrame:
    """
    Enrich powerplants dataframe with cost and technical parameters.

    Args:
        ppl_path: Path to powerplant data CSV file
        tech_costs_path: Path to technology costs CSV file
        fes_power_costs_path: Path to FES power costs CSV file
        fes_carbon_costs_path: Path to FES carbon costs CSV file
        costs_config: Configuration dict containing mappings and conversion rates
        fes_scenario: FES scenario name (e.g., "leading the way")
        data_file: Data file identifier
        max_hours_path: Path to max hours CSV file
        gb_config: Configuration dict of GB carrier mapping

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
        9. Integrate max hours for storage technologies
    """
    # Load powerplant data
    df = pd.read_csv(ppl_path).assign(**costs_config["add_cols"][data_file])

    # Load costs data
    costs = _load_costs(tech_costs_path, costs_config)
    fes_power_costs = _load_fes_power_costs(fes_power_costs_path, fes_scenario)
    fes_carbon_costs = _load_fes_carbon_costs(fes_carbon_costs_path, fes_scenario)
    logger.info("Loaded technology costs and FES power and carbon costs data")

    # Join cost data
    add_set = (
        ("-" + df["set"].fillna("")).where(~df["set"].isin(DEFAULT_SETS)).fillna("")
    )
    df["name"] = df["carrier"] + add_set

    # Integrate FES power costs
    df = _integrate_fes_power_costs(df, fes_power_costs, costs_config)

    # Calculate marginal costs
    df = _calculate_marginal_costs(df, costs, costs_config, fes_carbon_costs)

    # Format bus, build_year, and name columns
    df["bus"] = df["bus"].astype(str)
    df["build_year"] = df["year"].astype(int)
    df["name"] = df["bus"] + " " + df["name"] + "-" + df["build_year"].astype(str)

    # Add country columns
    df["country"] = df["bus"].str[:2]

    # Integrate max_hours
    df_max_hours = pd.read_csv(max_hours_path)
    max_hours = (
        df_max_hours.groupby(["carrier", "year"])
        .max_hours.mean()
        .reindex(df[["carrier", "year"]].values)
    )
    if max_hours.notnull().any():
        df["max_hours"] = max_hours.values

    # PyPSA-Eur expects 'overnight_cost' column for the same meaning as given in the source data under "investment"
    df = df.rename(columns={"investment": "overnight_cost"}, errors="ignore")
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
    costs_config = snakemake.params.costs_config
    fes_scenario = get_scenario_name(snakemake)

    # Enrich powerplants with technical/cost parameters
    df_powerplants = assign_technical_and_costs_defaults(
        ppl_path=ppl_path,
        tech_costs_path=tech_costs_path,
        fes_power_costs_path=fes_power_costs_path,
        fes_carbon_costs_path=fes_carbon_costs_path,
        costs_config=costs_config,
        fes_scenario=fes_scenario,
        data_file=snakemake.wildcards.data_file,
        max_hours_path=snakemake.input.max_hours,
        gb_config=snakemake.params.gb_config,
    )
    logger.info("Enriched powerplants with cost and technical parameters")

    # Save with index (contains unique generator IDs)
    df_powerplants.to_csv(snakemake.output.enriched_powerplants, index=False)

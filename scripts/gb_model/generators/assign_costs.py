# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
Costs assigner.

This script enriches powerplants CSV data with costs information.

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

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_scenario_name

logger = logging.getLogger(__name__)

COST_NAME_MAPPING = {"Fuel Cost": "fuel", "Variable Other Work Costs": "VOM"}
DEFAULT_SETS = {"PP", "Store"}


def load_costs(
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


def load_fes_power_costs(
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


def load_fes_carbon_costs(
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
    cost_cols = ["VOM", "fuel"]
    _df = df.copy()
    for col in cost_cols:
        _df["technology"] = _df["name"].map(
            costs_config[f"fes_{col}_carrier_set_mapping"]
        )

        assert not (
            missing := set(_df["technology"].dropna()).difference(
                fes_power_costs.index.get_level_values("technology")
            )
        ), (
            f"Some mapped FES technologies for {col} costs are missing in FES power costs data: {missing}"
        )
        _df = _df.merge(fes_power_costs[[col]].reset_index(), how="left")

    return _df.drop(columns=["technology"])


def _map_tech_data(
    df: pd.DataFrame,
    costs: pd.DataFrame,
    costs_config: dict,
) -> pd.DataFrame:
    """
    Map technology data from costs DataFrame to the powerplants DataFrame based on carrier and set mappings.

    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to finalize
    costs: pd.DataFrame
        Technology costs dataframe
    costs_config: dict
        config dictionary to map technology names and fill default characteristics

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
    return df


def _calculate_marginal_costs(
    df: pd.DataFrame, fes_carbon_costs: pd.DataFrame
) -> pd.Series:
    """Calculate marginal cost from VOM, fuel, efficiency, CO2 intensity, and carbon_cost."""

    # CCS is expected to not be subject to a carbon tax on its fossil fuel intake.
    carbon_tax = (
        df["CO2 intensity"]
        .mul(fes_carbon_costs.reindex(df["year"]).values)
        .where(df["set"] != "CCS")
        .fillna(0)
    )
    return (
        df["VOM"]
        .fillna(0)
        .add(df["fuel"].add(carbon_tax).fillna(0))
        .div(df["efficiency"])
        .fillna(0)
    )


def _postprocess_format_cols(df):
    """Post-process formatting of columns after all cost and technical data has been integrated."""
    # Format bus, build_year, and name columns
    df["bus"] = df["bus"].astype(str)
    df["build_year"] = df["year"].astype(int)
    df["name"] = df["bus"] + " " + df["name"] + "-" + df["build_year"].astype(str)

    # Add country columns
    df["country"] = df["bus"].str[:2]
    # PyPSA-Eur expects 'overnight_cost' column for the same meaning as given in the source data under "investment"
    df = df.rename(columns={"investment": "overnight_cost"}, errors="ignore")
    return df


def add_max_hours(df, max_hours_path):
    """Integrate max hours for storage technologies based on FES ES1 sheet data."""
    # Integrate max_hours
    df_max_hours = pd.read_csv(max_hours_path)
    max_hours = (
        df_max_hours.groupby(["carrier", "year"])
        .max_hours.mean()
        .reindex(df[["carrier", "year"]].values)
    )
    if max_hours.notnull().any():
        df["max_hours"] = max_hours.values


def assign_technical_and_costs_defaults(
    df: pd.DataFrame,
    fes_power_costs: pd.DataFrame,
    costs: pd.DataFrame,
    fes_carbon_costs: pd.DataFrame,
    costs_config: dict[str, dict],
) -> pd.DataFrame:
    """
    Enrich powerplants dataframe with cost and technical parameters.

    Parameters
    ----------
    df: pd.DataFrame
        Powerplants dataframe with at least 'carrier', 'set', and 'year' columns
    fes_power_costs: pd.DataFrame
        FES power costs with multi-index (technology, year) and columns for 'fuel' and 'VOM'
    costs: pd.DataFrame
        Technology costs dataframe indexed by technology with columns for efficiency, CO2 intensity, capital cost, and lifetime
    fes_carbon_costs: pd.DataFrame
        Carbon costs indexed by year from FES data

    Returns
    -------
    pd.DataFrame
        Enriched powerplants DataFrame with efficiency, marginal_cost, VOM, fuel,
        CO2 intensity, capital_cost, lifetime, build_year, and unique index

    """
    add_set = (
        ("-" + df["set"].fillna(""))
        .where(~df["set"].isin(DEFAULT_SETS) & df["set"].notnull())
        .fillna("")
    )
    df["name"] = df["carrier"] + add_set

    # Integrate FES power costs
    df = _integrate_fes_power_costs(df, fes_power_costs, costs_config)
    df = _map_tech_data(df, costs, costs_config)
    df["marginal_cost"] = _calculate_marginal_costs(df, fes_carbon_costs)
    df = _postprocess_format_cols(df)
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

    # Load powerplant data
    df = pd.read_csv(ppl_path).assign(
        **costs_config["add_cols"][snakemake.wildcards.data_file]
    )

    # Load costs data
    costs = load_costs(tech_costs_path, costs_config)
    fes_power_costs = load_fes_power_costs(fes_power_costs_path, fes_scenario)
    fes_carbon_costs = load_fes_carbon_costs(fes_carbon_costs_path, fes_scenario)

    # Enrich powerplants with technical/cost parameters
    df_powerplants = assign_technical_and_costs_defaults(
        df=df,
        fes_power_costs=fes_power_costs,
        costs=costs,
        fes_carbon_costs=fes_carbon_costs,
        costs_config=costs_config,
    )
    add_max_hours(df_powerplants, snakemake.input.max_hours)
    logger.info("Enriched powerplants with cost and technical parameters")

    # Save with index (contains unique generator IDs)
    df_powerplants.to_csv(snakemake.output.enriched_powerplants, index=False)

# SPDX-FileCopyrightText:  gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT


"""
Capacity table generator.

This is a script to GB/Eur capacities defined for / by the FES to fix `p_nom` in PyPSA-Eur.
"""

import logging

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _map_names(
    df: pd.DataFrame, mapping: dict[str, dict[str, str]], default: str | None = None
) -> str | None:
    """Map carriers/sets to a standard name."""
    mapped = pd.Series(default, index=df.index, dtype="object")
    for col, mappings in mapping.items():
        mapped = mapped.fillna(df[col].map(mappings))
    return mapped


def capacity_table(
    df: pd.DataFrame,
    mapping_config: dict,
    default_set: str,
) -> pd.DataFrame:
    """
    Format the capacity table in a format required by PyPSA-Eur

    Args:
        df (pd.DataFrame): powerplant data table
        mapping_config (dict): dictionary to map technologies to PyPSA-Eur carriers names
        default_set (str): default set to use if no mapping is found
    """
    df_cleaned = df.where(df.data > 0).dropna(subset=["data"])
    df_cleaned["carrier"] = _map_names(df_cleaned, mapping_config["carrier_mapping"])
    df_cleaned["set"] = _map_names(
        df_cleaned, mapping_config["set_mapping"], default_set
    )

    if any(missing := df_cleaned["carrier"].isnull()):
        cols = list(mapping_config["carrier_mapping"])
        missing_names = df_cleaned[missing][cols].drop_duplicates()
        logger.warning(
            f"Some technologies could not be mapped to a carrier: {missing_names}"
        )

    df_cleaned = df_cleaned.dropna(subset=["carrier"])

    df_capacity = (
        df_cleaned.groupby(["bus", "year", "carrier", "set"])["data"]
        .sum()
        .rename("p_nom")
        .reset_index()
    )

    return df_capacity


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


def assign_technical_and_costs_defaults(
    df: pd.DataFrame,
    costs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich powerplants dataframe with cost and technical parameters.

    Adds efficiency, marginal_cost, capital_cost, lifetime, build_year, and unique index.

    Args:
        df (pd.DataFrame): powerplant data with bus, year, carrier, set, p_nom columns
        costs (pd.DataFrame): cost data indexed by carrier

    Returns:
        pd.DataFrame: enriched powerplants data ready for PyPSA
    """
    df["bus"] = df["bus"].astype(str)
    df["build_year"] = df["year"].astype(int)

    # Join cost data if available
    cost_columns = ["VOM", "FOM", "efficiency", "capital_cost", "fuel", "lifetime"]
    available_cost_cols = [col for col in cost_columns if col in costs.columns]

    if available_cost_cols:
        df = df.join(costs[available_cost_cols], on="carrier")
    else:
        logger.warning("No cost columns found in costs dataframe")

    # Calculate marginal cost if possible
    if all(col in df.columns for col in ["VOM", "fuel", "efficiency"]):
        df["marginal_cost"] = (
            df["carrier"].map(costs["VOM"])
            + df["carrier"].map(costs["fuel"]) / df["efficiency"]
        )

    # Ensure all required columns exist with defaults
    defaults = {
        "efficiency": (0.4, ""),
        "capital_cost": (0.0, ""),
        "lifetime": (25.0, "years"),
        "marginal_cost": (0.0, ""),
    }

    for col, (default, units) in defaults.items():
        df = _ensure_column_with_default(df, col, default, units)

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
    df = df.drop(columns=["idx_counter"])

    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("create_powerplants_table")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    df_gsp = (
        pd.read_csv(snakemake.input.gsp_data)
        .query("Template == 'Generation'")
        .dropna(subset=["Latitude", "Longitude"])
    )
    df_eur = pd.read_csv(snakemake.input.eur_data).query("Variable == 'Capacity (MW)'")

    # Load all the params
    gb_config = snakemake.params.gb_config
    eur_config = snakemake.params.eur_config
    default_set = snakemake.params.default_set

    df_capacity_gb = capacity_table(df_gsp, gb_config, default_set)
    logger.info("Tabulated the capacities into a table in PyPSA-Eur format")

    df_capacity_eur = capacity_table(df_eur, eur_config, default_set)
    logger.info("Added the EU wide capacities to the capacity table")

    df_capacity = pd.concat([df_capacity_gb, df_capacity_eur], ignore_index=True)

    # Load costs data and enrich powerplants with technical/cost parameters
    costs = pd.read_csv(snakemake.input.tech_costs, index_col=0)
    logger.info("Loaded technology costs data")

    df_powerplants = assign_technical_and_costs_defaults(df_capacity, costs)
    logger.info("Enriched powerplants with cost and technical parameters")

    # Save with index (contains unique generator IDs)
    df_powerplants.to_csv(snakemake.output.csv, index=True)

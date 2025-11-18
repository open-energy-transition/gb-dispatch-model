# SPDX-FileCopyrightText: gb-dispatch-model contributors
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
    mapped = pd.Series(float("nan"), index=df.index, dtype="object")
    for col, mappings in mapping.items():
        mapped = mapped.fillna(df[col].map(mappings))
    if default is not None:
        mapped = mapped.fillna(default)
    return mapped


def capacity_table(
    df: pd.DataFrame,
    mapping_config: dict,
    default_set: str,
    geographic_level: str = "bus",
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

    df_cleaned_nona = df_cleaned.dropna(subset=["carrier"])

    df_capacity = (
        df_cleaned_nona.groupby([geographic_level, "year", "carrier", "set"])["data"]
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
    tech_costs_path: str,
    fes_power_costs_path: str,
    fes_carbon_costs_path: str,
    default_characteristics: dict[str, dict],
    costs_config: dict[str, dict],
) -> pd.DataFrame:
    """
    Enrich powerplants dataframe with cost and technical parameters.

    Adds efficiency, marginal_cost, capital_cost, lifetime, build_year, and unique index.

    Args:
        df (pd.DataFrame): powerplant data with bus, year, carrier, set, p_nom columns.
        costs (pd.DataFrame): cost data indexed by carrier.
        default_characteristics (dict): default values for technical and cost parameters

    Returns:
        pd.DataFrame: enriched powerplants data ready for PyPSA
    """
    # Load costs data
    costs = load_costs(tech_costs_path, costs_config)
    # fes_power_costs = pd.read_csv(fes_power_costs_path)
    # fes_carbon_costs = pd.read_csv(fes_carbon_costs_path)
    logger.info("Loaded technology costs and FES power and carbon costs data")

    # Join cost data
    df = df.join(costs[costs_config["marginal_cost_columns"]], on="carrier")

    # Fill CO2 intensities and fuel costs using carrier_fuel_mapping
    df["CO2 intensity"] = (
        df["carrier"]
        .map(costs_config["carrier_fuel_mapping"])
        .map(costs["CO2 intensity"])
    )
    df["fuel"] = (
        df["carrier"].map(costs_config["carrier_fuel_mapping"]).map(costs["fuel"])
    )

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

    for name, config in default_characteristics.items():
        df = _ensure_column_with_default(df, name, config["data"], config["unit"])

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


def _remaining_to_distribute(
    df_TO: pd.DataFrame, df_dist: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate the remaining capacity to distribute.

    Args:
        df_TO (pd.DataFrame): The total capacity dataframe.
        df_dist (pd.DataFrame): The distributed capacity dataframe.

    Returns:
        pd.DataFrame: The remaining capacity to distribute.
    """
    return df_TO[
        df_TO.subtract(df_dist.groupby(df_TO.index.names).sum(), fill_value=0).abs()
        > 1e-2
    ].dropna()


def _create_relative_table(
    df: pd.DataFrame, bus_to_TO: pd.Series, cols: list[str]
) -> pd.DataFrame:
    """
    Create a table of relative capacity per bus within each TO region.

    Args:
        df (pd.DataFrame): DataFrame with capacity data.
        bus_to_TO (pd.Series): Series mapping buses to TO regions.
        cols (list[str]): List of non-geographical / non-data columns to keep.

    Returns:
        pd.DataFrame: DataFrame with relative capacity data.
    """
    df_rel = (
        df.groupby(["bus", *cols])[["p_nom"]]
        .sum()
        .merge(bus_to_TO, left_index=True, right_index=True)
        .set_index("TO_region", append=True)
        .groupby(["TO_region", *cols], group_keys=False)
        .apply(lambda x: x / x.sum())
    )
    return df_rel


def distribute_direct_data(
    df_TO: pd.DataFrame,
    df_gsp: pd.DataFrame,
    df_dukes: pd.DataFrame,
    df_gb_expected: pd.DataFrame,
    bus_to_TO: pd.Series,
) -> pd.DataFrame:
    """
    Distribute non-GSP GB capacity data to GSPs based on available data.

    The hierarchy for data used for distribution is:
    1. GSP-level data for the same carriers from FES (i.e. future capacity)
    2. GSP-level data for the same carriers from DUKES (i.e. existing capacity)
    3. GSP-level data for _all_ carriers from FES+DUKES

    Lastly, if there is still remaining capacity to distribute
    (i.e., after the above is applied, there is still a gap compared to total GB capacity),
    it is distributed according to the overall distribution of capacity for each carrier across GSPs.

    Args:
        df_TO (pd.DataFrame): DataFrame with TO-level capacity data.
        df_gsp (pd.DataFrame): DataFrame with GSP-level capacity data from FES.
        df_dukes (pd.DataFrame): DataFrame with GSP-level capacity data from DUKES.
        df_gb_expected (pd.DataFrame): DataFrame with expected total GB capacity data.
        bus_to_TO (pd.Series): Series mapping buses to TO regions.

    Returns:
        pd.DataFrame: DataFrame with distributed GSP-level capacity data.
    """
    df_gsp_with_TO = _create_relative_table(
        df_gsp, bus_to_TO, ["year", "carrier", "set"]
    )
    gsp_dist = df_TO.multiply(df_gsp_with_TO)
    df_TO_remaining = _remaining_to_distribute(df_TO, gsp_dist)
    df_dukes_with_TO = _create_relative_table(df_dukes, bus_to_TO, ["carrier", "set"])
    dukes_dist = df_TO_remaining.multiply(df_dukes_with_TO)

    all_dist = pd.concat([gsp_dist.dropna(), dukes_dist.dropna()])
    df_TO_remaining = _remaining_to_distribute(df_TO, all_dist)

    if not df_TO_remaining.empty:
        logger.warning(
            f"Could not fully distribute TO-level data, remaining:\n{df_TO_remaining}."
            "\nDistributing using TO-level aggregate powerplant distributions."
        )
        last_dist = (
            # Get the relative distribution in each bus for the sum of all carrier capacities
            _create_relative_table(all_dist, bus_to_TO, ["set", "year"])
            .multiply(df_TO_remaining)
            .dropna()
        )
        all_dist = pd.concat(
            [all_dist, last_dist.reorder_levels(all_dist.index.names)]
        ).droplevel("TO_region")
    df_gsp_and_TO = df_gsp.set_index(all_dist.index.names).add(all_dist, fill_value=0)

    df_gsp_and_TO_relative = (
        df_gsp_and_TO.groupby(["carrier", "set", "year"], group_keys=False)
        .apply(lambda x: x / x.sum())
        .p_nom
    )
    df_capacity_gb_final = df_gb_expected.multiply(df_gsp_and_TO_relative).dropna()
    if (diff := df_capacity_gb_final.sum() - df_gb_expected.sum()) > 0:
        logger.error(
            f"""
            Final distributed GB capacity does not match total FES capacity after distribution
            ({diff / df_gb_expected.sum() * 100:.2f}% difference.)
            """
        )
    return df_capacity_gb_final.to_frame("p_nom").reset_index()


def load_costs(
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


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("create_powerplants_table")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    df_gsp = pd.read_csv(snakemake.input.gsp_data).query(
        "Template in ['Generation', 'Storage & Flexibility']"
    )
    df_dukes = pd.read_csv(snakemake.input.dukes_data)

    df_eur = pd.read_csv(snakemake.input.eur_data).query("Variable == 'Capacity (MW)'")

    # Load all the params
    gb_config = snakemake.params.gb_config
    eur_config = snakemake.params.eur_config
    dukes_config = snakemake.params.dukes_config
    default_set = snakemake.params.default_set
    costs_config = snakemake.params.costs_config

    df_capacity_gb_gsp = capacity_table(
        df_gsp[df_gsp.bus.notnull()], gb_config, default_set
    )
    logger.info("Tabulated the capacities into a table in PyPSA-Eur format")

    df_capacity_gb_TO = capacity_table(
        df_gsp[df_gsp.bus.isnull()], gb_config, default_set, "TO_region"
    ).set_index(["carrier", "set", "TO_region", "year"])

    df_capacity_gb_dukes = capacity_table(df_dukes, dukes_config, default_set)
    bus_to_TO = df_dukes.groupby("bus").TO_region.first()

    df_capacity_gb_expected = (
        capacity_table(df_gsp, gb_config, default_set, "Unit")
        .set_index(["carrier", "set", "year"])
        .p_nom
    )
    df_capacity_gb = distribute_direct_data(
        df_capacity_gb_TO,
        df_capacity_gb_gsp,
        df_capacity_gb_dukes,
        df_capacity_gb_expected,
        bus_to_TO,
    )

    df_capacity_eur = capacity_table(df_eur, eur_config, default_set)
    logger.info("Added the EU wide capacities to the capacity table")

    df_capacity = pd.concat([df_capacity_gb, df_capacity_eur], ignore_index=True)

    # Enrich powerplants with technical/cost parameters
    df_powerplants = assign_technical_and_costs_defaults(
        df_capacity,
        tech_costs_path=snakemake.input.tech_costs,
        fes_power_costs_path=snakemake.input.fes_power_costs,
        fes_carbon_costs_path=snakemake.input.fes_carbon_costs,
        default_characteristics=snakemake.params.default_characteristics,
        costs_config=costs_config,
    )
    logger.info("Enriched powerplants with cost and technical parameters")

    # Save with index (contains unique generator IDs)
    df_powerplants.to_csv(snakemake.output.csv, index=True)

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

    df_cleaned_nona = df_cleaned.dropna(subset=["carrier"])

    df_capacity = (
        df_cleaned_nona.groupby(["bus", "year", "carrier", "set"])["data"]
        .sum()
        .rename("p_nom")
        .reset_index()
    )

    return df_capacity


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("create_powerplants_table")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    df_gsp = (
        pd.read_csv(snakemake.input.gsp_data)
        .query("Template in ['Generation', 'Storage & Flexibility']")
        .dropna(subset=["Latitude", "Longitude"])
    )
    # Simply add any distributed TO_data to the GSP data for capacity calculations
    # FIXME: this will place some capacity into the wrong regions (e.g. pumped hydro).
    # We should ideally distribute TO-level capacity based on existing UK capacities from DUKES.
    df_gsp["data"] = df_gsp["data"].fillna(0) + df_gsp["TO_data"].fillna(0)

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

    df_capacity.to_csv(snakemake.output.csv, index=False)

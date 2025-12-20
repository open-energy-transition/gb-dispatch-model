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

    # Save with index (contains unique generator IDs)
    df_capacity.to_csv(snakemake.output.csv, index=False)

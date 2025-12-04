# SPDX-FileCopyrightText:  gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT


"""
GSP-level data table generator.

This is a script to combine the BB1 sheet with the BB2 (metadata) sheet of the FES workbook.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import map_points_to_regions

logger = logging.getLogger(__name__)


def process_dukes_data(
    df_dukes: pd.DataFrame, gdf_regions: gpd.GeoDataFrame, target_crs: str
) -> pd.DataFrame:
    region_data = map_points_to_regions(
        df_dukes.dropna(subset=["Y-Coordinate", "X-Coordinate"]),
        gdf_regions,
        "Y-Coordinate",
        "X-Coordinate",
        "EPSG:27700",
        target_crs,
    )
    df_dukes_with_regions = df_dukes.merge(
        region_data[["name", "TO_region"]].rename(columns={"name": "bus"}),
        left_index=True,
        right_index=True,
    )
    initial_drop = df_dukes["X-Coordinate"].isnull().sum()
    bus_drop = df_dukes_with_regions["bus"].isnull().sum()

    logger.info(
        f"DUKES capacity table | Dropping {initial_drop} rows with missing coordinate data plus {bus_drop} rows that could not be mapped to a bus."
    )

    return df_dukes_with_regions.dropna(subset=["bus"])


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    gdf_regions = gpd.read_file(snakemake.input.regions)
    gdf_regions_offshore = gpd.read_file(snakemake.input.regions_offshore)
    gb_regions_offshore = gdf_regions_offshore[
        gdf_regions_offshore.name.str.startswith("GB ")
    ].assign(country="GB")
    gdf_regions_all = (
        pd.concat([gdf_regions, gb_regions_offshore])
        .query("country == 'GB'")
        .dissolve("name")
    )

    sheet_config = snakemake.params.sheet_config
    sheet_name = sheet_config.pop("sheet_name")
    df_dukes = pd.read_excel(
        snakemake.input.dukes_data, sheet_name=sheet_name, **sheet_config
    )
    df_dukes_cleaned = process_dukes_data(
        df_dukes, gdf_regions_all, snakemake.params.target_crs
    )
    df_dukes_cleaned = df_dukes_cleaned.rename(
        columns={"InstalledCapacity (MW)": "data", "Year Commissioned": "year"}
    )
    df_dukes_cleaned.to_csv(snakemake.output.csv, index=False)

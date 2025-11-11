# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
Interconnector capacity table generator.

This is a script to set the interconnector capacity per GB region for each scenario year.
"""

import logging
from pathlib import Path

import country_converter as coco
import geopandas as gpd
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import map_points_to_regions

logger = logging.getLogger(__name__)


def projects_to_pypsa_links(
    interconnector_config: dict, gdf_regions: gpd.GeoDataFrame
) -> pd.DataFrame:
    """
    Map interconnector projects to links in our PyPSA network

    Args:
        interconnector_config (dict): Configuration dictionary defining interconnector projects
        gdf_regions (gpd.GeoDataFrame): GeoDataFrame containing network bus geometries

    Returns:
        pd.DataFrame: DataFrame mapping interconnector projects to PyPSA links
    """
    df_cols = ["name", "neighbour", "capacity_mw", "lat", "lon"]
    df = (
        pd.concat(
            [
                pd.DataFrame(
                    {k: [v] for k, v in interconnector.items() if k in df_cols}
                )
                for interconnector in interconnector_config["options"]
            ]
        )
        .set_index("name")
        .rename_axis(index="project")
    )

    df["bus0"] = map_points_to_regions(
        df, gdf_regions, "lat", "lon", "EPSG:4326", snakemake.params.target_crs
    )["name"]
    country_codes = {x: coco.convert(x, to="ISO2") for x in df["neighbour"].unique()}
    df["bus1"] = df["neighbour"].replace(country_codes)

    df_capacity = pd.DataFrame(
        {
            year: df.loc[projects].groupby(["bus0", "bus1"]).capacity_mw.sum()
            for year, projects in interconnector_config["plan"].items()
        }
    ).T.rename_axis(index="year")
    all_years = list(range(df_capacity.index.min(), snakemake.params.year_range[1] + 1))
    years_to_keep = list(
        range(snakemake.params.year_range[0], snakemake.params.year_range[1] + 1)
    )

    df_capacity_all_years = df_capacity.reindex(all_years).cumsum().ffill().fillna(0)
    df_capacity_all_years = df_capacity_all_years.loc[years_to_keep]

    logger.info(
        f"Total Interconnector capacity (MW): {df_capacity_all_years.sum(axis=1)}"
    )
    df_capacity_all_years = (
        df_capacity_all_years.unstack()
        .rename("p_nom")
        .reset_index()
        .assign(carrier="DC")
    )
    return df_capacity_all_years


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    interconnector_config = snakemake.params.interconnector_config
    gdf_regions = gpd.read_file(snakemake.input.regions)
    df_capacity = projects_to_pypsa_links(interconnector_config, gdf_regions)
    df_capacity.to_csv(snakemake.output.gsp_data, index=False)

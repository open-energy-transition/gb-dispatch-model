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


def synthesise_line_geometries(
    df: pd.DataFrame, gdf_regions: gpd.GeoDataFrame, projected_crs: str
) -> gpd.GeoSeries:
    """
    Get shortest line between GB connection point and neighbouring country and define this as the interconnector.

    Args:
        df (pd.DataFrame): interconnector projects data
        gdf_regions (gpd.GeoDataFrame): GeoDataFrame containing network bus geometries
        projected_crs (str): Coordinate reference system to project geometries

    Returns:
        gpd.GeoSeries: GeoSeries of line geometries representing interconnectors
    """
    df_filtered = df[df.bus1.isin(gdf_regions["name"])]
    points = gpd.GeoSeries(
        gpd.points_from_xy(df_filtered.lon, df_filtered.lat), crs="EPSG:4326"
    ).to_crs(projected_crs)
    neighbours = (
        gdf_regions.set_index("name")
        .loc[df_filtered.bus1, "geometry"]
        .to_crs(projected_crs)
        .reset_index(drop=True)
    )

    lines = points.shortest_line(neighbours)
    lines.index = df_filtered.index
    return lines


def projects_to_pypsa_links(
    interconnector_options: list[dict],
    interconnector_plan: dict[int, list[str]],
    gdf_regions: gpd.GeoDataFrame,
    year_range: list[int],
    target_crs: str,
) -> pd.DataFrame:
    """
    Map interconnector projects to links in our PyPSA network

    Args:
        interconnector_options (list[dict]): List of dictionaries defining interconnector projects
        interconnector_plan (dict[int, list[str]]): Dictionary mapping years to list of interconnector project names
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
                for interconnector in interconnector_options
            ]
        )
        .set_index("name")
        .rename_axis(index="project")
    )

    df["bus0"] = map_points_to_regions(
        df, gdf_regions, "lat", "lon", "EPSG:4326", target_crs
    )["name"]
    country_codes = {x: coco.convert(x, to="ISO2") for x in df["neighbour"].unique()}
    df["bus1"] = df["neighbour"].replace(country_codes)

    df_capacity = pd.DataFrame(
        {
            year: df.loc[projects]
            .reset_index()
            .groupby(["project", "bus0", "bus1"])
            .capacity_mw.sum()
            for year, projects in interconnector_plan.items()
        }
    ).T.rename_axis(index="year")
    all_years = list(range(df_capacity.index.min(), year_range[1] + 1))
    years_to_keep = list(range(year_range[0], year_range[1] + 1))

    df_capacity_all_years = df_capacity.reindex(all_years).cumsum().ffill().fillna(0)
    df_capacity_all_years = df_capacity_all_years.loc[years_to_keep]

    logger.info(
        f"Total Interconnector capacity (MW): {df_capacity_all_years.sum(axis=1)}"
    )
    df_capacity_all_years = (
        df_capacity_all_years.unstack()
        .rename("p_nom")
        .reset_index()
        # p_min_pu=-1 to allow for bidirectional flow
        .assign(
            carrier="DC", underwater_fraction=0.9, underground=True, p_min_pu=-1, dc=1.0
        )
    )

    # filter out links to countries not included in the model regions

    if not_in := set(df_capacity_all_years.bus1).difference(gdf_regions["name"]):
        logger.info(
            "The following neighbouring countries are not in the model scope; "
            f"their interconnectors will be excluded: {not_in}"
        )
    df_filtered = df_capacity_all_years[
        df_capacity_all_years.bus1.isin(gdf_regions["name"])
    ]
    # Add extra columns of data
    lines_geoms = synthesise_line_geometries(df, gdf_regions, target_crs)
    df_with_line_data = df_filtered.merge(
        lines_geoms.to_frame("geometry").assign(
            length=lines_geoms.length.div(1000).astype(int)
        ),
        left_on="project",
        right_index=True,
    )
    df_with_build_year = df_with_line_data.merge(
        df_capacity.unstack()
        .dropna()
        .reset_index("year")
        .year.rename("build_year")
        .reset_index("project"),
        on="project",
    )
    assert (
        df_with_build_year[["build_year", "length", "geometry"]].notnull().all().all()
    ), (
        "Some interconnector lines are missing required data (build year, line geometry, length)"
    )
    return df_with_build_year


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    gdf_regions = gpd.read_file(snakemake.input.regions)
    df_capacity = projects_to_pypsa_links(
        snakemake.params.interconnector_options,
        snakemake.params.interconnector_plan[snakemake.wildcards.fes_scenario],
        gdf_regions,
        snakemake.params.year_range,
        snakemake.params.target_crs,
    )
    df_capacity.to_csv(snakemake.output.gsp_data, index=False)

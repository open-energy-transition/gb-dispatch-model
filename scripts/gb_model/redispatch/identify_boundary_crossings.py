# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Identify the ETYS boundaries that each line/link in the network crosses.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def lines_boundaries(
    etys_gdf: gpd.GeoDataFrame, lines_gdf: gpd.GeoDataFrame, linemap: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify the ETYS boundaries that each line in the network crosses.

    Parameters
    ----------
    etys_gdf: gpd.GeoDataFrame
        GeoDataFrame containing the ETYS boundaries
    lines_gdf: gpd.GeoDataFrame
        GeoDataFrame containing the PyPSA base network line geometries
    linemap: pd.DataFrame
        DataFrame mapping line names in the base network to line names in the clustered network

    Returns
    -------
    pd.DataFrame
        DataFrame containing line IDs and the ETYS boundaries they cross
    """
    lines_gdf["line_id"] = lines_gdf.index.map(linemap)
    lines_gdf = lines_gdf.dropna(subset=["line_id"]).reset_index(drop=True)
    line_crossings = gpd.sjoin(lines_gdf, etys_gdf, how="left", predicate="intersects")
    line_crossings_cleaned = (
        line_crossings[["line_id", "Boundary_n"]]
        .dropna(subset=["Boundary_n"])
        .reset_index(drop=True)
    )
    line_crossings_cleaned["line_id"] = (
        line_crossings_cleaned["line_id"].astype(int).astype(str)
    )
    return line_crossings_cleaned


def links_boundaries(
    etys_gdf: gpd.GeoDataFrame, links_gdf: gpd.GeoDataFrame, n: pypsa.Network
) -> pd.DataFrame:
    """
    Identify the ETYS boundaries that each link in the network crosses.

    Parameters
    ----------
    etys_gdf: gpd.GeoDataFrame
        GeoDataFrame containing the ETYS boundaries
    links_gdf: gpd.GeoDataFrame
        GeoDataFrame containing the PyPSA base network link geometries
    n: pypsa.Network
        Clustered PyPSA network containing the final links

    Returns
    -------
    pd.DataFrame
        DataFrame containing link IDs and the ETYS boundaries they cross
    """
    selected_links = n.links[
        n.links.bus0.str.startswith("GB ") & n.links.bus1.str.startswith("GB ")
    ].index
    linkmap = pd.Series(
        index=selected_links,
        data=selected_links.str.replace(r"[\+]\d+", "", regex=True),
    )
    links_gdf = links_gdf[links_gdf.index.isin(linkmap.values)]

    links_gdf["link_id"] = linkmap.index
    link_crossings = gpd.sjoin(links_gdf, etys_gdf, how="left", predicate="intersects")
    link_crossings_cleaned = (
        link_crossings[["link_id", "Boundary_n"]]
        .dropna(subset=["Boundary_n"])
        .reset_index(drop=True)
    )

    return link_crossings_cleaned


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    etys_gdf = gpd.read_file(snakemake.input.etys_boundaries)
    n_base = pypsa.Network(snakemake.input.base_network)
    lines_gdf = (
        gpd.GeoSeries.from_wkt(n_base.lines.geometry, crs=4326)
        .to_frame()
        .to_crs(etys_gdf.crs)
    )
    linemap = pd.read_csv(snakemake.input.linemap, index_col=0).squeeze()

    line_boundary_df = lines_boundaries(etys_gdf, lines_gdf, linemap)

    links_gdf = (
        gpd.GeoSeries.from_wkt(n_base.links.geometry, crs=4326)
        .to_frame()
        .to_crs(etys_gdf.crs)
    )
    n_clustered = pypsa.Network(snakemake.input.clustered_network)

    link_boundary_df = links_boundaries(etys_gdf, links_gdf, n_clustered)
    all_crossings = (
        pd.concat(
            [
                line_boundary_df.rename(columns={"line_id": "name"}),
                link_boundary_df.rename(columns={"link_id": "name"}),
            ],
            keys=["Line", "Link"],
            names=["component"],
        )
        .reset_index("component")
        .drop_duplicates(subset=["component", "name", "Boundary_n"])
    )
    all_crossings.to_csv(snakemake.output.csv, index=False)

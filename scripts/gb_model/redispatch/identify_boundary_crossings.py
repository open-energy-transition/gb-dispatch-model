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
from shapely.ops import split

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def define_side_for_region(
    etys_gdf: gpd.GeoDataFrame,
    region_gdf: gpd.GeoDataFrame,
    buffer_mainland_m: float = 1000,
) -> pd.DataFrame:
    """
    Define a side for each model region relative to each ETYS boundary.

    Parameters
    ----------
    etys_gdf: gpd.GeoDataFrame
        GeoDataFrame containing the ETYS boundaries
    region_gdf: gpd.GeoDataFrame
        GeoDataFrame containing the model regions

    Returns
    -------
    pd.DataFrame
        DataFrame containing region IDs and the side of each ETYS boundary they are on (0 or 1).
        The side number is only meaningful in a relative sense for each boundary.
        That is, it indicates which side of the boundary the region is on, but not an absolute direction.
    """
    line_sides = {}
    # Create a buffer to get nearby islands into the same region as the mainland, using the unit of the region geometries.
    region_units = region_gdf.crs.axis_info[0].unit_name
    buffer = (
        buffer_mainland_m if region_units == "metre" else buffer_mainland_m / 100000
    )
    one_gb_gdf = (
        gpd.GeoSeries([region_gdf.buffer(buffer).union_all()])
        .explode()
        .reset_index(drop=True)
    )
    biggest_gb_geom = one_gb_gdf.loc[one_gb_gdf.area.idxmax()]
    region_points = region_gdf.set_index("name").representative_point().to_frame()
    for line in etys_gdf.itertuples():
        line_geom = line.geometry
        split_result = list(split(biggest_gb_geom, line_geom).geoms)
        split_gdf = gpd.GeoDataFrame(geometry=split_result, crs=region_gdf.crs)
        region_position = gpd.sjoin_nearest(region_points, split_gdf)
        line_sides[line.Boundary_n] = region_position["index_right"].clip(upper=1)
    line_sides_df = gpd.GeoDataFrame.from_dict(line_sides, orient="index")
    return line_sides_df.stack()


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


def set_flow_direction(series: pd.Series, sides: pd.DataFrame, n: pypsa.Network) -> int:
    """
    Set the flow direction for each line/link crossing based on the side of the boundary they are on.

    Parameters
    ----------
    series: pd.Series
        Series containing line/link information incl. the ETYS boundary being crossed
    sides: pd.DataFrame
        DataFrame containing region IDs and the side of each ETYS boundary they are on (0 or 1)
    n: pypsa.Network
        Clustered PyPSA network containing the final lines and links

    Returns
    -------
    int:
        The flow direction (1 or -1) for the line/link crossing.
        The value is only meaningful per boundary, where it indicates the relative direction of flow across that boundary.
    """
    buses = n.components[series.component].static.loc[series["name"], ["bus0", "bus1"]]
    side0 = sides.loc[series.Boundary_n, buses.bus0]
    side1 = sides.loc[series.Boundary_n, buses.bus1]
    direction = side1 - side0
    return direction


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    etys_gdf = gpd.read_file(snakemake.input.etys_boundaries)
    relevant_boundaries = pd.read_csv(snakemake.input.relevant_boundaries)
    etys_gdf = etys_gdf.query(
        "Boundary_n in @to_keep",
        local_dict={"to_keep": relevant_boundaries.boundary_name},
    )
    region_gdf = (
        gpd.read_file(snakemake.input.regions)
        .to_crs(etys_gdf.crs)
        .query("country == 'GB'")
    )
    n_base = pypsa.Network(snakemake.input.base_network)
    n_clustered = pypsa.Network(snakemake.input.clustered_network)

    sides = define_side_for_region(
        etys_gdf, region_gdf, snakemake.params.buffer_mainland_m
    )

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
        .reset_index(drop=True)
    )
    all_crossings["flow_direction"] = all_crossings.apply(
        set_flow_direction, sides=sides, n=n_clustered, axis=1
    )
    if (zeros := all_crossings.flow_direction == 0).any():
        logger.warning(
            "Some lines/links are crossing unexpected boundaries based on their assigned start & end regions. "
            "This may be caused by small branch lines, in which case you can ignore this issue. "
            "However, it may also indicate an issue with region boundary placement.\n"
            "Lines/links with zero flow direction across relevant boundaries:\n%s",
            all_crossings[zeros].to_string(index=False),
        )
    all_crossings_cleaned = all_crossings[~zeros]

    all_crossings_cleaned.to_csv(snakemake.output.csv, index=False)

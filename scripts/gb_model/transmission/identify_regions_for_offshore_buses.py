# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Create a busmap of offshore stub buses (from offshore wind farms) to the onshore bus that they are connected to.
"""

import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pypsa
from shapely import Point

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def create_busmap(
    lines: gpd.GeoDataFrame,
    buses: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    network: pypsa.Network,
) -> pd.Series:
    """
    Create a busmap mapping each bus to a region based on the line geometries and region geometries.

    Args:
        lines (gpd.GeoDataFrame): GeoDataFrame containing line geometries and their associated region information.
        buses (gpd.GeoDataFrame): GeoDataFrame containing bus geometries.
        regions (gpd.GeoDataFrame): GeoDataFrame containing region geometries.
        network (pypsa.Network): PyPSA network object containing the bus and line information.

    Returns:
        pd.Series: Series mapping each bus to its associated region.
    """

    # Identify offshore buses that are connected to onshore buses
    offshore_buses = _get_offshore_buses(lines, buses, regions, regions[["geometry"]])

    # Next, identify offshore buses that are themselves only connected to other offshore buses
    num_buses = len(offshore_buses)
    lines_in = offshore_buses
    while True:
        all_offshore_buses = _get_offshore_buses(
            lines, buses, regions, lines_in[["geometry", "bus_id"]]
        )
        # stop when we add no new offshore buses
        if len(all_offshore_buses) == num_buses:
            break
        elif len(all_offshore_buses) > num_buses:
            num_buses = len(all_offshore_buses)
            lines_in = all_offshore_buses
        elif len(all_offshore_buses) < num_buses:
            raise ValueError(
                "Unexpected reduction in number of offshore buses. "
                "This should not happen, please investigate."
            )

    onshore_buses = all_offshore_buses.apply(
        _get_onshore_bus, args=(buses, network, all_offshore_buses), axis=1
    )
    onshore_buses.index = all_offshore_buses.bus_id
    onshore_points = gpd.GeoSeries(onshore_buses, crs=all_offshore_buses.crs)
    busmap = gpd.sjoin(
        onshore_points.reset_index(), regions.to_crs(onshore_points.crs), how="left"
    ).set_index("bus_id")["region"]
    return busmap


def _get_offshore_buses(
    lines: gpd.GeoDataFrame,
    buses: gpd.GeoDataFrame,
    regions: gpd.GeoDataFrame,
    lines_in: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Identify offshore buses based on line and region geometries.

    Args:
        lines (gpd.GeoDataFrame): GeoDataFrame containing line geometries.
        buses (gpd.GeoDataFrame): GeoDataFrame containing bus geometries.
        regions (gpd.GeoDataFrame): GeoDataFrame containing region geometries.
        lines_in (gpd.GeoDataFrame): GeoDataFrame with geometries for which we want lines to overlap for us to consider them.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing offshore buses.
    """
    relevant_lines = gpd.sjoin(lines, lines_in.to_crs(lines.crs))
    relevant_buses = gpd.sjoin(buses, relevant_lines[["geometry", "line_id"]])
    offshore_buses = gpd.overlay(
        relevant_buses, regions.to_crs(buses.crs), how="difference"
    )
    return offshore_buses.drop_duplicates("bus_id")


def _get_onshore_bus(
    bus_series: gpd.GeoSeries,
    buses: gpd.GeoDataFrame,
    network: pypsa.Network,
    all_offshore_buses: gpd.GeoDataFrame,
) -> Point:
    """
    Resolve duplicate bus mappings by prioritising the closest region among the duplicates.

    Args:
        bus_series (pd.GeoSeries): GeoSeries containing bus geometry and associated region information, potentially with duplicates.
        buses (gpd.GeoDataFrame): GeoDataFrame containing bus geometries.
        network (pypsa.Network): PyPSA network object.

    Returns:
        pd.GeoDataFrame: GeoDataFrame with duplicates resolved, mapping each bus to a single region.
    """
    other_bus = (
        set(network.lines.loc[bus_series.line_id, ["bus0", "bus1"]].values)
        .difference([bus_series.bus_id])
        .pop()
    )
    if other_bus in all_offshore_buses.bus_id.values:
        logger.info(
            "Identifying busmap for offshore bus %s connected to another offshore bus %s",
            bus_series.bus_id,
            other_bus,
        )
        other_bus_series = all_offshore_buses[
            all_offshore_buses.bus_id == other_bus
        ].iloc[0]
        return _get_onshore_bus(other_bus_series, buses, network, all_offshore_buses)
    else:
        onshore_bus_point = buses.set_index("bus_id").loc[other_bus, "geometry"]
    return onshore_bus_point


def create_geom(n_data: pd.DataFrame, idx_name: str) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame of geometries from a DataFrame containing WKT formatted geometry strings.

    Args:
        n_data (pd.DataFrame): DataFrame containing network data with a 'geometry' column in WKT format.
        idx_name (str): Name of the index column to be used in the resulting GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame of geometries.
    """
    gseries = gpd.GeoSeries.from_wkt(
        n_data.geometry, index=n_data.index.rename(idx_name), crs="EPSG:4326"
    )
    gdf = gseries.to_frame("geometry").reset_index()
    return gdf


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    network = pypsa.Network(snakemake.input.base_network)
    lines = create_geom(network.lines, "line_id")
    buses = create_geom(network.buses, "bus_id")
    regions = (
        gpd.read_file(snakemake.input.regions)
        .query("`country` == 'GB'")
        .rename(columns={"name": "region"})
    )
    busmap = create_busmap(lines, buses, regions, network)
    busmap = busmap.rename_axis(index="Index").rename("busmap")
    busmap.to_csv(snakemake.output.csv)

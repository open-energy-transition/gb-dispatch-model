# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
Script to create region shapes by dividing country shapes based on ETYS boundary lines.

This script reads country shapes from a GeoJSON file and divides them using
boundary lines from the ETYS boundary data to create regional divisions.
The resulting regions are saved as a new GeoJSON file.
"""

import logging

import geopandas as gpd
from shapely.ops import split

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def load_country_shapes(filepath: str) -> gpd.GeoDataFrame:
    """
    Load country shapes from GeoJSON file.

    Args:
        filepath (str): Path to the country shapes GeoJSON file

    Returns:
        geopandas.GeoDataFrame: Country shapes data
    """
    logger.debug(f"Loading country shapes from: {filepath}")
    country_gdf = gpd.read_file(filepath)
    logger.debug(f"Loaded {len(country_gdf)} country shapes")
    return country_gdf


def load_boundary_lines(
    filepath: str, focus_filepath: str, pre_filter_boundaries: bool
) -> gpd.GeoDataFrame:
    """
    Load boundary lines from shapefile or GeoJSON.

    Args:
        filepath (str): Path to the boundary lines file
        focus_filepath (str): Path to the focus boundary lines file
        pre_filter_boundaries (bool): Whether to pre-filter boundaries

    Returns:
        geopandas.GeoDataFrame: Boundary lines data
    """

    logger.debug(f"Loading boundary lines from: {filepath}")
    boundary_gdf = gpd.read_file(filepath)

    if pre_filter_boundaries:
        focus_gdf = gpd.read_file(focus_filepath)
        boundaries_to_keep = boundary_gdf.Boundary_n.isin(focus_gdf.boundary_name)
        logger.debug(
            f"Pre-filtering boundaries: {len(boundary_gdf)} -> {boundaries_to_keep.sum()}"
        )
        boundary_gdf = boundary_gdf[boundaries_to_keep]

    logger.debug(f"Loaded {len(boundary_gdf)} boundary features")
    return boundary_gdf


def align_crs(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> tuple:
    """
    Ensure both GeoDataFrames have the same CRS.

    Args:
        gdf1, gdf2 (geopandas.GeoDataFrame): GeoDataFrames to align

    Returns:
        tuple: Both GeoDataFrames with the same CRS
    """
    if gdf1.crs != gdf2.crs:
        logger.debug(f"Converting CRS from {gdf2.crs} to {gdf1.crs}")
        gdf2 = gdf2.to_crs(gdf1.crs)
    return gdf1, gdf2


def create_regions_from_boundaries(
    country_shapes: gpd.GeoDataFrame,
    boundary_lines: gpd.GeoDataFrame,
    min_area_threshold: float,
    area_loss_tolerance_percent: float,
) -> gpd.GeoDataFrame:
    """
    Create regions by dividing country shapes using boundary lines.

    Args:
        country_shapes (geopandas.GeoDataFrame): Country polygons
        boundary_lines (geopandas.GeoDataFrame): Boundary lines for division

    Returns:
        geopandas.GeoDataFrame: Regional divisions
    """
    # Convert to a projected CRS for accurate area calculations
    # Use British National Grid (EPSG:27700) which is appropriate for UK
    target_crs = snakemake.config["target_crs"]
    logger.debug(f"Converting to projected CRS {target_crs} for accurate measurements")

    if country_shapes.crs != target_crs:
        country_shapes = country_shapes.to_crs(target_crs)
        logger.debug(
            f"Country shapes converted. New total area: {country_shapes.geometry.area.sum() / 1000000:.0f} km²"
        )
    if boundary_lines.crs != target_crs:
        boundary_lines = boundary_lines.to_crs(target_crs)
        logger.debug(
            f"Boundary lines converted. Length range: {boundary_lines.geometry.length.min():.0f} - {boundary_lines.geometry.length.max():.0f} meters"
        )

    if len(boundary_lines) == 0:
        raise ValueError("Non-empty boundary lines are expected for splitting!")

    logger.debug(f"Using {len(boundary_lines)} boundary lines for splitting")

    for idx, country in country_shapes.iterrows():
        logger.debug(f"Processing country/region {idx + 1}/{len(country_shapes)}")
        overlapping_lines = gpd.sjoin(
            boundary_lines,
            gpd.GeoDataFrame(geometry=[country.geometry], crs=target_crs),
            predicate="intersects",
            how="left",
        )
        if overlapping_lines.empty:
            logger.debug(
                f"No overlapping boundary lines found for {country['name']}; skipping splitting"
            )
            continue
        boundary_lines_agg = overlapping_lines.dissolve()
        regions = (
            gpd.GeoSeries(
                split(country.geometry, boundary_lines_agg.geometry.item()),
                crs=target_crs,
            )
            .explode()
            .to_frame("geometry")
            .reset_index(drop=True)
        )
        regions_filtered = drop_small_regions(
            regions, min_area_threshold=min_area_threshold
        )  # 10,000 m² = 0.01 km²
        logger.debug(
            f"Country {idx} split into {len(regions_filtered)} regions using {len(overlapping_lines)} boundaries"
        )
        regions_filtered["name"] = country["name"]
        regions_filtered["area_km2"] = regions_filtered.area / 1000000  # Convert to km²
        regions_filtered["region_id"] = regions_filtered.index.map(
            lambda x: f"region_{x:03d}"
        )

    # Create GeoDataFrame from regions
    if not regions_filtered.empty:
        # Calculate total area of regions
        original_area = country_shapes.geometry.area.sum()
        regions_area = regions_filtered.geometry.area.sum()
        area_difference = abs(original_area - regions_area)
        area_loss_percent = (
            (area_difference / original_area) * 100 if original_area > 0 else 0
        )

        logger.debug("Area comparison:")
        logger.debug(f"  Original area: {original_area / 1000000:.2f} km²")
        logger.debug(f"  Regions area:  {regions_area / 1000000:.2f} km²")
        logger.debug(
            f"  Difference:    {area_difference / 1000000:.2f} km² ({area_loss_percent:.3f}%)"
        )

        # raise exception if area loss is significant
        if area_loss_percent > area_loss_tolerance_percent:
            raise ValueError(
                f"Significant area loss detected after splitting: {area_loss_percent:.3f}%"
            )

        return regions_filtered
    else:
        raise ValueError(
            "Failed to create any regions from the provided country shapes and boundary lines"
        )


def drop_small_regions(
    regions_gdf: gpd.GeoDataFrame, min_area_threshold: float = 1000
) -> gpd.GeoDataFrame:
    """
    Clean up regions by removing very small polygons and fixing invalid geometries.

    Args:
        regions_gdf (geopandas.GeoDataFrame): Regions to clean
        min_area_threshold (float): Minimum area threshold for keeping regions

    Returns:
        geopandas.GeoDataFrame: Cleaned regions
    """
    initial_count = len(regions_gdf)

    # Fix invalid geometries
    regions_gdf["geometry"] = regions_gdf["geometry"].buffer(0)

    # Calculate areas
    regions_gdf["area"] = regions_gdf.geometry.area

    # Remove very small regions
    regions_gdf = regions_gdf[regions_gdf["area"] > min_area_threshold]

    # Reset index
    regions_gdf = regions_gdf.reset_index(drop=True)

    logger.debug(
        f"Cleaned regions: {initial_count} -> {len(regions_gdf)} (removed {initial_count - len(regions_gdf)} small regions)"
    )

    return regions_gdf


def save_regions(regions_gdf: gpd.GeoDataFrame, output_path: str) -> None:
    """
    Save regions to GeoJSON file.

    Args:
        regions_gdf (geopandas.GeoDataFrame): Regions to save
        output_path (str): Output file path
    """
    # Convert back to WGS84 for better map compatibility
    regions_wgs84 = regions_gdf.to_crs("EPSG:4326")

    # Save to GeoJSON in WGS84 format
    regions_wgs84.to_file(output_path, driver="GeoJSON")

    logger.debug(f"Successfully saved {len(regions_gdf)} regions to {output_path}")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("create_region_shapes")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # load country shapes
    country_shapes = load_country_shapes(snakemake.input.country_shapes)

    # Filter out GB shapes
    country_shapes = country_shapes[country_shapes.name == "GB"]

    # load ETYS boundary lines
    boundary_lines = load_boundary_lines(
        snakemake.input.etys_boundary_lines,
        snakemake.input.etys_focus_boundary_lines,
        pre_filter_boundaries=snakemake.params.pre_filter_boundaries,
    )

    # align CRS
    country_shapes, boundary_lines = align_crs(country_shapes, boundary_lines)

    # Ensure boundary lines exist
    if boundary_lines.empty:
        raise ValueError("No boundary lines found in the provided ETYS data!")

    # create regions from boundaries
    regions = create_regions_from_boundaries(
        country_shapes,
        boundary_lines,
        snakemake.params.min_region_area,
        snakemake.params.area_loss_tolerance_percent,
    )
    logger.debug(f"Created {len(regions)} regions")
    if len(regions) > 1:
        logger.debug(
            f"- Area range: {regions.geometry.area.min() / 1000000:.1f} - {regions.geometry.area.max() / 1000000:.1f} km²"
        )
        logger.debug(
            f"- Average area: {regions.geometry.area.mean() / 1000000:.1f} km²"
        )

    # save regions to output file
    save_regions(regions, snakemake.output.raw_region_shapes)

    # log final summary
    logger.debug("REGION CREATION SUMMARY")
    logger.debug(f"ETYS boundary lines: {len(boundary_lines)}")
    logger.info(f"Regions after splitting: {len(regions)}")

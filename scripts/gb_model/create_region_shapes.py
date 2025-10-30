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


def load_boundary_lines(filepath: str) -> gpd.GeoDataFrame:
    """
    Load boundary lines from shapefile or GeoJSON.

    Args:
        filepath (str): Path to the boundary lines file

    Returns:
        geopandas.GeoDataFrame: Boundary lines data
    """
    logger.debug(f"Loading boundary lines from: {filepath}")
    boundary_gdf = gpd.read_file(filepath)
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
    country_shapes: gpd.GeoDataFrame, boundary_lines: gpd.GeoDataFrame
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

    regions = []
    region_id = 1

    for idx, country in country_shapes.iterrows():
        logger.debug(f"Processing country/region {idx + 1}/{len(country_shapes)}")

        # Start with the original country geometry
        current_polygons = [country.geometry]
        split_count = 0

        # Process each boundary line
        for boundary_idx, boundary in boundary_lines.iterrows():
            boundary_geom = boundary.geometry
            new_polygons = []

            for polygon in current_polygons:
                # Check if boundary intersects this polygon
                if polygon.intersects(boundary_geom):
                    try:
                        # Attempt to split the polygon
                        split_result = split(polygon, boundary_geom)

                        # Check if splitting actually occurred
                        if hasattr(split_result, "geoms"):
                            split_geoms = list(split_result.geoms)
                            logger.debug(
                                f"Split produced {len(split_geoms)} geometries"
                            )
                            if len(split_geoms) > 1:
                                # Successfully split - add all valid pieces
                                valid_pieces = 0
                                for geom in split_geoms:
                                    if (
                                        geom.is_valid
                                        and not geom.is_empty
                                        and geom.geom_type
                                        in ["Polygon", "MultiPolygon"]
                                        and geom.area > 1000000
                                    ):  # 1 km² minimum (1,000,000 sq meters in projected CRS)
                                        new_polygons.append(geom)
                                        valid_pieces += 1
                                    else:
                                        logger.debug(
                                            f"Filtered out small geometry: area={geom.area:.0f} sq meters"
                                        )

                                if valid_pieces > 0:
                                    split_count += 1
                                    logger.debug(
                                        f"Split polygon into {valid_pieces} valid pieces (from {len(split_geoms)} total)"
                                    )
                                else:
                                    logger.warning(
                                        f"All {len(split_geoms)} split pieces were too small - keeping original"
                                    )
                                    new_polygons.append(polygon)
                            else:
                                # No actual split occurred
                                new_polygons.append(polygon)
                        else:
                            # Single geometry result - no split
                            new_polygons.append(polygon)

                    except Exception as split_error:
                        logger.warning(
                            f"Split failed for boundary {boundary_idx}: {split_error}"
                        )
                        new_polygons.append(polygon)
                else:
                    # No intersection - keep original polygon
                    new_polygons.append(polygon)

            # Update current polygons for next iteration
            current_polygons = new_polygons

        logger.debug(
            f"Country {idx} split into {len(current_polygons)} regions using {split_count} boundaries"
        )

        # Create region entries from final polygons
        for i, polygon in enumerate(current_polygons):
            if polygon.is_valid and not polygon.is_empty:
                # Create region data
                region_data = country.copy()
                region_data["geometry"] = polygon
                region_data["region_id"] = f"region_{region_id:03d}"
                region_data["original_country_id"] = idx
                region_data["sub_region_id"] = i
                region_data["area_km2"] = polygon.area / 1000000  # Convert to km²
                region_data["num_boundaries_used"] = split_count

                regions.append(region_data)
                region_id += 1

    # Create GeoDataFrame from regions
    if regions:
        regions_gdf = gpd.GeoDataFrame(regions, crs=country_shapes.crs)

        # Calculate total area of regions
        original_area = country_shapes.geometry.area.sum()
        regions_area = regions_gdf.geometry.area.sum()
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
        if area_loss_percent > snakemake.config["area_loss_tolerance_percent"]:
            raise ValueError(
                f"Significant area loss detected after splitting: {area_loss_percent:.3f}%"
            )

        logger.debug(
            f"Successfully created {len(regions_gdf)} regions from {len(country_shapes)} original shapes"
        )
        return regions_gdf
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
    boundary_lines = load_boundary_lines(snakemake.input.etys_boundary_lines)

    # align CRS
    country_shapes, boundary_lines = align_crs(country_shapes, boundary_lines)

    # Ensure boundary lines exist
    if boundary_lines.empty:
        raise ValueError("No boundary lines found in the provided ETYS data!")

    # create regions from boundaries
    regions = create_regions_from_boundaries(country_shapes, boundary_lines)
    logger.debug(f"Created {len(regions)} initial regions")
    if len(regions) > 1:
        logger.debug(
            f"- Area range: {regions.geometry.area.min() / 1000000:.1f} - {regions.geometry.area.max() / 1000000:.1f} km²"
        )
        logger.debug(
            f"- Average area: {regions.geometry.area.mean() / 1000000:.1f} km²"
        )

    # Clean regions with appropriate threshold
    min_area = snakemake.config["min_region_area"]
    logger.debug(
        f"\nCleaning regions (removing regions < {min_area / 1000000:.0f} km²)..."
    )
    cleaned_regions = drop_small_regions(regions, min_area_threshold=min_area)

    # save regions to output file
    save_regions(cleaned_regions, snakemake.output.raw_region_shapes)

    # log final summary
    logger.debug("REGION CREATION SUMMARY")
    logger.debug(f"ETYS boundary lines: {len(boundary_lines)}")
    logger.info(f"Regions after splitting: {len(regions)}")

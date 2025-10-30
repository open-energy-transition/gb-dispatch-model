# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
Manual Region Merger Script

This script allows manual splitting and joining of specific regions from raw_region_shapes.geojson
based on provided region IDs and split specifications.
"""

import logging

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString
from shapely.ops import split, unary_union

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def calculate_area_km2(geometry, crs) -> float:
    """Calculate area in square kilometers, assuming CRS is in meters"""
    if crs.is_projected:
        return geometry.area / 1000000  # Convert m² to km²
    else:
        return abs(crs.get_geod().geometry_area_perimeter(geometry)[0]) / 1000000


def split_region_vertical(
    regions_gdf: gpd.GeoDataFrame, region_num: int, longitude: float
) -> gpd.GeoDataFrame:
    """
    Split a region vertically at specified longitude

    Args:
        regions_gdf: GeoDataFrame containing regions
        region_num: Numeric ID of region to split
        longitude: Longitude coordinate to split at

    Returns:
        Updated GeoDataFrame with split regions
    """
    logger.debug(f"Splitting region {region_num} vertically at longitude {longitude}")

    # Find the target region
    target_region_mask = regions_gdf["numeric_id"] == region_num
    if not target_region_mask.any():
        logger.error(f"Region {region_num} not found!")
        return regions_gdf

    target_region = regions_gdf[target_region_mask].iloc[0]
    target_idx = regions_gdf[target_region_mask].index[0]

    # Get region bounds to create a splitting line
    bounds = target_region.geometry.bounds
    min_lat, max_lat = bounds[1], bounds[3]

    # Create vertical splitting line at specified longitude
    splitting_line = LineString([(longitude, min_lat), (longitude, max_lat)])

    # Perform split
    split_result = split(target_region.geometry, splitting_line)

    if hasattr(split_result, "geoms") and len(split_result.geoms) >= 2:
        logger.debug(
            f"Successfully split region {region_num} into {len(split_result.geoms)} parts"
        )

        # For region 6, handle special case with 3 parts, because it is a concave shaped region.
        # When split by vertical line, it can produce 3 parts: west-north, west-south, east regions.
        # This is an exception case, because west-south part needs to be joined with east part.
        if region_num == 6 and len(split_result.geoms) >= 3:
            # Sort all parts by longitude first
            parts_with_centroids = []
            for geom in split_result.geoms:
                centroid = geom.centroid
                parts_with_centroids.append((geom, centroid.x, centroid.y))

            # Sort by longitude (west to east)
            parts_with_centroids.sort(key=lambda x: x[1])

            # The westernmost part - check if it has north and south components
            west_parts = []
            east_part = None

            # Separate into west (< longitude) and east (>= longitude)
            for geom, lon, lat in parts_with_centroids:
                if lon < longitude:
                    west_parts.append((geom, lon, lat))
                else:
                    if east_part is None:
                        east_part = (geom, lon, lat)
                    else:
                        # If multiple east parts, combine them
                        from shapely.ops import unary_union

                        combined_geom = unary_union([east_part[0], geom])
                        east_part = (
                            combined_geom,
                            combined_geom.centroid.x,
                            combined_geom.centroid.y,
                        )

            new_regions = []

            # Handle west parts - if 2 or more, name them 6wn and 6ws
            if len(west_parts) >= 2:
                # Sort west parts by latitude (south to north)
                west_parts.sort(key=lambda x: x[2])

                suffixes = ["ws", "wn"]  # south first, then north
                for i, (geom, _, _) in enumerate(west_parts[:2]):  # Take max 2 parts
                    new_region = target_region.copy()
                    new_region["geometry"] = geom
                    new_region["region_id"] = f"region_{region_num:03d}{suffixes[i]}"
                    new_region["numeric_id"] = f"{region_num}{suffixes[i]}"

                    if "area_km2" in new_region.index:
                        new_area_km2 = calculate_area_km2(geom, regions_gdf.crs)
                        new_region["area_km2"] = new_area_km2

                    new_regions.append(new_region)
                    logger.debug(
                        f"Created region {new_region['region_id']} ({suffixes[i]})"
                    )

            elif len(west_parts) == 1:
                # Only one west part, name it 6w
                geom = west_parts[0][0]
                new_region = target_region.copy()
                new_region["geometry"] = geom
                new_region["region_id"] = f"region_{region_num:03d}w"
                new_region["numeric_id"] = f"{region_num}w"

                if "area_km2" in new_region.index:
                    new_area_km2 = calculate_area_km2(geom, regions_gdf.crs)
                    new_region["area_km2"] = new_area_km2

                new_regions.append(new_region)
                logger.debug(f"Created region {new_region['region_id']} (w)")

            # Handle east part
            if east_part is not None:
                geom = east_part[0]
                new_region = target_region.copy()
                new_region["geometry"] = geom
                new_region["region_id"] = f"region_{region_num:03d}e"
                new_region["numeric_id"] = f"{region_num}e"

                if "area_km2" in new_region.index:
                    new_area_km2 = calculate_area_km2(geom, regions_gdf.crs)
                    new_region["area_km2"] = new_area_km2

                new_regions.append(new_region)
                logger.debug(f"Created region {new_region['region_id']} (e)")

        else:
            # Standard 2-part split (for other regions or if region 6 only splits into 2)
            centroids = [geom.centroid for geom in split_result.geoms]
            longitudes = [centroid.x for centroid in centroids]

            # Sort by longitude (west to east)
            sorted_parts = sorted(
                zip(split_result.geoms, longitudes), key=lambda x: x[1]
            )

            # Create new region entries
            new_regions = []
            suffixes = ["w", "e"]  # west, east

            for i, (geom, _) in enumerate(sorted_parts):
                if i < len(suffixes):
                    new_region = target_region.copy()
                    new_region["geometry"] = geom
                    new_region["region_id"] = f"region_{region_num:03d}{suffixes[i]}"
                    new_region["numeric_id"] = f"{region_num}{suffixes[i]}"

                    if "area_km2" in new_region.index:
                        new_area_km2 = calculate_area_km2(geom, regions_gdf.crs)
                        new_region["area_km2"] = new_area_km2

                    new_regions.append(new_region)
                    logger.debug(
                        f"Created region {new_region['region_id']} ({suffixes[i]})"
                    )

        # Remove original region and add new ones
        result_gdf = regions_gdf.drop(index=target_idx)

        # Create a GeoDataFrame from new regions to preserve CRS
        if new_regions:
            new_regions_gdf = gpd.GeoDataFrame(new_regions, crs=regions_gdf.crs)
            result_gdf = pd.concat([result_gdf, new_regions_gdf], ignore_index=True)

        # Ensure the result maintains the original CRS
        if result_gdf.crs is None:
            result_gdf = result_gdf.set_crs(regions_gdf.crs)
        return result_gdf

    else:
        logger.error(f"Split failed - line doesn't properly divide region {region_num}")
        return regions_gdf


def split_region_horizontal(
    regions_gdf: gpd.GeoDataFrame, region_num: int, latitude: float
) -> gpd.GeoDataFrame:
    """
    Split a region horizontally at specified latitude

    Args:
        regions_gdf: GeoDataFrame containing regions
        region_num: Numeric ID of region to split
        latitude: Latitude coordinate to split at

    Returns:
        Updated GeoDataFrame with split regions
    """
    logger.debug(f"Splitting region {region_num} horizontally at latitude {latitude}")

    # Find the target region
    target_region_mask = regions_gdf["numeric_id"] == region_num
    if not target_region_mask.any():
        logger.error(f"Region {region_num} not found!")
        return regions_gdf

    target_region = regions_gdf[target_region_mask].iloc[0]
    target_idx = regions_gdf[target_region_mask].index[0]

    # Get region bounds to create a splitting line
    bounds = target_region.geometry.bounds
    min_lon, max_lon = bounds[0], bounds[2]

    # Create horizontal splitting line at specified latitude
    splitting_line = LineString([(min_lon, latitude), (max_lon, latitude)])

    # Perform split
    split_result = split(target_region.geometry, splitting_line)

    if hasattr(split_result, "geoms") and len(split_result.geoms) >= 2:
        logger.debug(
            f"Successfully split region {region_num} into {len(split_result.geoms)} parts"
        )

        # Determine which part is north and which is south
        centroids = [geom.centroid for geom in split_result.geoms]
        latitudes = [centroid.y for centroid in centroids]

        # Sort by latitude (south to north)
        sorted_parts = sorted(zip(split_result.geoms, latitudes), key=lambda x: x[1])

        # Create new region entries
        new_regions = []
        suffixes = ["s", "n"]  # south, north

        for i, (geom, _) in enumerate(sorted_parts):
            if i < len(suffixes):
                new_region = target_region.copy()
                new_region["geometry"] = geom
                new_region["region_id"] = f"region_{region_num:03d}{suffixes[i]}"
                new_region["numeric_id"] = f"{region_num}{suffixes[i]}"

                # Update area if column exists
                if "area_km2" in new_region.index:
                    new_area_km2 = calculate_area_km2(geom, regions_gdf.crs)
                    new_region["area_km2"] = new_area_km2

                new_regions.append(new_region)
                logger.debug(
                    f"Created region {new_region['region_id']} ({suffixes[i]})"
                )

        # Remove original region and add new ones
        result_gdf = regions_gdf.drop(index=target_idx)

        # Create a GeoDataFrame from new regions to preserve CRS
        if new_regions:
            new_regions_gdf = gpd.GeoDataFrame(new_regions, crs=regions_gdf.crs)
            result_gdf = pd.concat([result_gdf, new_regions_gdf], ignore_index=True)

        # Ensure the result maintains the original CRS
        if result_gdf.crs is None:
            result_gdf = result_gdf.set_crs(regions_gdf.crs)
        return result_gdf

    else:
        logger.error(f"Split failed - line doesn't properly divide region {region_num}")
        return regions_gdf


def load_regions(input_file: str) -> gpd.GeoDataFrame:
    """Load regions from GeoJSON file"""
    logger.debug(f"Loading regions from: {input_file}")
    regions_gdf = gpd.read_file(input_file)
    logger.debug(f"Loaded {len(regions_gdf)} regions")

    # Show available region IDs for reference
    if "region_id" in regions_gdf.columns:
        region_ids = regions_gdf["region_id"].tolist()
        logger.debug(f"Sample region IDs: {region_ids[:10]}")
        logger.debug(f"Region ID type: {type(region_ids[0])}")

        # Extract numeric parts from region IDs for matching
        numeric_ids = []
        for rid in region_ids:
            # Try to extract number from region ID (e.g., "region_54" -> 54)
            import re

            numbers = re.findall(r"\d+", str(rid))
            if numbers:
                numeric_ids.append(int(numbers[-1]))  # Take the last number found
            else:
                numeric_ids.append(None)

        # Add numeric ID column for easier matching
        regions_gdf["numeric_id"] = numeric_ids
        logger.debug(
            f"Extracted numeric IDs: {sorted([x for x in numeric_ids if x is not None])}"
        )

    else:
        logger.debug("No 'region_id' column found, using index as region ID")
        regions_gdf["region_id"] = regions_gdf.index + 1
        regions_gdf["numeric_id"] = regions_gdf.index + 1

    return regions_gdf


def cut_regions_before_merge(regions_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Cut regions before merging based on configuration
    """
    splits = snakemake.config["region_operations"]["splits"]

    for split_config in splits:
        region = split_config["region"]
        split_type = split_config["type"]
        coordinate = split_config["coordinate"]

        if split_type == "vertical":
            regions_gdf = split_region_vertical(regions_gdf, region, coordinate)
        elif split_type == "horizontal":
            regions_gdf = split_region_horizontal(regions_gdf, region, coordinate)
        else:
            logger.warning(f"Unknown split type: {split_type}")

    # Show split regions created
    split_regions = regions_gdf[
        regions_gdf["region_id"].str.contains("w|e|n|s", na=False)
    ]
    if len(split_regions) > 0:
        logger.debug("Split regions created:")
        for _, region in split_regions.iterrows():
            logger.debug(f"  - {region['region_id']} (numeric: {region['numeric_id']})")

    return regions_gdf


def merge_regions(
    regions_gdf: gpd.GeoDataFrame, merge_groups: list
) -> gpd.GeoDataFrame:
    """
    Merge specified groups of regions

    Args:
        regions_gdf: GeoDataFrame containing regions
        merge_groups: List of lists, each containing region IDs to merge

    Returns:
        GeoDataFrame with merged regions
    """
    logger.debug(f"Processing {len(merge_groups)} merge groups")

    # Create a copy to work with
    result_gdf = regions_gdf.copy()
    regions_to_remove = set()

    for i, group in enumerate(merge_groups):
        logger.debug(f"Processing merge group {i + 1}: regions {group}")

        # Find regions in this group (handle both numeric and string IDs)
        group_regions_list = []
        for region_id in group:
            if isinstance(region_id, str):
                # Handle split regions like '6w', '6e', '46n', '46s'
                mask = regions_gdf["numeric_id"] == region_id
            else:
                # Handle numeric IDs
                mask = regions_gdf["numeric_id"] == region_id

            matching_regions = result_gdf[mask]
            if len(matching_regions) > 0:
                group_regions_list.append(matching_regions.iloc[0])
            else:
                logger.warning(f"Region {region_id} not found in group {group}")

        if len(group_regions_list) == 0:
            logger.warning(f"No regions found for group {group}")
            continue

        # Convert to GeoDataFrame
        group_regions = gpd.GeoDataFrame(group_regions_list, crs=result_gdf.crs)

        if len(group_regions) < 2:
            logger.warning(f"Group {group} has less than 2 regions, skipping merge")
            continue

        # Get the first region as the base (will keep this one)
        base_region = group_regions.iloc[0]
        base_region_idx = None

        # Find the index in result_gdf
        for idx, row in result_gdf.iterrows():
            if row["region_id"] == base_region["region_id"]:
                base_region_idx = idx
                break

        if base_region_idx is None:
            logger.error(
                f"Could not find base region index for {base_region['region_id']}"
            )
            continue

        # Collect geometries to merge
        geometries_to_merge = [
            region.geometry for _, region in group_regions.iterrows()
        ]

        # Merge geometries
        try:
            merged_geometry = unary_union(geometries_to_merge)
            logger.debug(f"Successfully merged {len(geometries_to_merge)} geometries")
        except Exception as e:
            logger.error(f"Failed to merge geometries for group {group}: {e}")
            continue

        # Update the base region with merged geometry
        result_gdf.loc[base_region_idx, "geometry"] = merged_geometry

        # Update area if column exists
        if "area_km2" in result_gdf.columns:
            # Calculate new area (assuming CRS is in meters)
            new_area_km2 = calculate_area_km2(merged_geometry, regions_gdf.crs)
            result_gdf.loc[base_region_idx, "area_km2"] = new_area_km2
            logger.debug(f"Updated area to {new_area_km2:.2f} km²")

        # Track merged region IDs
        merged_region_ids = ", ".join(map(str, group[1:]))  # All except the first
        if "merged_regions" in result_gdf.columns:
            existing_merged = result_gdf.loc[base_region_idx, "merged_regions"]
            if pd.isna(existing_merged) or existing_merged == "":
                result_gdf.loc[base_region_idx, "merged_regions"] = merged_region_ids
            else:
                result_gdf.loc[base_region_idx, "merged_regions"] = (
                    f"{existing_merged}, {merged_region_ids}"
                )
        else:
            result_gdf["merged_regions"] = ""
            result_gdf.loc[base_region_idx, "merged_regions"] = merged_region_ids

        # Mark other regions in group for removal
        for region_id in group[1:]:  # All except the first
            if isinstance(region_id, str):
                region_indices = result_gdf[result_gdf["numeric_id"] == region_id].index
            else:
                region_indices = result_gdf[result_gdf["numeric_id"] == region_id].index
            regions_to_remove.update(region_indices)

        logger.debug(f"Merged regions {group} into region {base_region['region_id']}")

    # Remove merged regions
    if regions_to_remove:
        logger.debug(f"Removing {len(regions_to_remove)} merged regions")
        result_gdf = result_gdf.drop(index=regions_to_remove)

    # Reset index
    result_gdf = result_gdf.reset_index(drop=True)

    result_gdf["name"] = result_gdf["name"] + " " + result_gdf["numeric_id"].astype(str)

    logger.debug(f"Final result: {len(regions_gdf)} -> {len(result_gdf)} regions")
    return result_gdf


def append_country_shapes(
    regions_gdf: gpd.GeoDataFrame, country_shapes_file: str
) -> gpd.GeoDataFrame:
    """
    Append country shapes other than GB to the regions GeoDataFrame

    Args:
        regions_gdf: GeoDataFrame containing merged regions
        country_shapes_file: Path to country shapes GeoJSON file

    Returns:
        GeoDataFrame with country shapes appended
    """
    logger.debug(f"Loading country shapes from: {country_shapes_file}")
    country_shapes_gdf = gpd.read_file(country_shapes_file)
    logger.debug(f"Loaded {len(country_shapes_gdf)} country shapes")

    # Ensure CRS compatibility
    if country_shapes_gdf.crs != regions_gdf.crs:
        logger.debug(
            f"Converting country shapes CRS from {country_shapes_gdf.crs} to {regions_gdf.crs}"
        )
        country_shapes_gdf = country_shapes_gdf.to_crs(regions_gdf.crs)

    # assign country column
    regions_gdf["country"] = "GB"
    country_shapes_gdf["country"] = country_shapes_gdf["name"]

    # Filter out GB country shape
    country_shapes_gdf = country_shapes_gdf[country_shapes_gdf["name"] != "GB"]

    # Concatenate the GeoDataFrames
    result_gdf = pd.concat([regions_gdf, country_shapes_gdf], ignore_index=True)
    logger.debug(
        f"Appended country shapes: {len(regions_gdf)} regions + {len(country_shapes_gdf)} country shapes = {len(result_gdf)} total"
    )

    return result_gdf


def save_regions(regions_gdf: gpd.GeoDataFrame, output_file: str) -> None:
    """Save regions to GeoJSON file"""
    logger.debug(f"Saving {len(regions_gdf)} regions to: {output_file}")
    regions_gdf.to_file(output_file, driver="GeoJSON")
    logger.debug("Save completed successfully")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "manual_region_merger", configfiles="config/config.GB_main.yaml"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # load regions
    regions_gdf = load_regions(snakemake.input.raw_region_shapes)

    # Perform splitting operations first
    regions_gdf = cut_regions_before_merge(regions_gdf)

    # Get merge groups from config
    merge_groups = snakemake.config["region_operations"]["merge_groups"]

    # Perform merging operations
    merged_regions = merge_regions(regions_gdf, merge_groups)
    logger.info(
        f"Merging completed. Total regions after merging: {len(merged_regions)}"
    )

    # Append country shapes
    merged_regions = append_country_shapes(
        merged_regions, snakemake.input.country_shapes
    )
    # Save results
    save_regions(merged_regions, snakemake.output.merged_shapes)

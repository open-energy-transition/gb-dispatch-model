# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
OSM name to ID mapper.

Maps OpenStreetMap names to their corresponding IDs for GB model processing.
"""

import json
import logging
from pathlib import Path

import pandas as pd
from shapely.geometry import Polygon

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_osm_network import DISTANCE_CRS, GEO_CRS
from scripts.clean_osm_data import _clean_voltage

logger = logging.getLogger(__name__)


class OSMNameMapper:
    def __init__(self, osm_files: dict[str, Path]) -> None:
        """
        Initialize the OSMNameMapper with paths to OSM data files.

        Args:
            osm_files (dict): Dictionary mapping OSM feature types to file paths.
                Keys: 'cables_way', 'lines_way', 'routes_relation',
                      'substations_way', 'substations_relation'
        """
        self.osm_files = osm_files
        self.geo_crs = GEO_CRS
        self.distance_crs = DISTANCE_CRS
        self.combined_df = self._create_combined_df()

    def _read_osm_file(self, file_path: Path, feature_type: str) -> pd.DataFrame:
        """
        Read an OSM JSON file and extract data as a DataFrame.

        Args:
            file_path (Path): Path to the OSM JSON file.
            feature_type (str): Type of feature (e.g., 'cables_way', 'lines_way').

        Returns:
            pd.DataFrame: DataFrame with columns:
                - id: OSM ID
                - name: Feature name
                - ref: Reference code
                - operator: Operator name
                - voltage: Voltage level
                - type: Feature type
        """
        logger.info(f"Reading OSM file: {file_path} for feature type: {feature_type}")

        osm_data = json.load(file_path.open(encoding="utf-8"))

        elements = osm_data.get("elements", [])
        logger.info(f"Found {len(elements)} elements in {file_path}")

        # Extract data into list of dictionaries
        data = []
        for element in elements:
            osm_id = element.get("id")
            tags = element.get("tags", {})
            geometry_data = element.get("geometry", {})

            geometry = None
            if geometry_data:
                try:
                    # OSM geometry is typically: [{"lat": 52.1, "lon": 1.2}, ...]
                    coords = [(point["lon"], point["lat"]) for point in geometry_data]

                    # Create Polygon if closed (first == last)
                    if len(coords) >= 3:
                        # Check if polygon is closed
                        is_closed = coords[0] == coords[-1]

                        if is_closed:
                            geometry = Polygon(coords)
                        else:
                            # If not closed, close it
                            geometry = Polygon(coords + [coords[0]])
                except (KeyError, TypeError, ValueError) as e:
                    logger.debug(f"Could not create polygon for element {osm_id}: {e}")

            data.append(
                {
                    "id": osm_id,
                    "name": tags.get("name", ""),
                    "voltage": tags.get("voltage", ""),
                    "geometry": geometry,
                    "type": feature_type,
                }
            )

        df = pd.DataFrame(data)
        logger.info(f"Created DataFrame with {len(df)} rows for {feature_type}")

        return df

    def _check_duplicate_ids(self, df: pd.DataFrame) -> None:
        """
        Check for duplicate OSM IDs in the DataFrame and log warnings if found.

        Args:
            df (pd.DataFrame): DataFrame containing OSM data.
        """
        duplicate_ids = df[df.duplicated(subset=["id"], keep=False)]
        if not duplicate_ids.empty:
            logger.warning(f"Found {len(duplicate_ids)} rows with duplicate IDs")
            logger.debug(f"Duplicate IDs: {duplicate_ids['id'].unique().tolist()}")

    def _drop_empty_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop rows with empty names from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing OSM data.

        Returns:
            pd.DataFrame: DataFrame with empty names dropped.
        """
        initial_count = len(df)
        df_cleaned = df[df["name"].str.strip() != ""]
        dropped_count = initial_count - len(df_cleaned)
        if dropped_count > 0:
            logger.info(f"Dropped {dropped_count} rows with empty names")
        return df_cleaned

    def _create_combined_df(self) -> pd.DataFrame:
        """
        Create a combined DataFrame from all OSM data files.

        Reads all OSM data files and creates a unified DataFrame with all entries.

        Returns:
            pd.DataFrame: Combined DataFrame with columns:
                - id: OSM ID
                - name: Feature name
                - voltage: Voltage level
                - geometry: Geometry data
                - type: Feature type
        """
        logger.info("Creating combined OSM DataFrame.")

        dfs = []
        for feature_type, file_path in self.osm_files.items():
            df = self._read_osm_file(file_path, feature_type)
            if not df.empty:
                dfs.append(df)

        if dfs:
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(
                f"Created combined DataFrame with {len(combined_df)} total entries"
            )

            # Check for duplicate IDs
            self._check_duplicate_ids(combined_df)

            # Drop entries with empty names
            combined_df = self._drop_empty_names(combined_df)

            # Clean voltage data
            combined_df["voltage"] = _clean_voltage(combined_df["voltage"])

            # Convert geometry to WKT for easier storage
            combined_df["geometry"] = combined_df["geometry"].apply(
                lambda g: g.wkt if g is not None else None
            )

            return combined_df
        else:
            raise ValueError("No data found in any OSM files")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Create dictionaries from snakemake inputs
    osm_files = {
        "cables_way": Path(snakemake.input.raw_cables_way),
        "lines_way": Path(snakemake.input.raw_lines_way),
        "routes_relation": Path(snakemake.input.raw_routes_relation),
        "substations_way": Path(snakemake.input.raw_substations_way),
        "substations_relation": Path(snakemake.input.raw_substations_relation),
    }

    # Get mapping of names to IDs
    mapper = OSMNameMapper(osm_files=osm_files)

    # Access the DataFrame
    osm_mapping_df = mapper.combined_df

    # Save to CSV
    osm_mapping_df.to_csv(snakemake.output.osm_mapping, index=False)

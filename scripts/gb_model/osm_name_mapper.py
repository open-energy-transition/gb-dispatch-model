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

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


class OSMNameMapper:
    def __init__(
        self,
        cables_way_path: Path,
        lines_way_path: Path,
        routes_relation_path: Path,
        substations_way_path: Path,
        substations_relation_path: Path,
    ) -> None:
        """
        Initialize the OSMNameMapper with paths to OSM data files.

        Args:
            cables_way_path (Path): Path to the cables way OSM data.
            lines_way_path (Path): Path to the lines way OSM data.
            routes_relation_path (Path): Path to the routes relation OSM data.
            substations_way_path (Path): Path to the substations way OSM data.
            substations_relation_path (Path): Path to the substations relation OSM data.
        """
        self.osm_files = {
            "cables_way": cables_way_path,
            "lines_way": lines_way_path,
            "routes_relation": routes_relation_path,
            "substations_way": substations_way_path,
            "substations_relation": substations_relation_path,
        }

        self.name_to_id_map = self._create_name_to_id_map()

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

        try:
            with open(file_path, encoding="utf-8") as f:
                osm_data = json.load(f)

            elements = osm_data.get("elements", [])
            logger.info(f"Found {len(elements)} elements in {file_path}")

            # Extract data into list of dictionaries
            data = []
            for element in elements:
                osm_id = element.get("id")
                tags = element.get("tags", {})

                data.append(
                    {
                        "id": osm_id,
                        "name": tags.get("name", ""),
                        "voltage": tags.get("voltage", ""),
                        "type": feature_type,
                    }
                )

            df = pd.DataFrame(data)
            logger.info(f"Created DataFrame with {len(df)} rows for {feature_type}")

            return df

        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            return pd.DataFrame()
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {file_path}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path}: {e}")
            return pd.DataFrame()

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
        else:
            logger.info("No duplicate IDs found")

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

    def _aggregate_by_name(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate OSM data by name, creating lists of IDs, types, and voltages.

        For each unique name, collect all associated IDs, feature types, and voltage levels.

        Args:
            df (pd.DataFrame): DataFrame with columns: id, name, voltage, type

        Returns:
            pd.DataFrame: Aggregated DataFrame with columns:
                - name: Feature name (index)
                - ids: List of OSM IDs
                - types: List of feature types
                - voltages: List of voltage levels
                - count: Number of occurrences
        """
        logger.info("Aggregating data by name")

        aggregated = (
            df.groupby("name")
            .agg(
                {
                    "id": lambda x: x.tolist(),
                    "type": lambda x: x.tolist(),
                    "voltage": lambda x: x.tolist(),
                }
            )
            .rename(columns={"id": "ids", "type": "types", "voltage": "voltages"})
        )

        # Add count column
        aggregated["count"] = df.groupby("name").size()

        # Reset index to make 'name' a column
        aggregated = aggregated.reset_index()

        logger.info(f"Aggregated to {len(aggregated)} unique names")

        # Log some statistics
        multiple_ids = aggregated[aggregated["count"] > 1]
        if not multiple_ids.empty:
            logger.info(f"Found {len(multiple_ids)} names with multiple entries")
            logger.debug(f"Examples: {multiple_ids.head()['name'].tolist()}")

        return aggregated

    def _create_name_to_id_map(self) -> pd.DataFrame:
        """
        Create a mapping DataFrame from OSM names to their corresponding IDs.

        Reads all OSM data files and creates a unified DataFrame.

        Returns:
            pd.DataFrame: DataFrame with columns:
                - name: Feature name
                - ids: List of OSM IDs
                - types: List of feature types
                - voltages: List of voltage levels
                - count: Number of occurrences
        """
        logger.info("Creating OSM name to ID mapping.")

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
        else:
            raise ValueError("No data found in any OSM files")

        # Drop entries with empty names
        combined_df = self._drop_empty_names(combined_df)

        # Aggregate by name
        aggregated_df = self._aggregate_by_name(combined_df)

        return aggregated_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load input paths
    cables_way_path = snakemake.input.cables_way
    lines_way_path = snakemake.input.lines_way
    routes_relation_path = snakemake.input.routes_relation
    substations_way_path = snakemake.input.substations_way
    substations_relation_path = snakemake.input.substations_relation

    # Get mapping of names to IDs
    mapper = OSMNameMapper(
        cables_way_path=cables_way_path,
        lines_way_path=lines_way_path,
        routes_relation_path=routes_relation_path,
        substations_way_path=substations_way_path,
        substations_relation_path=substations_relation_path,
    )

    # Access the DataFrame
    osm_mapping_df = mapper.name_to_id_map

    # Save to CSV
    osm_mapping_df.to_csv(snakemake.output.osm_mapping, index=False)

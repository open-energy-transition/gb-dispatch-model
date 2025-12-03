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
from scripts.clean_osm_data import _clean_voltage

logger = logging.getLogger(__name__)


class OSMNameMapper:
    def __init__(
        self, osm_files: dict[str, Path], build_files: dict[str, Path]
    ) -> None:
        """
        Initialize the OSMNameMapper with paths to OSM data files.

        Args:
            osm_files (dict): Dictionary mapping OSM feature types to file paths.
                Keys: 'cables_way', 'lines_way', 'routes_relation',
                      'substations_way', 'substations_relation'
            build_files (dict): Dictionary mapping build component types to file paths.
                Keys: 'lines', 'links', 'converters', 'transformers', 'substations'
        """
        self.osm_files = osm_files
        self.build_files = build_files

        # Store the combined DataFrame for direct access
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

    def _create_combined_df(self) -> pd.DataFrame:
        """
        Create a combined DataFrame from all OSM data files.

        Reads all OSM data files and creates a unified DataFrame with all entries.

        Returns:
            pd.DataFrame: Combined DataFrame with columns:
                - id: OSM ID
                - name: Feature name
                - voltage: Voltage level
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

            # Split cells with multiple values
            # combined_df = _split_cells(combined_df, ["voltage"])

            return combined_df
        else:
            raise ValueError("No data found in any OSM files")

    def get_raw_id(
        self, name: str, component_type: str, voltage: int | str = ""
    ) -> tuple[list[int], list[str]]:
        """
        Get OSM entries matching both name and component type.

        Args:
            name (str): The name to search for.
            component_type (str): The component type (e.g., 'cable', 'line', 'substation').
            voltage (int): The voltage level in kV to filter by.

        Returns:
            pd.DataFrame: DataFrame with entries matching both name and type.
        """
        # Filter by component type
        result = self.combined_df[self.combined_df["type"].str.contains(component_type)]

        # Filter by name by simply checking if the name is contained (case-insensitive)
        # TODO: Improve name matching if necessary with robust methods
        result = result[result["name"].str.lower().str.contains(name.lower())]

        # Filter by voltage only if provided
        if voltage is not None:
            result = result[result.voltage.str.contains(str(voltage))]

        if not result.empty:
            ids = result["id"].tolist()
            names = result["name"].tolist()

            if len(ids) > 1:
                voltage_info = f", voltage: {voltage}kV" if voltage is not None else ""
                logger.warning(
                    f"Multiple entries found for name: {name}, type: {component_type}{voltage_info}. IDs: {ids}, component names: {names}"
                )
            return ids, names
        else:
            voltage_info = f", voltage: {voltage}kV" if voltage is not None else ""
            logger.warning(
                f"No entries found for name: {name} and type: {component_type}{voltage_info}"
            )
            return [], []

    def get_network_id(self, raw_id: int, component_type: str) -> pd.Series:
        """
        Get the network component ID corresponding to a given OSM raw ID.

        Args:
            raw_id (int): The OSM raw ID.

        Returns:
            pd.Series: Series with network component IDs.
        """
        # This method would require access to the build files to map raw IDs to network IDs.
        # Implementation would depend on the structure of the build files.
        pass


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Create dictionaries from snakemake inputs
    osm_files = {
        "cables_way": snakemake.input.raw_cables_way,
        "lines_way": snakemake.input.raw_lines_way,
        "routes_relation": snakemake.input.raw_routes_relation,
        "substations_way": snakemake.input.raw_substations_way,
        "substations_relation": snakemake.input.raw_substations_relation,
    }

    build_files = {
        "lines": snakemake.input.build_lines,
        "links": snakemake.input.build_links,
        "converters": snakemake.input.build_converters,
        "transformers": snakemake.input.build_transformers,
        "substations": snakemake.input.build_substations,
    }

    # Get mapping of names to IDs
    mapper = OSMNameMapper(
        osm_files=osm_files,
        build_files=build_files,
    )

    # Access the DataFrame
    osm_mapping_df = mapper.combined_df

    # Get substation example
    substation_list = snakemake.config["noa_options"]["substations_list"]

    results = []

    for substation_data in substation_list:
        substation_data = [x.strip() for x in substation_data.split(",")]
        raw_ids, raw_names = mapper.get_raw_id(
            name=substation_data[0],
            component_type="substation",
            voltage=substation_data[2],
        )
        print(f"Results for substation: {substation_data[0]}")

        if not raw_ids:
            # Append entry for substations with no matches
            print(f"No matches found for: {substation_data[0]}")
            results.append(
                {
                    "substation_query": substation_data[0],
                    "name": None,
                    "id": None,
                    "voltage": substation_data[2],
                }
            )
        else:
            for name, raw_id in zip(raw_names, raw_ids):
                print(f"Name: {name}, OSM ID: {raw_id}, Voltage: {substation_data[2]}")
                results.append(
                    {
                        "substation_query": substation_data[0],
                        "name": name,
                        "id": raw_id,
                        "voltage": substation_data[2],
                    }
                )

    df = pd.DataFrame(results)

    # Save to CSV
    osm_mapping_df.to_csv(snakemake.output.osm_mapping, index=False)

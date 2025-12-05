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
import pypsa

from scripts._helpers import configure_logging, set_scenario_config
from scripts.clean_osm_data import _clean_voltage

logger = logging.getLogger(__name__)


class OSMNameMapper:
    def __init__(
        self,
        osm_files: dict[str, Path] | None = None,
        network_path: str | None = None,
        csv_path: str | None = None,
    ) -> None:
        """
        Initialize the OSMNameMapper with paths to OSM data files.

        Args:
            osm_files (dict): Dictionary mapping OSM feature types to file paths.
                Keys: 'cables_way', 'lines_way', 'routes_relation',
                      'substations_way', 'substations_relation'
            network_path (str): Path to the network file.
            csv_path (Path): Path to pre-generated OSM mapping CSV.
        """
        self.osm_files = osm_files
        self.network_path = network_path
        self.network = None
        if network_path is not None:
            self.network = self._load_network()

        # Convert csv_path to Path object if it's a string
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)

        # Load from CSV if provided
        if csv_path is not None and csv_path.exists():
            logger.info(f"Loading OSM mapping from CSV: {csv_path}")
            self.combined_df = pd.read_csv(csv_path)
        elif osm_files is not None:
            logger.info("Loading OSM data from raw files")
            # Store the combined DataFrame for direct access
            self.combined_df = self._create_combined_df()
        else:
            raise ValueError(
                "Either csv_path or both osm_files and build_files must be provided."
            )

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

    def _load_network(self) -> pypsa.Network:
        """Load the base PyPSA network from the specified path."""
        network = pypsa.Network(self.network_path)
        return network

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

    def _get_network_bus_from_raw_id(
        self, raw_id: str, voltage: int | str
    ) -> tuple[str | None, str]:
        """
        Find network bus that corresponds to a raw OSM ID.

        Strategy:
        1. Find all buses where raw_id appears in bus_id
        2. If multiple matches and voltage provided, filter by voltage suffix
        3. Return best match or None

        Args:
            raw_id: The raw OSM ID to search for (e.g., "way/123456")
            voltage: Optional voltage to use as tiebreaker (e.g., 400)

        Returns:
            Network bus ID and 'exists' status if found (with exact voltage match), else 'reference' status
            None, None otherwise
        """
        # Find all buses that contain raw_id
        matching_buses = [
            bus_id for bus_id in self.network.buses.index if str(raw_id) in bus_id
        ]

        # Apply voltage filtering if matches found
        if matching_buses:
            matching_buses_with_voltage = [
                bus_id for bus_id in matching_buses if bus_id.endswith(f"-{voltage}")
            ]
            if matching_buses_with_voltage:
                return matching_buses_with_voltage[0], "exists"
            else:
                return matching_buses[0], "reference"
        else:
            logger.debug(f"No contains match for raw_id: {raw_id}")
            return None, None

    def _find_substation_with_fallback(
        self, name: str, voltage: int | None = None
    ) -> tuple[str | None, str | None, str | None]:
        """
        Find substation using fallback matching strategies.

        Tries in order:
        1. Exact name + exact voltage
        2. Exact name + any voltage
        3. Contains name + exact voltage
        4. Contains name + any voltage

        Returns:
            tuple: (reference_bus_id, bus_status, raw_names) or (None, None, None) if not found
        """
        strategies = [
            ("exact", voltage, f"exact name + exact voltage ({voltage}kV)"),
            ("exact", "", "exact name + any voltage"),
            ("contains", voltage, f"contains name + exact voltage ({voltage}kV)"),
            ("contains", "", "contains name + any voltage"),
        ]  # We might need even more robust methods in the future for fallback

        for method, voltage_filter, description in strategies:
            logger.debug(f"Trying strategy: {description}")

            raw_ids, raw_names = self.get_raw_id(
                name=name,
                component_type="substation",
                method=method,
                voltage=voltage_filter,
            )

            if not raw_ids:
                continue

            # Try to find network bus for each raw_id
            for raw_id, raw_name in zip(raw_ids, raw_names):
                network_bus_id, bus_status = self._get_network_bus_from_raw_id(
                    raw_id, voltage
                )

                if network_bus_id:
                    logger.info(
                        f"Match found using: {description}\n"
                        f"OSM: {raw_name} (ID: {raw_id})\n"
                        f"Network bus: {network_bus_id} (status: {bus_status})"
                    )
                    return network_bus_id, bus_status, raw_name

            if raw_ids:  # This needs to be handled if such case happen
                logger.debug(f"Found {len(raw_ids)} OSM matches but none in network")

        return None, None, None

    def get_raw_id(
        self,
        name: str,
        component_type: str,
        method: str,
        voltage: int | str = "",
    ) -> tuple[list[int], list[str]]:
        """
        Get OSM entries matching both name and component type.

        Args:
            name (str): The name to search for.
            component_type (str): The component type (e.g., 'cable', 'line', 'substation').
            voltage (int): The voltage level in kV to filter by.
            method (str): The matching method to use ('exact', 'contains', 'robust', etc.).

        Returns:
            pd.DataFrame: DataFrame with entries matching both name and type.
        """
        # Filter by component type
        result = self.combined_df[self.combined_df["type"].str.contains(component_type)]

        # Filter by name by simply checking if the name is contained (case-insensitive)
        # TODO: Improve name matching if necessary with robust methods
        if method == "exact":
            result = result[result["name"].str.lower() == name.lower()]
        elif method == "contains":
            result = result[result["name"].str.lower().str.contains(name.lower())]
        else:
            logger.warning(
                "For now, only 'exact' and 'contains' methods are implemented for name matching."
            )

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

    # Get mapping of names to IDs
    mapper = OSMNameMapper(
        osm_files=osm_files,
        network_path=snakemake.input.network,
    )

    # Access the DataFrame
    osm_mapping_df = mapper.combined_df

    # Save to CSV
    osm_mapping_df.to_csv(snakemake.output.osm_mapping, index=False)

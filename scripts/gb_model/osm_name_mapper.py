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

import geopandas as gpd
import pandas as pd
import pypsa
from shapely import wkt
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import Point, Polygon

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_osm_network import BUS_TOL, DISTANCE_CRS, GEO_CRS
from scripts.clean_osm_data import _clean_voltage

logger = logging.getLogger(__name__)


class OSMNameMapper:
    def __init__(
        self,
        osm_files: dict[str, Path] | None = None,
        csv_path: str | None = None,
    ) -> None:
        """
        Initialize the OSMNameMapper with paths to OSM data files.

        Args:
            osm_files (dict): Dictionary mapping OSM feature types to file paths.
                Keys: 'cables_way', 'lines_way', 'routes_relation',
                      'substations_way', 'substations_relation'
            csv_path (Path): Path to pre-generated OSM mapping CSV.
        """
        self.osm_files = osm_files
        self.geo_crs = GEO_CRS
        self.distance_crs = DISTANCE_CRS

        # Convert csv_path to Path object if it's a string
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)

        # Load from CSV if provided
        if csv_path is not None and csv_path.exists():
            logger.info(f"Loading OSM mapping from CSV: {csv_path}")
            self.combined_df = pd.read_csv(csv_path)

            # Convert WKT strings back to Shapely geometries
            self.combined_df["geometry"] = self.combined_df["geometry"].apply(
                lambda x: wkt.loads(x) if pd.notna(x) else None
            )
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

        osm_data = json.load(file_path.read_text(encoding="utf-8"))

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

            # Split cells with multiple values
            # combined_df = _split_cells(combined_df, ["voltage"])

            return combined_df
        else:
            raise ValueError("No data found in any OSM files")

    def _get_substation_x_y(
        self, name: str, voltage: int, tol: float = BUS_TOL / 2
    ) -> pd.DataFrame:
        """
        Get the x and y coordinates of substations using polylabel on their geometries.

        Args:
            name (str): The name of the substation.
            voltage (int): The voltage level in kV.
            tol (float): Tolerance for polylabel calculation.

        Returns:
            pd.DataFrame: DataFrame with columns 'id', 'x', 'y' for substations.
        """
        # Filter for substations
        substations_df = self.combined_df[
            self.combined_df["type"].str.contains("substations")
        ].copy()

        # Select the substations by name
        substation_group = substations_df[
            substations_df["name"].str.lower() == name.lower()
        ]
        if substation_group.empty:
            raise ValueError(f"Substation '{name}' not found in OSM data.")

        # Try to find exact voltage match
        substation_exact_voltage = substation_group[
            substation_group["voltage"].str.contains(str(voltage))
        ]
        if not substation_exact_voltage.empty:
            substation = substation_exact_voltage
            substation_status = "exists"
        else:
            substation = substation_group.iloc[[0]]  # select first match
            substation_status = "reference"

        # Ensure we have exactly one match
        if substation.empty:
            raise ValueError(f"Substation '{name}' not found in OSM data.")
        elif len(substation) > 1:
            raise ValueError(
                f"Multiple substations found for name: {name} and voltage: {voltage}kV. IDs: {substation['id'].tolist()}"
            )

        # Get single substation row
        row = substation.iloc[0]

        # Raise error if geometry is missing
        if row["geometry"] is None:
            raise ValueError(
                f"No geometry found for substation '{name}' (ID: {row['id']})."
            )

        # Apply polylabel to get the pole of inaccessibility
        point = polylabel(row["geometry"], tol)
        x = point.x
        y = point.y

        return row["id"], x, y, substation_status

    def _get_closest_bus(
        self,
        network: pypsa.Network,
        x: float,
        y: float,
        voltage: int,
        tol: float = BUS_TOL,
    ) -> str | None:
        """
        Find the closest bus in the network to the given x, y coordinates.

        Args:
            network: The PyPSA network to search in.
            x: The x coordinate to search for.
            y: The y coordinate to search for.
            tol: Tolerance for coordinate matching.
            voltage: Voltage level to filter by.

        Returns:
            The closest_bus_id of the closest buses within tolerance, raise error if none found.
        """
        buses = network.buses.copy()

        # Create GeoDataFrame of OSM CRS
        buses_gdf = gpd.GeoDataFrame(
            buses,
            geometry=gpd.points_from_xy(buses["x"], buses["y"]),
            crs=self.geo_crs,
        )

        # Project to distance CRS for accurate distance calculations
        buses_projected = buses_gdf.to_crs(self.distance_crs)

        # Create target point and project it
        target_point = gpd.GeoSeries([Point(x, y)], crs=self.geo_crs).to_crs(
            self.distance_crs
        )[0]

        # Calculate distance in meters
        buses_projected["distance_m"] = buses_projected.geometry.distance(target_point)

        # Filter by tolerance
        nearby_buses = buses_projected[buses_projected["distance_m"] <= tol]

        if nearby_buses.empty:
            raise ValueError(f"No buses found within {tol}m of point ({x}, {y}).")

        # Filter by voltage
        nearby_bus_exact_voltage = nearby_buses[nearby_buses["v_nom"] == voltage]

        if not nearby_bus_exact_voltage.empty:
            closest_bus_id = nearby_bus_exact_voltage["distance_m"].idxmin()
            bus_status = "exists"
        else:
            closest_bus_id = nearby_buses["distance_m"].idxmin()
            bus_status = "reference"

        return closest_bus_id, bus_status

    def _get_network_bus_id(
        self, network: pypsa.Network, name: str, voltage: int, tol: float = BUS_TOL
    ) -> str | None:
        """
        Find network bus that corresponds to given name.

        Args:
            network: The PyPSA network to search in.
            name: The name of the substation.
            voltage: Voltage level to filter by.
            tol: Tolerance for coordinate matching.
        """
        # Get raw ID, x, y from OSM data
        raw_id, x, y, substation_status = self._get_substation_x_y(name, voltage)

        # Find closest buses in network within tolerance
        network_bus_id, bus_status = self._get_closest_bus(
            network=network, x=x, y=y, voltage=voltage, tol=tol
        )

        return network_bus_id, bus_status


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
    mapper = OSMNameMapper(osm_files=osm_files)

    # Access the DataFrame
    osm_mapping_df = mapper.combined_df

    # Save to CSV
    osm_mapping_df.to_csv(snakemake.output.osm_mapping, index=False)

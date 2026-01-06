# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
NOA options adder.

Adds NOA options to the GB model.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pypsa
from shapely import wkt
from shapely.algorithms.polylabel import polylabel
from shapely.geometry import Point

from scripts._helpers import configure_logging, set_scenario_config
from scripts.build_osm_network import BUS_TOL, DISTANCE_CRS, GEO_CRS

logger = logging.getLogger(__name__)


class NetworkBusMapper:
    """Class to map OSM names to network bus IDs."""

    def __init__(self, csv_path: Path):
        self.geo_crs = GEO_CRS
        self.distance_crs = DISTANCE_CRS
        self.csv_path = csv_path
        self.osm_mapping = self._read_osm_mapping_csv()

    def _read_osm_mapping_csv(self) -> pd.DataFrame:
        """
        Read OSM mapping from CSV file.

        Args:
            csv_path (Path): Path to the CSV file.

        Returns:
            pd.DataFrame: DataFrame with OSM mapping data.
        """
        # Read the CSV file
        osm_mapping = pd.read_csv(self.csv_path)

        # Convert WKT strings back to Shapely geometries
        osm_mapping["geometry"] = osm_mapping["geometry"].apply(
            lambda x: wkt.loads(x) if pd.notna(x) else None
        )
        return osm_mapping

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
        substations_df = self.osm_mapping[
            self.osm_mapping["type"].str.contains("substations")
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


class AddNOAOption:
    """Class to add NOA option to the network."""

    def __init__(
        self,
        network: pypsa.Network,
        noa_option: str,
        noa_options_config: dict,
        mapper: NetworkBusMapper,
    ):
        self.network = network
        self.noa_option = noa_option
        self.noa_options_config = noa_options_config
        self.mapper = mapper
        self._bus_lookup_cache = {}
        self.add_option()

    def _get_noa_option(self) -> dict:
        """Get NOA option details from config."""
        for option in self.noa_options_config:
            if option.get("id") == self.noa_option:
                return option
        raise ValueError(f"NOA option {self.noa_option} not found in config.")

    def _execute_operation(self, operation: dict) -> None:
        """
        Execute a single operation by dispatching to the appropriate method.

        Dispatches based on action and component_type:
        - action='add', component_type='substation' -> _add_substation()
        - action='add', component_type='line' -> _add_line()
        - action='update', component_type='line' -> _update_component()
        - action='remove', component_type='line' -> _remove_component()
        """
        action = operation["action"]
        component_type = operation["component_type"]

        # Dispatch based on action and component_type
        if action == "add":
            if component_type == "substation":
                self._add_substation(operation)
            elif component_type == "line":
                self._add_line(operation)
            elif component_type == "link":
                self._add_link(operation)
            else:
                raise KeyError(f"Unknown component_type for add: {component_type}")

        elif action == "update":
            if component_type == "line":
                self._update_line(operation)
            elif component_type == "link":
                self._update_link(operation)
            else:
                raise KeyError(f"Unknown component_type for update: {component_type}")

        elif action == "remove":
            self._remove_component(operation)

        else:
            raise KeyError(f"Unknown action: {action}")

    def _add_substation(self, operation: dict) -> None:
        """Add substation to the network."""
        substation_names = operation.get("names", [])
        voltage = operation["voltage"]
        carrier = operation["carrier"]

        for substation_name in substation_names:
            # Determine bus closest to the substation
            network_bus_id, bus_status = self._find_bus_cached(
                name=substation_name,
                voltage=voltage,
            )

            if bus_status == "exists":
                logger.info(
                    f"Substation '{substation_name}' with voltage '{voltage}'kV already exists in the network."
                )
            elif network_bus_id and bus_status == "reference":
                # Get attributes from reference bus
                ref_attrs = self.network.buses.loc[network_bus_id].to_dict()
                ref_attrs.update({"v_nom": voltage, "carrier": carrier})
                new_bus_id = f"{network_bus_id.split('-')[0]}-{voltage}"
                # Add new bus to the network
                self.network.add("Bus", new_bus_id, **ref_attrs)

                # Update cache with new bus
                self._bus_lookup_cache[(substation_name, voltage)] = (
                    new_bus_id,
                    "exists",
                )

                logger.info(
                    f"Substation '{substation_name}' with voltage '{voltage}'kV added to the network."
                )
            elif not network_bus_id:
                raise ValueError(
                    f"Cannot add substation '{substation_name}' with voltage '{voltage}'kV using any strategy."
                )

    def _find_bus_cached(
        self, name: str, voltage: int
    ) -> tuple[str | None, str | None, str | None]:
        """
        Find bus with caching to avoid duplicate lookups.

        Args:
            name: Substation name
            voltage: Voltage level in kV

        Returns:
            Tuple of (bus_id, bus_status, raw_name)
        """
        # Create cache key
        cache_key = (name, voltage)

        # Check cache first
        if cache_key in self._bus_lookup_cache:
            logger.debug(f"Using cached lookup for '{name}' at {voltage}kV")
            return self._bus_lookup_cache[cache_key]

        # Perform lookup
        result = self.mapper._get_network_bus_id(
            network=self.network,
            name=name,
            voltage=voltage,
        )

        # Cache the result
        self._bus_lookup_cache[cache_key] = result

        return result

    def _calculate_s_nom(self, line_type: str, voltage: float, circuits: int) -> float:
        """Calculate nominal power (s_nom) based on voltage and number of circuits."""
        s_nom = (
            np.sqrt(3)
            * self.network.line_types.loc[line_type, "i_nom"]
            * voltage
            * circuits
        )
        return s_nom

    def _get_line_parameters(
        self,
        voltage: int,
        from_name: str,
        to_name: str,
        carrier: str,
        length: float | None = None,
        circuits: int = 1,
        capacity: float | None = None,
    ) -> dict:
        """
        Get line parameters for adding to network.

        Args:
            voltage: Voltage level in kV
            from_name: Start substation name
            to_name: End substation name
            carrier: Line carrier
            length: Line length
            circuits: Number of circuits
            capacity: Optional fixed capacity (overrides calculation)

        Returns:
            Dictionary of line parameters ready for network.add()
        """
        line_type = snakemake.config["lines"]["types"][voltage]

        # Calculate s_nom if capacity not provided
        if capacity is not None:
            s_nom = capacity
        else:
            s_nom = self._calculate_s_nom(line_type, voltage, circuits)

        # Find buses
        from_bus, _ = self._find_bus_cached(name=from_name, voltage=voltage)
        to_bus, _ = self._find_bus_cached(name=to_name, voltage=voltage)

        return {
            "bus0": from_bus,
            "bus1": to_bus,
            "length": length,
            "carrier": carrier,
            "type": line_type,
            "v_nom": voltage,
            "s_nom": s_nom,
            "num_parallel": circuits,
            "dc": False,
            "underground": False,
            "onshore_bus": True,
        }

    def _add_line(self, operation: dict) -> None:
        """Add line to the network."""
        line_name = operation["name"]
        voltage = operation["voltage"]

        # Get line parameters using helper
        line_params = self._get_line_parameters(
            voltage=voltage,
            from_name=operation["from"],
            to_name=operation["to"],
            carrier=operation["carrier"],
            length=operation.get("length", 0.0),
            circuits=operation.get("circuits", 1),
            capacity=operation.get("capacity", None),
        )

        # Add new line
        self.network.add("Line", line_name, **line_params)

        logger.info(
            f"Line '{line_name}' was added between '{line_params['bus0']}' and "
            f"'{line_params['bus1']}' at voltage '{voltage}'kV with s_nom={line_params['s_nom']:.1f}MW"
        )

    def _update_line(self, operation: dict) -> None:
        """
        Update line capacity by adding parallel line with povided capacity,
        or capacity difference between initial and final line types.
        """
        line_name = operation["name"]
        from_voltage = operation["from_voltage"]
        to_voltage = operation["to_voltage"]
        circuits = operation.get("circuits", 1)
        capacity = operation.get("capacity", None)

        # Calculate s_nom difference if capacity not provided
        if capacity is not None:
            s_nom = capacity
        else:
            from_line_type = snakemake.config["lines"]["types"][from_voltage]
            to_line_type = snakemake.config["lines"]["types"][to_voltage]

            s_nom_from = self._calculate_s_nom(from_line_type, from_voltage, circuits)
            s_nom_to = self._calculate_s_nom(to_line_type, to_voltage, circuits)
            s_nom = s_nom_to - s_nom_from

        # Get updated line parameters using helper
        line_params = self._get_line_parameters(
            voltage=to_voltage,
            from_name=operation["from"],
            to_name=operation["to"],
            carrier=operation["carrier"],
            length=operation.get("length", 0.0),
            circuits=circuits,
            capacity=s_nom,
        )

        # Add the parallel upgrade line
        self.network.add("Line", line_name, **line_params)

        logger.info(
            f"Line upgrade '{line_name}' added: {from_voltage}kV → {to_voltage}kV, "
            f"capacity difference: {s_nom:.1f}MW"
        )

    def add_option(self):
        """Add NOA option to the network."""
        option_details = self._get_noa_option()
        for sub_option in option_details.get("details", []):
            sub_option_name = sub_option["name"]
            logger.info(f"Applying NOA sub-option: {sub_option_name}")
            pypsa_operations = sub_option.get("pypsa_operations", [])
            for operation in pypsa_operations:
                # Execute the operation
                self._execute_operation(operation)


def add_noa_options(
    network_path: str,
    noa_options_config: dict,
    noa_sets_config: dict,
    noa_sets_selected: list[int],
    mapper: NetworkBusMapper,
    output_network_path: str,
) -> None:
    """
    Add NOA options to the GB model network.

    Args:
        network_path (Path): Path to the GB model network file.
        noa_options_config (dict): Configuration for NOA options.
        noa_sets_config (dict): Configuration for NOA sets.
        noa_sets_selected (list[int]): List of NOA set IDs to apply.
        mapper (NetworkBusMapper): Network bus mapper instance.
        output_network_path (Path): Path to save the updated network.

    Returns:
        None
    """
    # Load the network
    network = pypsa.Network(network_path)

    # Create NOA sets dictionary
    noa_sets_dict = {s["id"]: s for s in noa_sets_config}

    for noa_set_id in noa_sets_selected:
        noa_set = noa_sets_dict.get(noa_set_id)

        if not noa_set:
            logger.warning(f"NOA set ID '{noa_set_id}' not found in config. Skipping.")
            continue

        set_name = noa_set["name"]
        options = noa_set.get("noa_options", [])

        logger.info(f"\nApplying NOA set: '{set_name}' (options: {options})")

        for option in options:
            # Add NOA option to the network
            AddNOAOption(network, option, noa_options_config, mapper)

    # Save the updated network
    network.export_to_netcdf(output_network_path)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the input
    network_path = snakemake.input.network
    osm_mapping_path = snakemake.input.osm_mapping_csv
    output_network_path = snakemake.output.network

    # Load params
    noa_options_config = snakemake.params.noa_options
    noa_sets_config = snakemake.params.noa_sets
    noa_sets_selected = snakemake.params.noa_sets_selected

    # Create network bus mapper
    mapper = NetworkBusMapper(csv_path=osm_mapping_path)

    # Add NOA option
    add_noa_options(
        network_path=network_path,
        noa_options_config=noa_options_config,
        noa_sets_config=noa_sets_config,
        noa_sets_selected=noa_sets_selected,
        mapper=mapper,
        output_network_path=output_network_path,
    )

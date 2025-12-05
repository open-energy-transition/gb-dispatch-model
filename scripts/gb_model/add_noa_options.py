# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
NOA options adder.

Adds NOA options to the GB model.
"""

import logging
from pathlib import Path

import pypsa

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model.osm_name_mapper import OSMNameMapper

logger = logging.getLogger(__name__)


class AddNOAOption:
    """Class to add NOA option to the network."""

    def __init__(
        self,
        network: pypsa.Network,
        noa_option: str,
        noa_options_config: dict,
        mapper: OSMNameMapper,
    ):
        self.network = network
        self.noa_option = noa_option
        self.noa_options_config = noa_options_config
        self.mapper = mapper
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
        action = operation.get("action")
        component_type = operation.get("component_type")

        # Dispatch based on action and component_type
        if action == "add":
            if component_type == "substation":
                self._add_substation(operation)
            elif component_type in ["line", "link"]:
                self._add_line(self, operation)
            else:
                logger.warning(f"Unknown component_type for add: {component_type}")

        elif action == "update":
            self._update_component(self, operation)

        elif action == "remove":
            self._remove_component(self, operation)

        else:
            logger.warning(f"Unknown action: {action}")

    def _find_network_bus_from_raw_id(
        self, raw_id: str, voltage: int | str
    ) -> str | None:
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
        # Bus contains raw_id
        matching_buses = [
            bus_id for bus_id in self.network.buses.index if str(raw_id) in bus_id
        ]

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
    ) -> tuple[list, list, str] | tuple[None, None, None]:
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

            raw_ids, raw_names = self.mapper.get_raw_id(
                name=name,
                component_type="substation",
                method=method,
                voltage=voltage_filter,
            )

            if not raw_ids:
                continue

            # Try to find network bus for each raw_id
            for raw_id, raw_name in zip(raw_ids, raw_names):
                network_bus_id, bus_status = self._find_network_bus_from_raw_id(
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

    def _add_substation(self, operation: dict) -> None:
        """Add substation to the network."""
        substation_name = operation.get("name")
        voltage = operation.get("voltage")
        carrier = operation.get("carrier")

        # Use fallback matching to find substation
        reference_bus_id, bus_status, raw_name = self._find_substation_with_fallback(
            name=substation_name,
            voltage=voltage,
        )

        if reference_bus_id is None and bus_status == "exists":
            logger.info(
                f"Substation '{substation_name}' with voltage '{voltage}'kV already exists in the network."
            )
            return
        elif reference_bus_id is None and bus_status == "reference":
            self.network.add(
                "Bus",
                f"{reference_bus_id.split('-')[0]}-{voltage}",
                v_nom=voltage,
                carrier=carrier,
                x=self.network.buses.at[reference_bus_id, "x"],
                y=self.network.buses.at[reference_bus_id, "y"],
                country=self.network.buses.at[reference_bus_id, "country"],
                geometry=self.network.buses.at[reference_bus_id, "geometry"],
            )
            logger.info(
                f"Substation '{substation_name}' with voltage '{voltage}'kV added to the network."
            )
            return

        if not reference_bus_id:
            logger.error(
                f"Cannot add substation '{substation_name}' with voltage '{voltage}'kV using any strategy."
            )
            return

    def add_option(self):
        """Add NOA option to the network."""
        option_details = self._get_noa_option()
        for sub_option in option_details.get("details", []):
            sub_option_name = sub_option.get("name")
            logger.info(f"Applying NOA sub-option: {sub_option_name}")
            pypsa_operations = sub_option.get("pypsa_operations", [])
            for operation in pypsa_operations:
                # Execute the operation
                self._execute_operation(operation)


def add_noa_options(
    network_path: str,
    noa_options_config: dict,
    noa_sets_config: dict,
    mapper: OSMNameMapper,
) -> None:
    """
    Add NOA options to the GB model network.

    Args:
        network_path (Path): Path to the GB model network file.
        noa_options_config (dict): Configuration for NOA options.

    Returns:
        None
    """
    # Load the network
    network = pypsa.Network(network_path)

    for noa_set in noa_sets_config:
        set_name = noa_set["name"]
        options = noa_set.get("noa_options", [])

        logger.info(f"\nApplying NOA set: '{set_name}' ({len(options)} options)")

        for option in options:
            # Add NOA option to the network
            AddNOAOption(network, option, noa_options_config, mapper)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            Path(__file__).stem, configfiles="config/config.noa.sets.yaml"
        )
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the base network
    network_path = snakemake.input.network
    osm_mapping_path = snakemake.input.osm_mapping_csv

    # Load params
    noa_options_config = snakemake.params.noa_options
    noa_sets_config = snakemake.params.noa_sets

    # Create OSM name mapper
    mapper = OSMNameMapper(csv_path=osm_mapping_path)

    # Add NOA option
    add_noa_options(
        network_path=network_path,
        noa_options_config=noa_options_config,
        noa_sets_config=noa_sets_config,
        mapper=mapper,
    )

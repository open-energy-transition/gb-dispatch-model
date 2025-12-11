# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
NOA options adder.

Adds NOA options to the GB model.
"""

import logging
from pathlib import Path

import numpy as np
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
        action = operation.get("action")
        component_type = operation.get("component_type")

        # Dispatch based on action and component_type
        if action == "add":
            if component_type == "substation":
                self._add_substation(operation)
            elif component_type in ["line", "link"]:
                self._add_line(operation)
            else:
                logger.warning(f"Unknown component_type for add: {component_type}")

        elif action == "update":
            self._update_component(self, operation)

        elif action == "remove":
            self._remove_component(self, operation)

        else:
            logger.warning(f"Unknown action: {action}")

    def _add_substation(self, operation: dict) -> None:
        """Add substation to the network."""
        substation_name = operation.get("name")
        voltage = operation.get("voltage")
        carrier = operation.get("carrier")

        # Determine bus closest to the substation
        network_bus_id, bus_status = self._find_bus_cached(
            name=substation_name,
            voltage=voltage,
        )

        if bus_status == "exists":
            logger.info(
                f"Substation '{substation_name}' with voltage '{voltage}'kV already exists in the network."
            )
            return

        if network_bus_id and bus_status == "reference":
            # Get attributes from reference bus
            ref_attrs = self.network.buses.loc[network_bus_id].to_dict()
            ref_attrs.update({"v_nom": voltage, "carrier": carrier})
            new_bus_id = f"{network_bus_id.split('-')[0]}-{voltage}"
            # Add new bus to the network
            self.network.add("Bus", new_bus_id, **ref_attrs)

            # Update cache with new bus
            self._bus_lookup_cache[(substation_name, voltage)] = (new_bus_id, "exists")

            logger.info(
                f"Substation '{substation_name}' with voltage '{voltage}'kV added to the network."
            )
            return

        if not network_bus_id:
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

    def _add_line(self, operation: dict) -> None:
        """Add line to the network."""
        line_name = operation.get("name")
        voltage = operation.get("voltage")
        carrier = operation.get("carrier")
        circuits = operation.get("circuits")
        line_type = snakemake.config["lines"]["types"][voltage]

        from_bus, bus_status_from = self._find_bus_cached(
            name=operation.get("from"), voltage=voltage
        )
        to_bus, bus_status_to = self._find_bus_cached(
            name=operation.get("to"), voltage=voltage
        )

        # Add new line
        self.network.add(
            "Line",
            line_name,
            bus0=from_bus,
            bus1=to_bus,
            length=operation.get("length"),
            carrier=carrier,
            type=snakemake.config["lines"]["types"][voltage],
            v_nom=voltage,
            s_nom=self._calculate_s_nom(line_type, voltage, circuits),
            num_parallel=circuits,
            dc=False,
            underground=False,
            onshore_bus=True,
        )
        logger.info(
            f"Line '{line_name}' between '{from_bus}' and '{to_bus}' at voltage '{voltage}'kV added to the network."
        )

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
    noa_sets_selected: list[int],
    mapper: OSMNameMapper,
    output_network_path: str,
) -> None:
    """
    Add NOA options to the GB model network.

    Args:
        network_path (Path): Path to the GB model network file.
        noa_options_config (dict): Configuration for NOA options.
        noa_sets_config (dict): Configuration for NOA sets.
        noa_sets_selected (list[int]): List of NOA set IDs to apply.
        mapper (OSMNameMapper): OSM name mapper instance.
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

        set_name = noa_set.get("name")
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

        snakemake = mock_snakemake(
            Path(__file__).stem, configfiles="config/config.noa.sets.yaml"
        )
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

    # Create OSM name mapper
    mapper = OSMNameMapper(csv_path=osm_mapping_path)

    # Add NOA option
    add_noa_options(
        network_path=network_path,
        noa_options_config=noa_options_config,
        noa_sets_config=noa_sets_config,
        noa_sets_selected=noa_sets_selected,
        mapper=mapper,
        output_network_path=output_network_path,
    )

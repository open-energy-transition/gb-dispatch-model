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

        # Use fallback matching to find substation
        reference_bus_id, bus_status, raw_name = (
            self.mapper._find_substation_with_fallback(
                network=self.network,
                name=substation_name,
                voltage=voltage,
            )
        )

        if reference_bus_id is not None and bus_status == "exists":
            logger.info(
                f"Substation '{substation_name}' with voltage '{voltage}'kV already exists in the network."
            )
            return
        elif reference_bus_id is not None and bus_status == "reference":
            self.network.add(
                "Bus",
                f"{reference_bus_id.split('-')[0]}-{voltage}",
                v_nom=voltage,
                carrier=carrier,
                x=self.network.buses.at[reference_bus_id, "x"],
                y=self.network.buses.at[reference_bus_id, "y"],
                country=self.network.buses.at[reference_bus_id, "country"],
                geometry=self.network.buses.at[reference_bus_id, "geometry"],
                tags=self.network.buses.at[reference_bus_id, "tags"],
                symbol=self.network.buses.at[reference_bus_id, "symbol"],
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

        from_bus, bus_status_from, _ = self.mapper._find_substation_with_fallback(
            network=self.network, name=operation.get("from"), voltage=voltage
        )
        to_bus, bus_status_to, _ = self.mapper._find_substation_with_fallback(
            network=self.network, name=operation.get("to"), voltage=voltage
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
    output_network_path: str,
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

    # Create OSM name mapper
    mapper = OSMNameMapper(csv_path=osm_mapping_path)

    # Add NOA option
    add_noa_options(
        network_path=network_path,
        noa_options_config=noa_options_config,
        noa_sets_config=noa_sets_config,
        mapper=mapper,
        output_network_path=output_network_path,
    )

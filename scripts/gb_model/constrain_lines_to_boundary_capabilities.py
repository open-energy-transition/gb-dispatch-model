# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Scale the boundary capabilities in a GB model network to match ETYS capacities.
"""

import logging
from pathlib import Path

import pandas as pd
import pypsa

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def _get_lines(lines: pd.DataFrame, bus1: str | int, bus0: str | int) -> pd.Series:
    """
    Get a boolean mask of lines connecting bus0 and bus1.
    This accounts for cases where bus0 and bus1 may be swapped.
    """
    all_lines = set(lines["bus0"]).union(set(lines["bus1"]))
    for bus in (bus0, bus1):
        if f"GB {bus}" not in all_lines:
            logger.warning(f"Bus 'GB {bus}' not found in network lines")
    return ((lines["bus0"] == f"GB {bus0}") & (lines["bus1"] == f"GB {bus1}")) | (
        (lines["bus0"] == f"GB {bus1}") & (lines["bus1"] == f"GB {bus0}")
    )


def set_boundary_constraints(
    lines: pd.DataFrame,
    etys_boundaries: dict[str, list[dict[str, str]]],
    etys_capacities: pd.DataFrame,
) -> None:
    """
    Limit total line flows across each boundary to match ETYS capacities.

    Args:
        lines (pd.DataFrame): PyPSA network lines DataFrame
        etys_boundaries (dict[str, list[dict[str, str]]]): PyPSA bus to ETYS boundaries mapping
    """
    s_noms: dict[str, list[float]] = {}
    for boundary, bus_groups in etys_boundaries.items():
        s_noms[boundary] = []
        boundary_lines = pd.Series(False, index=lines.index)
        for buses in bus_groups:
            lines_mask = _get_lines(lines, buses["bus0"], buses["bus1"])
            if not lines_mask.any():
                logger.warning(
                    f"No lines found for boundary '{boundary}' between "
                    f"buses '{buses['bus0']}' and '{buses['bus1']}'"
                )
            boundary_lines = boundary_lines | lines_mask
        etys_capacities.loc[boundary, "capability_mw"]
        # Set flow constraint here.


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    network = pypsa.Network(snakemake.input.network)
    etys_boundaries = snakemake.params.etys_boundaries_to_lines
    etys_caps = pd.read_csv(snakemake.input.etys_caps, index_col="boundary_name")

    set_boundary_constraints(network.lines, etys_boundaries, etys_caps)

    for line in snakemake.params.prune_lines:
        mask = _get_lines(network.lines, line["bus0"], line["bus1"])
        if mask.any():
            network.lines.loc[mask, "s_max_pu"] = 0
            logger.info(
                f"Pruned line between bus {line['bus0']} and bus {line['bus1']}"
            )
        else:
            logger.warning(
                f"No line found to prune between bus {line['bus0']} and bus {line['bus1']}"
            )
    network.export_to_netcdf(snakemake.output.network)

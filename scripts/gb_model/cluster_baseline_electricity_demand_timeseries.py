# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""
Base electricity demand timeseries clustering.

This script clusters the regional base electricity demand by buses and saves it in CSV file format.
"""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from scripts._helpers import PYPSA_V1, configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, clusters="clustered")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the files
    load_fn = snakemake.input.load
    busmap_fn = snakemake.input.busmap

    # Load scaling factor
    scaling = snakemake.params.scaling_factor

    # Read the electricity demand base .nc file
    load = (
        xr.open_dataarray(load_fn).to_dataframe().squeeze(axis=1).unstack(level="time")
    )

    # apply clustering busmap
    logger.info("Clustering the base electricity demand using busmap")
    busmap = pd.read_csv(busmap_fn, dtype=str)
    index_col = "name" if PYPSA_V1 else "Bus"
    busmap = busmap.set_index(index_col).squeeze()
    load = load.groupby(busmap).sum().T

    logger.info(f"Load data scaled by factor {scaling}.")
    load *= scaling

    # Save the regional base electricity demand profiles
    load.to_csv(snakemake.output.csv_file)
    logger.info(
        f"Base electricity demand dataframe saved to {snakemake.output.csv_file}"
    )

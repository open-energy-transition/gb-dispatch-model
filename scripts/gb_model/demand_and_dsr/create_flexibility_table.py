# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Flexibility data processor.

This script processes required flexibility data from the FES workbook.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_scenario_name, parse_flexibility_data

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, flexibility_type="fes_ev_dsm")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the regional gb data file path
    flexibility_data = pd.read_csv(snakemake.input.flexibility_sheet)

    # Parse input data
    fes_scenario = get_scenario_name(snakemake)
    year_range = snakemake.params.year_range
    carrier_mapping = snakemake.params.carrier_mapping

    df_flexibility = parse_flexibility_data(
        flexibility_data, fes_scenario, year_range, carrier_mapping
    )

    df_flexibility = (df_flexibility * 1000).to_frame("p_nom")  # Convert GW to MW

    # Write flexibility dataframe to csv file
    df_flexibility.to_csv(snakemake.output["flexibility"])
    logger.info(f"Flexibility data saved to {snakemake.output['flexibility']}")

# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
EV V2G storage data processor.

This script multiplies V2G capacity data by a specified multiplier to synthesise V2G storage capacity.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    capacity_data = pd.read_csv(snakemake.input.v2g_cap, index_col=["bus", "year"])

    synthesised_v2g_data = capacity_data.abs() * snakemake.params.v2g_multiplier

    synthesised_v2g_data.squeeze().to_frame("e_nom").to_csv(snakemake.output.csv)

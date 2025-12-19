# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Calculate net H2 demand.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, data_type="unmanaged_charging")
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    demand = pd.read_csv(snakemake.input.demand, index_col="year")
    fixed_supply = pd.read_csv(snakemake.input.fixed_supply, index_col="year")

    net_demand = (demand.sum(axis=1) - fixed_supply.sum(axis=1)).to_frame("p_set")
    net_demand.to_csv(snakemake.output.csv)

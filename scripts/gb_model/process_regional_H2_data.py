# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Regional H2 data processor.

This script prepares regional disaggregation of H2 data.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import get_regional_distribution

logger = logging.getLogger(__name__)


def prepare_regional_data(
    data_to_regionalise: pd.DataFrame, regional_reference_data: pd.Series
) -> pd.DataFrame:
    """
    Prepare regional disaggregation of H2 data using reference data distribution patterns.

    Args:
        data_to_regionalise (pd.DataFrame): DataFrame containing annual aggregated H2 data indexed by year
        regional_reference_data (pd.Series): Series containing regional data with MultiIndex [bus, year]

    Returns:
        pd.Series:
            Series with MultiIndex [bus, year] containing regionally distributed H2 data.
            Each region gets a proportional share of the input data based on reference distribution patterns.
    """
    # Calculate regional distribution of grid-connected electrolysis capacities
    electrolysis_distribution = get_regional_distribution(regional_reference_data)

    # Apply regional distribution to electricity demand data
    regionalised = data_to_regionalise.mul(electrolysis_distribution)
    assert np.isclose(regionalised.sum(), data_to_regionalise.sum()), (
        "Regional H2 data disaggregation error: totals do not match"
    )

    return regionalised


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, data_type="unmanaged_charging")
    configure_logging(snakemake)
    set_scenario_config(snakemake)
    to_regionalise = pd.read_csv(snakemake.input.to_regionalise, index_col="year")
    regional_distribution = pd.read_csv(
        snakemake.input.regional_distribution, index_col=["bus", "year"]
    ).squeeze()
    # Prepare regional H2 data

    regional_data = prepare_regional_data(
        to_regionalise.squeeze(),
        regional_distribution,
    ).to_frame(snakemake.params.param_name)

    regional_data.to_csv(snakemake.output.csv)

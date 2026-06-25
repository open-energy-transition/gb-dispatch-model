# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Map Elexon Balancing mechanism units to a carrier
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def map_carrier_bm_units(
    bm_units_path: str,
    technology_mapping: dict[str,str]
) -> pd.DataFrame:
    """
    Get mapping of BM units to fuel types (carriers)

    Parameters
    ----------
    technology_mapping: dict[str, str]
        Map Elexon carrier types to PyPSA carrier types
    bm_units_path: str
        CSV path for BMunits
    """

    df_bmu=pd.read_csv(bm_units_path)

    # Map BM unit fuel types to PyPSA carriers
    df_bmu["carrier"] = df_bmu["fuelType"].map(technology_mapping)

    # Filter to only include BM units relevant to conventional technologies in the model
    df_bmu_filtered = df_bmu.loc[df_bmu["carrier"].isin(technology_mapping.values())]

    # Dictionary to map BM units to carriers
    bmu_carrier_map = dict(df_bmu_filtered[["nationalGridBmUnit", "carrier"]].values)

    logger.info(f"Created BM unit mapping to carriers {technology_mapping.keys()}.")

    return bmu_carrier_map


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)


    bmu_carrier_map = map_carrier_bm_units(
        bm_units_path=snakemake.input.bm_units_path,
        technology_mapping=snakemake.params.technology_mapping,
    )

    df_bod=pd.read_csv(snakemake.input.bod_path)
    df_bod["carrier"] = df_bod["nationalGridBmUnit"].map(bmu_carrier_map)


    df_bod.to_csv(snakemake.output.csv)
    logger.info(
        f"Exported carrier mapped bids/offer data to {snakemake.output.csv}"
    )

# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
FES worksheet extractor.

This is a generalised script to extract tables based on their manually configured positions in non-machine-readable Excel sheets.
"""

import logging
from pathlib import Path

import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def extract_fes_worksheet(
    excel_path: str, sheet_name: str, sheet_config: dict
) -> pd.DataFrame:
    """
    Extract FES worksheet data from an Excel file.

    Args:
        excel_path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet to extract.
        sheet_config (dict): Configuration for the sheet extraction.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted FES worksheet data.
    """
    renamers = sheet_config.pop("rename", {})
    fes_data = pd.read_excel(excel_path, sheet_name=sheet_name, **sheet_config)

    fes_data = fes_data.rename_axis(**renamers)

    if pd.notnull(sheet_config["header"]):
        fes_data = fes_data.stack(sheet_config["header"])

    if not (unnamed_data := fes_data.filter(regex="Unnamed")).empty:
        logger.error(
            "The extracted DataFrame contains unnamed columns/rows, please check the configuration."
            f"First rows of unnamed data:\n{unnamed_data.head()}"
        )
        raise
    return fes_data.to_frame("data")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    availability_df = extract_fes_worksheet(
        snakemake.input.workbook,
        snakemake.wildcards.fes_sheet,
        snakemake.params.sheet_extract_config,
    )
    availability_df.to_csv(snakemake.output.csv)

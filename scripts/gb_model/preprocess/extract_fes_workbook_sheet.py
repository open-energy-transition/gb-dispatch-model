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
    excel_path: str, sheet_name: str, sheet_config: dict | list[dict]
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
    if isinstance(sheet_config, list):
        fes_data_list = []
        for config in sheet_config:
            fes_data_list.append(extract_fes_worksheet(excel_path, sheet_name, config))
        return pd.concat(fes_data_list)

    renamers = sheet_config.pop("rename", {})
    new_idx = sheet_config.pop("add_index", {})
    header = sheet_config.get("header", None)
    header_name = sheet_config.pop("header_name", None)
    index_col = sheet_config.pop("index_col", None)
    if isinstance(header, list):
        # We cannot use both "usecols" and "header" as list in pd.read_excel,
        # so set header to 0 here and dealt with later
        sheet_config["header"] = 0
    fes_data = pd.read_excel(excel_path, sheet_name=sheet_name, **sheet_config)

    if isinstance(index_col, list):
        fes_data = fes_data.set_index(fes_data.columns[index_col].tolist())
    elif isinstance(index_col, int):
        fes_data = fes_data.set_index(fes_data.columns[index_col])
    elif index_col is not None:
        raise ValueError("index_col must be int or list of int")

    if isinstance(header, list):
        for _ in header[1:]:
            fes_data = fes_data.T.set_index(fes_data.index[0], append=True).T

    fes_data = fes_data.rename_axis(**renamers)

    if fes_data.columns.name is None and header_name is not None:
        fes_data.columns.name = header_name

    if header is not None:
        fes_data = fes_data.stack(header)

    for idx_name, idx_value in new_idx.items():
        fes_data = pd.concat([fes_data], keys=[idx_value], names=[idx_name])

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

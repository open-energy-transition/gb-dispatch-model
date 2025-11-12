# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
PDF table extractor for monthly transmission availability data from NESO reports.
"""

import logging

import pandas as pd
import pdfplumber

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def extract_transmission_availability(pdf_path: str) -> pd.DataFrame:
    """
    Extract monthly transmission availability data table from a NESO PDF report.

    Args:
        pdf_path (str): Path to the PDF report.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted transmission availability data.
    """
    transmission_data = {}
    interconnector_data = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables(
                {
                    "vertical_strategy": "lines_strict",
                    "horizontal_strategy": "lines_strict",
                }
            )
            for table in tables:
                try:
                    title = table[0][0].lower()
                except IndexError:
                    continue

                if (
                    title.startswith("planned and unplanned unavailability (%)")
                    and "reactive" not in title
                ):
                    df = pd.DataFrame(table[1:]).set_index(0).rename_axis(index="month")
                    df = (
                        df.rename(columns=df.iloc[0].str.replace("\n", " "))
                        .drop("", errors="ignore")
                        .astype(float)
                    )
                    geography = page.extract_text_lines(layout=True)[0]["text"]
                    df.index = pd.to_datetime(df.index, format="%b").month
                    transmission_data[geography] = df
                    logger.info(
                        f"Found monthly availability table for {geography} transmission system with average annual unavailability of {df.Total.mean():.2f}%."
                    )
                elif "% monthly unavailability" in title:
                    geography = title.removesuffix("% monthly unavailability").rstrip(
                        " \n"
                    )
                    project = table[1][1]
                    name = f"{geography} | {project}"
                    df = (
                        pd.DataFrame(table[2:-2])
                        .set_index(0)
                        .rename_axis(index="month", columns=name)
                        .squeeze()
                        .replace("N/A", float("nan"))
                        .astype(float)
                    )
                    df.index = pd.to_datetime(df.index, format="%B").month
                    interconnector_data[name] = df
                    logger.info(
                        f"Found interconnector monthly unavailability table for {name} with average monthly unavailability of {df.mean():.2f}%."
                    )
    gb_df = pd.concat(
        transmission_data.values(),
        keys=transmission_data.keys(),
        names=["geography", "month"],
    )
    interconnector_df = pd.concat(
        interconnector_data.values(),
        keys=interconnector_data.keys(),
        names=["geography", "month"],
    )
    final_df = pd.concat([gb_df, interconnector_df.to_frame("Total")])
    return final_df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("extract_transmission_availability")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    availability_df = extract_transmission_availability(snakemake.input.pdf_report)
    availability_df.to_csv(snakemake.output.csv)

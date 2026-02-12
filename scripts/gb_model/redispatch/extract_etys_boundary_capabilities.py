# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
PDF extractor for boundary capability data from ETYS reports.

This script extracts boundary capability values from NESO ETYS PDF reports.
It searches for numbers preceding "GW" or "MW" in the bottom right corner of each page
and captures the value along with the boundary name from the page title.
Values are converted to MW.
"""

import logging
import os
import re
from zipfile import Path

import geopandas as gpd
import pandas as pd
import pdfplumber

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(os.path.basename(__file__))


def _extract_capability(text: str, gw_pattern: str, mw_pattern: str) -> float:
    """Extract capability value from text, converting to MW."""
    if match := re.search(gw_pattern, text):
        return float(match.group(1).replace(",", "")) * 1000
    if match := re.search(mw_pattern, text):
        return float(match.group(1).replace(",", ""))
    return float("nan")


def extract_etys_boundary_capabilities(
    pdf_path: str, boundaries_path: str
) -> pd.DataFrame:
    """
    Extract boundary capability data from an ETYS PDF report.

    Searches for numbers preceding "GW" or "MW" in the bottom right corner of each page.
    After deduplication, searches full page for boundaries still missing data.

    Args:
        pdf_path (str): Path to the ETYS PDF report.
        boundaries_path (str): Path to the ETYS boundaries shapefile.

    Returns:
        pd.DataFrame: A DataFrame containing boundary names and their capability values (MW).
    """
    # Load ETYS boundaries to get expected boundary names
    gdf = gpd.read_file(boundaries_path)
    expected_boundaries = set(gdf["Boundary_n"].dropna().unique())
    logger.info(f"Loaded {len(expected_boundaries)} expected boundaries from shapefile")

    # Create regex pattern from expected boundary names
    boundary_names_escaped = [
        re.escape(b) for b in sorted(expected_boundaries, reverse=True)
    ]
    boundary_pattern = r"\b(" + "|".join(boundary_names_escaped) + r")\b"

    gw_pattern = r"([\d,]+\.?\d*)\s*GW"
    mw_pattern = r"([\d,]+)\s*MW"
    boundary_data = []
    page_cache = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            # Extract boundary code from first 10 lines
            lines = page.extract_text_lines(layout=True)
            boundary_name = next(
                (
                    match.group(1)
                    for line in (lines or [])[:10]
                    if (match := re.search(boundary_pattern, line.get("text", "")))
                ),
                None,
            )
            if not boundary_name:
                continue
            page_cache[boundary_name] = page

            # Search bottom-right corner
            bbox = (page.width * 0.5, page.height * 0.5, page.width, page.height)
            text = page.crop(bbox).extract_text() or ""
            capability_mw = _extract_capability(text, gw_pattern, mw_pattern)

            if not pd.isna(capability_mw):
                logger.info(
                    f"{boundary_name}: {capability_mw} MW p{page_num} (part-page search)"
                )

            boundary_data.append(
                {
                    "boundary_name": boundary_name,
                    "capability_mw": capability_mw,
                    "page_number": page_num,
                }
            )

    if not boundary_data:
        logger.warning("No boundary capability data found in the PDF")
        return pd.DataFrame(columns=["boundary_name", "capability_mw", "page_number"])

    # Deduplicate, keeping non-NaN values
    df = pd.DataFrame(boundary_data)
    df = df.sort_values(["boundary_name", "capability_mw"], na_position="last")
    df = df.drop_duplicates(subset=["boundary_name"], keep="first")

    # Second pass: full-page search for remaining NaN values
    nan_mask = df["capability_mw"].isna()
    if nan_mask.any():
        logger.info(f"Full-page search for {nan_mask.sum()} boundaries")

        for idx in df[nan_mask].index:
            boundary_name = df.loc[idx, "boundary_name"]
            if page := page_cache.get(boundary_name):
                capability_mw = _extract_capability(
                    page.extract_text() or "", gw_pattern, mw_pattern
                )
                if not pd.isna(capability_mw):
                    df.loc[idx, "capability_mw"] = capability_mw
                    logger.info(
                        f"{boundary_name}: {capability_mw} MW (full-page search)"
                    )

    # Log remaining NaN values
    for _, row in df[df["capability_mw"].isna()].iterrows():
        logger.warning(f"{row['boundary_name']} p{row['page_number']}: not found")

    logger.info(f"Extracted {len(df)} unique boundary entries")
    return df


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    boundary_df = extract_etys_boundary_capabilities(
        snakemake.input.pdf_report, snakemake.input.boundaries
    )
    boundary_df.to_csv(snakemake.output.csv, index=False)

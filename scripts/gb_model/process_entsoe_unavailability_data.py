# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Retrieve Generation Unit Unavailability Data from ENTSO-E Transparency Platform

This script retrieves generation unit unavailability data from the ENTSO-E API
for the UK bidding zones and processes it for use in PyPSA modeling.

API Documentation: 15.1.A&B Unavailability of Generation Units
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def merge_period_dataframes(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge DataFrames from multiple periods and remove duplicates

    Args:
        dataframes: List of DataFrames to merge

    Returns:
        Combined DataFrame with duplicates removed
    """

    # Remove duplicates based on key fields
    # Use resource_mrid and start_time/end_time as the primary key for deduplication
    # This focuses on actual outage periods rather than data reporting periods
    key_columns = ["resource_mrid", "start_time", "end_time"]

    # Only use columns that exist in the dataframe
    existing_key_columns = [col for col in key_columns if col in combined_df.columns]

    # If start_time/end_time not available, fall back to resource_mrid and business_type
    if (
        "start_time" not in existing_key_columns
        and "resource_mrid" in combined_df.columns
    ):
        key_columns = ["resource_mrid", "business_type"]
        existing_key_columns = [
            col for col in key_columns if col in combined_df.columns
        ]

    if existing_key_columns:
        initial_count = len(combined_df)
        combined_df = combined_df.drop_duplicates(
            subset=existing_key_columns, keep="first"
        )
        removed_count = initial_count - len(combined_df)

        if removed_count > 0:
            logger.info(
                f"Removed {removed_count} duplicate records, {len(combined_df)} records remaining"
            )

    # Sort by start time and resource name
    if "start_time" in combined_df.columns:
        combined_df = combined_df.sort_values(["start_time", "resource_name"])
    else:
        combined_df = combined_df.sort_values(["resource_name"])

    return combined_df


def parse_xml_content(xml_content: str, filename: str = "") -> list[dict]:
    """
    Parse individual XML content for unavailability data

    Args:
        xml_content: XML content as string
        filename: Optional filename for logging

    Returns:
        List of outage dictionaries
    """
    outages = []

    root = ET.fromstring(xml_content)
    namespaces = {"ns": "urn:iec62325.351:tc57wg16:451-6:outagedocument:3:0"}

    # Find all TimeSeries elements
    timeseries_elements = root.findall(".//ns:TimeSeries", namespaces)

    xml_text_mapping = {
        "business_type": ".//ns:businessType",
        "bidding_zone": ".//ns:biddingZone_Domain.mRID",
        "resource_mrid": ".//ns:production_RegisteredResource.mRID",
        "resource_name": ".//ns:production_RegisteredResource.name",
        "resource_location": ".//ns:production_RegisteredResource.location.name",
        "resource_type": ".//ns:production_RegisteredResource.pSRType.psrType",
        "psr_mrid": ".//ns:production_RegisteredResource.pSRType.powerSystemResources.mRID",
        "psr_name": ".//ns:production_RegisteredResource.pSRType.powerSystemResources.name",
        "nominal_power": ".//ns:production_RegisteredResource.pSRType.powerSystemResources.nominalP",
    }
    for ts in timeseries_elements:
        outage_data = {
            k: _get_text(ts, v, namespaces) for k, v in xml_text_mapping.items()
        }

        outage_data["nominal_power_mw"] = (
            float(outage_data["nominal_power"])
            if outage_data["nominal_power"]
            else None
        )

        # Time periods from TimeSeries level
        start_date = _get_text(ts, ".//ns:start_DateAndOrTime.date", namespaces)
        start_time = _get_text(ts, ".//ns:start_DateAndOrTime.time", namespaces)
        end_date = _get_text(ts, ".//ns:end_DateAndOrTime.date", namespaces)
        end_time = _get_text(ts, ".//ns:end_DateAndOrTime.time", namespaces)

        if start_date and start_time:
            outage_data["start_time"] = pd.to_datetime(f"{start_date} {start_time}")
        if end_date and end_time:
            outage_data["end_time"] = pd.to_datetime(f"{end_date} {end_time}")

        # Process Available_Period elements
        available_periods = ts.findall(".//ns:Available_Period", namespaces)

        if available_periods:
            for period in available_periods:
                period_data = outage_data.copy()

                # Extract data points
                points = period.findall(".//ns:Point", namespaces)
                if points:
                    quantities: list[float] = []
                    for point in points:
                        qty = _get_text(point, ".//ns:quantity", namespaces)
                        if qty:
                            quantities.append(float(qty))

                    if quantities:
                        period_data["min_available_mw"] = min(quantities)
                        period_data["max_available_mw"] = max(quantities)
                        period_data["avg_available_mw"] = sum(quantities) / len(
                            quantities
                        )
                        period_data["num_data_points"] = len(quantities)

                        # Calculate unavailable capacity
                        if outage_data["nominal_power_mw"]:
                            period_data["min_unavailable_mw"] = max(
                                0, outage_data["nominal_power_mw"] - max(quantities)
                            )
                            period_data["max_unavailable_mw"] = max(
                                0, outage_data["nominal_power_mw"] - min(quantities)
                            )

                outages.append(period_data)
        else:
            # No Available_Period elements, just add the TimeSeries data
            outages.append(outage_data)

    return outages


def _get_text(element, xpath: str, namespaces: dict[str, str]) -> str | None:
    """
    Helper function to safely extract text from XML element

    Args:
        element: XML element
        xpath: XPath expression
        namespaces: XML namespaces

    Returns:
        Text content or None
    """
    found = element.find(xpath, namespaces)
    return found.text if found is not None else None


def process_unavailability_data(
    df: pd.DataFrame, output_file: str, bidding_zone: str, business_types: dict
) -> None:
    """
    Process and save unavailability data

    Args:
        df: DataFrame with raw unavailability data
        output_file: Path to save processed data
        bidding_zone: Bidding zone identifier
        config: Snakemake configuration dictionary
    """
    # Select only the columns we want to keep in the output
    output_columns = [
        "resource_mrid",
        "resource_name",
        "resource_location",
        "resource_type",
        "business_type",
        "start_time",
        "end_time",
        "nominal_power_mw",
        "min_available_mw",
        "max_available_mw",
        "avg_available_mw",
        "min_unavailable_mw",
        "max_unavailable_mw",
        "bidding_zone",
        "business_type_desc",
    ]

    if df.empty:
        logger.warning("No data to process")
        # Create empty CSV with headers
        empty_df = pd.DataFrame(columns=output_columns)
        empty_df.to_csv(output_file, index=False)
        return

    # Add bidding zone information
    df["bidding_zone"] = bidding_zone

    # Clean and process data
    business_type_map = {v: k for k, v in business_types.items()}
    df["business_type_desc"] = df["business_type"].map(business_type_map)

    # Sort by start time
    df = df.sort_values(["start_time", "resource_name"])

    # Filter to only include columns that exist in the dataframe
    columns_to_keep = [col for col in output_columns if col in df.columns]
    df_filtered = df[columns_to_keep]

    # Save processed data
    df_filtered.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df_filtered)} records to {output_file}")

    # Print summary statistics
    logger.info("Data summary:")
    if (
        "start_time" in df_filtered.columns
        and not df_filtered["start_time"].isnull().all()
    ):
        logger.info(
            f"  Date range: {df_filtered['start_time'].min()} to {df_filtered['end_time'].max()}"
        )
    logger.info(f"  Total outage records: {len(df_filtered)}")
    logger.info(f"  Unique generation units: {df_filtered['resource_name'].nunique()}")

    if (
        "nominal_power_mw" in df_filtered.columns
        and not df_filtered["nominal_power_mw"].isnull().all()
    ):
        logger.info(
            f"  Total installed capacity: {df_filtered['nominal_power_mw'].sum():.1f} MW"
        )


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("process_entsoe_unavailability_data")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Get date parameters from config or use defaults
    business_type_codes = snakemake.params.business_type_codes

    all_content: list[dict] = []
    for file in tqdm(
        Path(snakemake.input.xml_base_dir).rglob("*.xml"),
        desc="Processing XML files",
        total=len(list(Path(snakemake.input.xml_base_dir).rglob("*.xml"))),
    ):
        parsed_content = parse_xml_content(file.read_text(), filename=str(file))
        all_content.extend(parsed_content)
    df = pd.DataFrame(all_content)
    # Merge all periods into a single DataFrame
    df = merge_period_dataframes(df)
    logger.info(f"Final merged DataFrame has {len(df)} rows")

    # Generate output filename
    output_file = snakemake.output.unavailability

    # Process and save data
    process_unavailability_data(
        df, output_file, snakemake.wildcards.zone, business_type_codes
    )

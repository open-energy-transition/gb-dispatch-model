# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Retrieve Generation Unit Unavailability Data from ENTSO-E Transparency Platform

This script retrieves generation unit unavailability data from the ENTSO-E API
for the UK bidding zones and processes it for use in PyPSA modeling.

API Documentation: 15.1.A&B Unavailability of Generation Units
"""

import io
import logging
import os
import time
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


class ENTSOEUnavailabilityAPI:
    """
    Interface to ENTSO-E Transparency Platform API for unavailability data
    """

    def __init__(self, api_key: str):
        """
        Initialize the API client

        Args:
            api_key: ENTSO-E API key for authentication
        """
        self.api_key = api_key
        self.base_url = "https://web-api.tp.entsoe.eu/api"
        self.session = requests.Session()
        self.rate_limit_delay = 2.0  # Seconds between requests

    def _make_request(self, params: dict[str, Any]) -> requests.Response:
        """
        Make API request with rate limiting and error handling

        Args:
            params: Query parameters for the API request

        Returns:
            Response object
        """
        # Add API key to parameters
        params["securityToken"] = self.api_key

        logger.debug(f"Making API request with params: {params}")

        try:
            # Rate limiting
            time.sleep(self.rate_limit_delay)

            response = self.session.get(self.base_url, params=params, timeout=30)
            logger.debug(f"API response status: {response.status_code}")
            logger.debug(f"API response length: {len(response.content)} bytes")

            if response.status_code != 200:
                logger.error(f"API returned status {response.status_code}")

            response.raise_for_status()
            return response

        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise

    def get_unavailability_data(
        self,
        bidding_zone: str,
        period_start: datetime,
        period_end: datetime,
        business_type: str | None = None,
        doc_status: str | None = None,
        registered_resource: str | None = None,
        offset: int = 0,
    ) -> bytes:
        """
        Retrieve unavailability data from ENTSO-E API

        Args:
            bidding_zone: EIC code for bidding zone (e.g., '10YGB----------A' for GB)
            period_start: Start datetime for the query period
            period_end: End datetime for the query period
            business_type: Optional business type ('A53' for planned, 'A54' for forced)
            doc_status: Optional document status ('A05', 'A09', 'A13')
            registered_resource: Optional EIC code of specific generation unit
            offset: Offset for pagination (0-4800)

        Returns:
            Raw response content as bytes (could be XML or ZIP)
        """
        params = {
            "documentType": "A80",  # Generation unavailability
            "BiddingZone_Domain": bidding_zone,
            "periodStart": period_start.strftime("%Y%m%d%H%M"),
            "periodEnd": period_end.strftime("%Y%m%d%H%M"),
        }

        # Add optional parameters
        if business_type:
            params["BusinessType"] = business_type
        if doc_status:
            params["DocStatus"] = doc_status
        if registered_resource:
            params["RegisteredResource"] = registered_resource
        if offset > 0:
            params["offset"] = offset

        response = self._make_request(params)
        return response.content


def generate_weekly_periods(
    start_date: datetime, end_date: datetime, max_days: int = 7
) -> list[tuple]:
    """
    Generate weekly periods for API requests to stay within API limits

    Args:
        start_date: Overall start date
        end_date: Overall end date
        max_days: Maximum days per request (default: 7)

    Returns:
        List of (start, end) datetime tuples for each period
    """
    periods = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=max_days), end_date)
        periods.append((current_start, current_end))
        current_start = current_end

    logger.info(
        f"Split {start_date.date()} to {end_date.date()} into {len(periods)} weekly periods"
    )
    return periods


def retrieve_data_by_periods(
    api_client,
    bidding_zone: str,
    periods: list[tuple],
    business_type: str,
    doc_status: str,
    xml_save_dir: Path,
):
    """
    Retrieve data for multiple periods and return list of DataFrames

    Args:
        api_client: ENTSOEUnavailabilityAPI instance
        bidding_zone: EIC code for bidding zone
        periods: List of (start, end) datetime tuples
        business_type: Business type code
        doc_status: Document status code
        xml_save_dir: Directory to save XML files

    Returns:
        List of DataFrames, one per period
    """
    successful_periods = 0

    # Log overall progress
    logger.info(
        f"Retrieving {len(periods)} periods from {periods[0][0].date()} to {periods[-1][1].date()}"
    )

    for i, (period_start, period_end) in tqdm(
        enumerate(periods, 1), desc="Retrieving XML files", total=len(periods)
    ):
        try:
            # Add period identifier to XML save directory
            period_xml_dir = (
                xml_save_dir
                / f"period_{period_start.strftime('%Y%m%d')}_{period_end.strftime('%Y%m%d')}"
            )

            response_content = api_client.get_unavailability_data(
                bidding_zone=bidding_zone,
                period_start=period_start,
                period_end=period_end,
                business_type=business_type,
                doc_status=doc_status,
            )

            # Parse data and save XML files
            expand_zip(response_content, period_xml_dir)

        except Exception as e:
            # Only log actual errors, not "no data" cases
            if "File is not a zip file" not in str(e):
                logger.warning(
                    f"Period {i} ({period_start.date()} to {period_end.date()}): {e}"
                )
            continue

    logger.info(
        f"Retrieved data from {successful_periods}/{len(periods)} periods ({(successful_periods / len(periods) * 100):.0f}% success rate)"
    )


def expand_zip(zip_content: bytes, xml_save_dir: Path) -> None:
    """
    Expand ZIP file containing XML unavailability data

    Args:
        zip_content: ZIP file content as bytes
        xml_save_dir: Optional directory to save extracted XML files

    Returns:
        DataFrame with parsed unavailability data
    """
    with zipfile.ZipFile(io.BytesIO(zip_content)) as zf:
        xml_files = [f for f in zf.namelist() if f.endswith(".xml")]
        # Only log if we actually find files
        if xml_files:
            logger.debug(f"Processing ZIP file with {len(xml_files)} XML files")
            xml_save_dir.mkdir(parents=True, exist_ok=True)

        for filename in xml_files:
            xml_content = zf.read(filename).decode("utf-8")
            file_path = xml_save_dir / filename
            file_path.write_text(xml_content, encoding="utf-8")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            Path(__file__).stem,
            zone="GB",
            business_type="planned",
        )

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment variable
    api_key = os.getenv("ENTSO_E_API_KEY")

    if not api_key:
        raise ValueError(
            "ENTSO-E API key not found. Please set ENTSO_E_API_KEY in your .env file or environment variables.\\n"
            "You can get an API key from: https://transparency.entsoe.eu/usrm/user/createPublicApiUser.do"
        )

    # Get date parameters from config or use defaults
    start_date = pd.to_datetime(snakemake.params.start_date)
    end_date = pd.to_datetime(snakemake.params.end_date)
    max_request_days = snakemake.params.max_request_days

    # Split the date range into weekly periods
    periods = generate_weekly_periods(start_date, end_date, max_days=max_request_days)

    # Get mapping dictionaries from config
    uk_bidding_zones = snakemake.params.api_params["bidding_zones"]
    business_types_map = snakemake.params.api_params["business_types"]
    doc_status_map = snakemake.params.api_params["doc_status"]

    # Initialize API client
    api_client = ENTSOEUnavailabilityAPI(api_key)

    xml_save_dir = Path(snakemake.output.xml_base_dir)

    zone = snakemake.wildcards.zone
    business_type = snakemake.wildcards.business_type
    zone_code = uk_bidding_zones[zone]
    business_code = business_types_map[business_type]
    # Process each bidding zone

    logger.info(
        f"Retrieving unavailability data for {zone} ({zone_code}) & business type {business_type} ({business_code})"
    )

    # Retrieve data for all periods
    retrieve_data_by_periods(
        api_client=api_client,
        bidding_zone=zone_code,
        periods=periods,
        business_type=business_code,
        doc_status=doc_status_map["active"],
        xml_save_dir=xml_save_dir,
    )

    # Count XML files across all period subdirectories
    total_xml_count = len(list(xml_save_dir.rglob("*.xml")))
    total_size = sum(f.stat().st_size for f in xml_save_dir.rglob("*.xml"))

    period_dirs = [
        d for d in xml_save_dir.iterdir() if d.is_dir() and d.name.startswith("period_")
    ]

    logger.info(
        f"Stored {total_xml_count} XML files ({total_size / 1024 / 1024:.1f} MB) across {len(period_dirs)} periods in {xml_save_dir}"
    )

    logger.info("Unavailability data retrieval completed")

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
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from tqdm import tqdm

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)

AVAIL_DOC_STATUS = "A05"


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
        self.session = EntsoePandasClient(api_key=api_key)

    def get_unavailability_data(
        self,
        country_code: str,
        period_start: datetime,
        period_end: datetime,
    ) -> pd.DataFrame:
        """
        Retrieve unavailability data from ENTSO-E API.

        Args:
            country_code (str): ISO2 country code (e.g. 'GB')
            period_start (datetime): Start datetime for the query period∑
            period_end (datetime): End datetime for the query period

        Returns:
            pd.DataFrame: DataFrame containing unavailability data
        """
        logger.info(
            f"Retrieving {country_code} unavailability data from {os.getenv('ENTSOE_ENDPOINT_URL')} for period {period_start} to {period_end}"
        )
        response = self.session.query_unavailability_of_generation_units(
            country_code,
            start=pd.Timestamp(period_start, tz="UTC"),
            end=pd.Timestamp(period_end, tz="UTC"),
            docstatus=AVAIL_DOC_STATUS,
        )
        return response


def generate_time_chunks(
    start_date: datetime, end_date: datetime, max_days: int
) -> list[tuple]:
    """
    Generate time chunks for API requests to stay within API limits

    Args:
        start_date: Overall start date
        end_date: Overall end date
        max_days: Maximum days per request

    Returns:
        List of (start, end) datetime tuples for each period
    """
    chunks = []
    current_start = start_date

    while current_start < end_date:
        current_end = min(current_start + timedelta(days=max_days), end_date)
        chunks.append((current_start, current_end))
        current_start = current_end

    logger.info(
        f"Split {start_date.date()} to {end_date.date()} into {len(chunks)} chunks"
    )
    return chunks


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

    # Split the date range into time chunks
    chunks = generate_time_chunks(start_date, end_date, max_days=max_request_days)

    # Initialize API client
    api_client = ENTSOEUnavailabilityAPI(api_key)

    zone = snakemake.wildcards.entsoe_country_code

    dfs = []
    for period_start, period_end in tqdm(
        chunks, desc="Retrieving ENTSO-E data", total=len(chunks)
    ):
        response_content = api_client.get_unavailability_data(
            country_code=zone,
            period_start=period_start,
            period_end=period_end,
        )
        dfs.append(response_content)
    df = pd.concat(dfs)

    df.to_csv(snakemake.output.unavailability)

    logger.info(f"Logged {len(df)} outage events for zone {zone}")

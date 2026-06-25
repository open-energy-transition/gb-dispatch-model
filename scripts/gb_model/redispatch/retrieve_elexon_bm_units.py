# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Fetch Balancing Mechanism units from Elexon BMRS API
"""

import logging
from pathlib import Path

import pandas as pd
import requests

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def fetch_api_request_data(
    url: str, retrieval_message: str, session: requests.Session = None
) -> pd.DataFrame:
    """
    Fetch Bid Offer Data (BOD) from Elexon API for the specified date range.

    Parameters
    ----------
    url: str
        API request URL
    retrieval_message: str
        Message to log for the retrieval operation
    session: requests.Session or aiohttp.ClientSession, optional
        client session to use for the request
    """

    df = pd.DataFrame()

    max_retries = 3
    for i in range(1, max_retries + 1):
        try:
            headers = {"User-Agent": "Mozilla/5.0"}

            response = session.get(url, headers=headers)
            response.raise_for_status()

            json_data = response.json()

            # Convert JSON data to Dataframe
            if isinstance(json_data, dict):
                df = pd.DataFrame(json_data.get("data", []))
            elif isinstance(json_data, list):
                df = pd.DataFrame(json_data)

            break  # Exit retry loop if successful

        except requests.exceptions.RequestException as e:
            logger.warning(
                f" Attempt {i}/{max_retries}: Error in fetching {retrieval_message}: {e}"
            )
            if i == max_retries:
                logger.error(
                    f"Max retries reached. Failed to fetch {retrieval_message}."
                )

    return df


def fetch_BM_units(
    base_url: str,
    bmu_fuel_map_path: str,
    api_bmu_fuel_map: bool,
) -> tuple[str, dict]:
    """
    Fetch BM Unit data from Elexon API to get mapping of BM units to fuel types

    Parameters
    ----------
    base_url: str
        Base URL for the Elexon BMRS API
    technology_mapping: dict[str, str]
        Map Elexon carrier types to PyPSA carrier types
    bmu_fuel_map_path: str
        CSV path for mapping of BMU units to their fueltype
    api_bmu_fuel_map: bool
        Boolean to choose between fetching BM units via API (True) / reading existing excel (False)
    """

    if api_bmu_fuel_map:
        # URL to fetch BM unit data
        url = f"{base_url}/reference/bmunits/all"

        df_bmu = fetch_api_request_data(url, retrieval_message="BMU Unit Data")
    else:
        df_bmu = pd.read_excel(bmu_fuel_map_path).rename(
            columns={"NESO BMU ID": "nationalGridBmUnit", "REG FUEL TYPE": "fuelType"}
        )

    return df_bmu


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    base_url = "https://data.elexon.co.uk/bmrs/api/v1"

    bmu_units = fetch_BM_units(
        base_url,
        bmu_fuel_map_path=snakemake.input.bmu_fuel_map_path,
        api_bmu_fuel_map=snakemake.params.api_bmu_fuel_map,
    )

    bmu_units.to_csv(snakemake.output.csv)
    logger.info(f"Saved BMU units to {snakemake.output.csv}")

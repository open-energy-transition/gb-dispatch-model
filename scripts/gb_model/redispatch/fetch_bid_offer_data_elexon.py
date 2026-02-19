# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Fetch Bid Offer Data from Elexon BMRS API
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

import aiohttp
import numpy as np
import pandas as pd
import requests
from tqdm.asyncio import tqdm_asyncio

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


async def fetch_api_request_data(
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

    async with SEM:
        max_retries = 3
        for i in range(1, max_retries + 1):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}

                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()

                    json_data = await response.json()

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


async def get_historical_bod(
    base_url: str, bmu_carrier_map: dict[str, str], year: int
) -> pd.DataFrame:
    """
    Fetch bid / offer data for an entire year by querying day by day

    Parameters
    ----------
    base_url: str
        Base URL for the Elexon BMRS API
    bmu_carrier_map: dict[str, str]
        Mapping of BM units to carriers
    year: int
        Year for which to fetch the data
    """
    dfs = []

    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)

    current = start

    async with aiohttp.ClientSession(trust_env=True) as session:
        while current <= end:
            # There are 48 settlement periods per day - every 30 mins
            for settlement_period in np.arange(1, 49):
                # API request URL for the day
                url = f"{base_url}/balancing/settlement/acceptances/all/{current.strftime('%Y-%m-%d')}/{settlement_period}"

                # Fetch data for the current day
                df = asyncio.create_task(
                    fetch_api_request_data(
                        url,
                        retrieval_message=f"Bid/Offer price data for the dates {current} and settlement period {settlement_period}",
                        session=session,
                    )
                )

                dfs.append(df)

            # Reset current date to next day
            current += pd.DateOffset(days=1)

        results = []
        for task in tqdm_asyncio.as_completed(
            dfs, total=len(dfs), desc=f"Fetching Bid/Offer data for the year {year}"
        ):
            res = await task
            results.append(res)

    logger.info(f"Successfully retrieved Bid/offer price data for {year}.")

    df_bod = pd.concat(results, ignore_index=True)
    df_bod["carrier"] = df_bod["nationalGridBmUnit"].map(bmu_carrier_map)

    df_bod_mean = pd.DataFrame()
    # BidOfferPairId is an indication of the bandwidth within which a BMunit can increase / decrease it's power output.
    # -ve pairId's indicate bids and +ve pairId's indicate offers
    # The bid / offer price can vary with the pairId - for simplicity an average of the prices over the pair id's is used here
    df_bod_mean["bid"] = (
        df_bod.query("bidOfferPairId < 0").groupby("carrier").bidPrice.mean()
    )
    df_bod_mean["offer"] = (
        df_bod.query("bidOfferPairId > 0").groupby("carrier").offerPrice.mean()
    )

    return df_bod_mean


async def fetch_BM_units(
    base_url: str,
    technology_mapping: dict[str, str],
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

        # Fetch BM unit data
        async with aiohttp.ClientSession(trust_env=True) as session:
            df_bmu = await fetch_api_request_data(
                url, retrieval_message="BMU Unit Data", session=session
            )
    else:
        df_bmu = pd.read_excel(bmu_fuel_map_path).rename(
            columns={"NESO BMU ID": "nationalGridBmUnit", "REG FUEL TYPE": "fuelType"}
        )

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

    base_url = "https://data.elexon.co.uk/bmrs/api/v1"

    # tune if HTTP 429 error occurs
    max_concurrent_requests = int(snakemake.params.max_concurrent_requests)
    logger.info(
        "N concurrent requests set to %d. "
        "Tune this parameter if you encounter HTTP 429 errors.",
        max_concurrent_requests,
    )
    global SEM
    SEM = asyncio.Semaphore(max_concurrent_requests)

    bmu_carrier_map = asyncio.run(
        fetch_BM_units(
            base_url,
            technology_mapping=snakemake.params.technology_mapping,
            bmu_fuel_map_path=snakemake.input.bmu_fuel_map_path,
            api_bmu_fuel_map=snakemake.params.api_bmu_fuel_map,
        )
    )

    df_bod = asyncio.run(
        get_historical_bod(
            base_url,
            bmu_carrier_map,
            year=int(snakemake.wildcards.bod_year),
        )
    )

    df_bod.to_csv(snakemake.output.csv)
    logger.info(
        f"Saved Bid/Offer average cost data for the year {snakemake.wildcards.bod_year} to {snakemake.output.csv}"
    )

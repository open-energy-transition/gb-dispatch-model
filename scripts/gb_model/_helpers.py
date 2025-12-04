# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


import logging
from datetime import datetime

import geopandas as gpd
import numpy as np
import pandas as pd
import pytz
from pytz import country_timezones

logger = logging.getLogger(__name__)


def map_points_to_regions(
    df: pd.DataFrame,
    gdf_regions: gpd.GeoDataFrame,
    lat_col: str,
    lon_col: str,
    point_crs: str,
    projected_crs: str,
    dwithin_distance: float = 100,
) -> pd.DataFrame:
    """
    Map points from a DataFrame to regions in a GeoDataFrame.

    Args:
        df (pd.DataFrame): input DataFrame with point coordinates
        gdf_regions (gpd.GeoDataFrame): GeoDataFrame with region geometries
        lat_col (str): latitude column name in df
        lon_col (str): longitude column name in df
        point_crs (str): CRS of the input points
        projected_crs (str): CRS to project the points and regions to when performing spatial join
        dwithin_distance (float, optional): distance (in `projected_crs` units (e.g. metres)) away from region in which points will still be considered as "within" a region . Defaults to 100.

    Returns:
        pd.DataFrame: DataFrame with the same index as `df`, but containing region information
    """
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=point_crs,
        index=df.index,
    ).to_crs(projected_crs)

    regions = gpd.sjoin(
        points,
        gdf_regions.to_crs(projected_crs),
        how="left",
        predicate="dwithin",
        distance=dwithin_distance,
    ).drop(columns="geometry")
    return regions


def strip_srt(series: pd.Series) -> pd.Series:
    """Strip whitespace from strings in a pandas Series."""
    return series.str.strip() if series.dtype == "object" else series


def to_numeric(series: pd.Series) -> pd.Series:
    """Convert a pandas Series to numeric, replacing - with 0."""
    series = series.astype(str).str.strip()
    series = series.replace("-", np.nan)
    series = pd.to_numeric(series).fillna(0)
    return series


def standardize_year(series: pd.Series) -> pd.Series:
    """Standardize year format in a pandas Series."""
    if series.dtype == "object" and "-" in str(series.iloc[0]):
        series = pd.to_datetime(series).dt.year
    return series.astype(int) if series.dtype == "object" else series


def pre_format(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-format dataframe by stripping string, converting numerics, and standardizing year."""
    df = df.apply(strip_srt)
    df["year"] = standardize_year(df["year"])
    df["data"] = to_numeric(df["data"])
    return df


def get_regional_distribution(df: pd.Series) -> pd.Series:
    """
    Calculate regional distribution of data for each year.

    Args:
        df (pd.Series): Series containing data with index as 'bus' and 'year'.

    Returns:
        pd.Series: Series with the same index as input, but containing regional distribution
                   proportions instead of absolute values. Each row (year) sums to 1.0 across all
                   regions (columns).
    """
    regional_distribution = df.groupby(level="year").transform(lambda x: x / x.sum())

    return regional_distribution


def get_lines(lines: pd.DataFrame, bus0: str | int, bus1: str | int) -> pd.Series:
    """
    Get a boolean mask of lines connecting bus0 and bus1.
    This accounts for cases where bus0 and bus1 may be swapped.
    """
    all_lines = set(lines["bus0"]).union(set(lines["bus1"]))
    for bus in (bus0, bus1):
        if f"GB {bus}" not in all_lines:
            logger.warning(f"Bus 'GB {bus}' not found in network lines")
    return ((lines["bus0"] == f"GB {bus0}") & (lines["bus1"] == f"GB {bus1}")) | (
        (lines["bus0"] == f"GB {bus1}") & (lines["bus1"] == f"GB {bus0}")
    )


def time_difference_hours(country):
    """
    Calculate time difference in hours between GB and specified country

    """

    # Get timezones for GB and the specified country
    try:
        tz_gb = pytz.timezone(country_timezones["GB"][0])
        tz_country = pytz.timezone(country_timezones[country][0])
    except KeyError:
        raise ValueError("Invalid ISO country code or timezone not found.")

    # Localize current UTC time into each timezone
    now_utc = datetime.now(pytz.utc)
    time_gb = now_utc.astimezone(tz_gb)
    time_country = now_utc.astimezone(tz_country)

    offset_gb = time_gb.utcoffset().total_seconds() / 3600
    offset_country = time_country.utcoffset().total_seconds() / 3600

    # Compute difference in hours
    diff = offset_country - offset_gb

    return int(diff)

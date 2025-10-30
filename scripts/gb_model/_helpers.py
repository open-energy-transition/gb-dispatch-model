# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


import logging

import geopandas as gpd
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def map_points_to_regions(
    df: pd.DataFrame,
    gdf_regions: gpd.GeoDataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    crs: str = "EPSG:4326",
) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]), crs=crs
    ).to_crs(gdf_regions.crs)
    regions = gpd.sjoin(points, gdf_regions, how="left", predicate="intersects")["name"]
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

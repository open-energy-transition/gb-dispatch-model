# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
GSP-level data table generator.

This is a script to combine the BB1 sheet with the BB2 (metadata) sheet of the FES workbook.
"""

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import map_points_to_regions, strip_srt

logger = logging.getLogger(__name__)


def parse_inputs(
    bb1_path: str,
    bb2_path: str,
    gsp_coordinates_path: str,
    fes_scenario: str,
    year_range: list,
) -> pd.DataFrame:
    """
    Parse the input data to the required format.

    Args:
        bb1_path (str): path of extracted sheet BB1 of the FES workbook
        bb2_path (str): path of extracted sheet BB2 of the FES workbook
        gsp_coordinates_path (str): path of the GSP supply point coordinates file
        fes_scenario (str): FES scenario
    """

    df_bb2 = pd.read_csv(bb2_path)

    # First step: extract the ID numbers from the Parameter column and set it as the index (it is the only unique identifier for table BB2)
    df_bb2 = (
        df_bb2.set_index(
            ["Template", "Technology", "Technology Detail", "Parameter"], append=True
        )
        .squeeze()
        .unstack("Parameter")
    )
    df_bb2_pivoted = (
        df_bb2.bfill()
        .where(~df_bb2["Building Block ID Number"].isnull())
        .dropna(how="all")
        .reset_index()
        .set_index("Building Block ID Number")
        .drop("level_0", axis=1)
        .apply(strip_srt)
    )

    df_bb1 = pd.read_csv(bb1_path)
    df_bb1 = df_bb1.apply(strip_srt)
    df_bb1_scenario = df_bb1[
        (df_bb1["FES Scenario"].str.lower() == fes_scenario)
        & (df_bb1["year"].isin(range(year_range[0], year_range[1] + 1)))
    ]
    non_data_cols = df_bb1_scenario.columns.drop("data")
    if (duplicates := df_bb1_scenario[non_data_cols].duplicated()).any():
        # Manual inspection suggests these are true duplicates that should be summed
        logger.warning(
            f"There are {duplicates.sum()} duplicate rows in BB1. These will be summed."
        )
    df_bb1_scenario_no_dups = df_bb1_scenario.groupby(
        non_data_cols.tolist(), as_index=False
    )["data"].sum()

    df_bb1_bb2_scenario = pd.merge(
        df_bb1_scenario_no_dups,
        df_bb2_pivoted,
        left_on="Building Block ID Number",
        right_index=True,
    )
    assert len(df_bb1_bb2_scenario) == len(df_bb1_scenario_no_dups), (
        "Some Building Blocks in BB1 are not present in BB2"
    )

    # We allow cases where there is only a partial match ("Number" vs "Number of" by comparing string starts)
    units_match = df_bb1_bb2_scenario.apply(
        lambda x: x.Units.startswith(x.Unit), axis=1
    )
    assert (units_match).all(), (
        "Mapping of building blocks between BB1 and BB2 may be incorrect as some units do not match: "
        f"{df_bb1_bb2_scenario[~units_match][['Unit', 'Units']]}"
    )

    df_bb1_bb2_scenario = df_bb1_bb2_scenario.drop(
        columns=["Units", "Building Block ID Number"]
    )

    df_gsp_coordinates = pd.read_csv(gsp_coordinates_path)
    df_gsp_coordinates = df_gsp_coordinates.apply(strip_srt)

    # Note
    # The GSP's "East Claydon" and "Ferrybridge B" have duplicates
    # the lat and lon information is the same but the GSP ID and GSP group are slightly different
    df_gsp_coordinates = df_gsp_coordinates.drop_duplicates(subset=["Name"])
    df_bb1_bb2_with_lat_lon = pd.merge(
        df_bb1_bb2_scenario, df_gsp_coordinates, left_on="GSP", right_on="Name"
    )

    # Missing data checks.
    # We won't raise errors here as we are willing to accept some missing data for now
    missing_lat_lon = df_bb1_bb2_with_lat_lon[
        df_bb1_bb2_with_lat_lon[["Latitude", "Longitude"]].isnull().any(axis=1)
    ].GSP.unique()
    if len(missing_lat_lon) > 0:
        logger.error(
            f"The following GSPs are missing latitude and/or longitude information: {missing_lat_lon}"
        )

    missing_gsps = set(df_bb1_bb2_scenario.GSP).difference(df_bb1_bb2_with_lat_lon.GSP)
    if missing_gsps:
        logger.error(
            f"The following GSPs are missing from the GSP coordinates file: {missing_gsps}"
        )
    df_final = pd.concat(
        [df_bb1_bb2_scenario.query("GSP in @missing_gsps"), df_bb1_bb2_with_lat_lon]
    )
    return df_final


def distribute_direct_gsp_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Distribute data from Direct(<TO_region>) rows to the GSPs in the same TO region.

    For each "Direct" data row, the data is distributed across other GSPs in the same
    TO_region that share the same FES Scenario, Technology Detail, year, Template,
    Unit, and Technology. Distribution is proportional to the existing GSP data values.
    If no other GSPs exist with data, the Direct data is distributed evenly across
    all GSPs in the region.

    Args:
        df (pd.DataFrame): DataFrame with GSP data including Direct GSPs

    Returns:
        pd.DataFrame: DataFrame with Direct data distributed to other GSPs, stored in a "TO_region" column
    """
    # Identify Direct GSP rows
    is_direct = df.GSP == f"Direct({df.name})"
    df_direct = df[is_direct].drop("GSP", axis=1).dropna(how="all", axis=1)

    non_data_cols = df_direct.columns.drop("data").to_list()
    data_direct = df_direct.set_index(non_data_cols).data
    data_gsp = df[~is_direct].set_index(non_data_cols + ["GSP"]).data

    GSPs = df.dropna(subset=["Latitude", "Longitude"]).GSP.unique().tolist()

    data_direct_extended = pd.concat(
        [data_direct for i in GSPs], keys=pd.Index(GSPs, name="GSP")
    )
    # Where possible, get the relative proportions of existing GSP data to distribute Direct data accordingly
    data_gsp_relative = data_gsp.groupby(non_data_cols, group_keys=False).apply(
        lambda x: x / x.sum()
    )
    distributed_data = data_direct_extended * data_gsp_relative

    # Some data will still be missing where there was no existing GSP data to base the distribution on
    still_missing = (
        (data_direct - distributed_data.groupby(non_data_cols, group_keys=False).sum())
        .dropna()
        .abs()
        .where(lambda x: x > 1e-10)
    )

    if (still_missing > 0).any():
        being_filled = (
            still_missing.groupby(["Template", "Technology", "Technology Detail"])
            .first()
            .index.values
        )
        logger.debug(
            f"No matching GSPs with data found to distribute {df.name} data for:\n{being_filled}\n"
            "Distributing TO-level data evenly across all GSPs for these cases."
        )
        still_missing_extended = pd.concat(
            [still_missing / len(GSPs) for i in GSPs], keys=pd.Index(GSPs, name="GSP")
        )
        distributed_data = distributed_data.fillna(
            still_missing_extended.reindex(distributed_data.index)
        )

    all_data = pd.concat(
        [
            df.set_index(data_gsp.index.names),
            distributed_data.to_frame("TO_data").reorder_levels(data_gsp.index.names),
        ],
        axis=1,
    ).dropna(subset=["data", "TO_data"], how="all")

    # The concat doesn't pass on some column data (e.g. lat/lon) into new rows, so we fill these with a per-GSP forward fill.
    all_data_filled = all_data.fillna(
        all_data.groupby("GSP", group_keys=False)
        .apply(lambda x: x.ffill())
        .drop(["TO_data", "data"], axis=1)
    )

    # We might get floating point precision issues from the data distribution, so use almost equal
    np.testing.assert_almost_equal(
        all_data["TO_data"].sum(),
        data_direct.sum(),
        err_msg=f"Data mismatch after distributing Direct {df.name} GSP data",
    )

    assert all_data_filled["data"].sum() == df["data"].sum(), (
        f"Data loss after distributing Direct {df.name} GSP data"
    )
    logger.info(
        f"Distributed {len(data_direct)} rows of {df.name} TO-level data to GSPs."
    )
    return all_data_filled.reset_index()


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file paths
    bb1_path = snakemake.input.bb1_sheet
    bb2_path = snakemake.input.bb2_sheet
    gsp_coordinates_path = snakemake.input.gsp_coordinates
    gdf_regions = gpd.read_file(snakemake.input.regions)

    # Load all the params
    fes_scenario = snakemake.params.scenario
    year_range = snakemake.params.year_range

    df = parse_inputs(
        bb1_path, bb2_path, gsp_coordinates_path, fes_scenario, year_range
    )

    region_data = map_points_to_regions(
        df,
        gdf_regions,
        "Latitude",
        "Longitude",
        "EPSG:4326",
        snakemake.params.target_crs,
    )[["name", "TO_region"]]
    df_with_regions = pd.concat(
        [df, region_data.rename(columns={"name": "bus"})], axis=1
    )
    for TO_region in gdf_regions["TO_region"].unique():
        df_with_regions.loc[
            df_with_regions.GSP == f"Direct({TO_region})", "TO_region"
        ] = TO_region
    logger.info(f"Extracted the {fes_scenario} relevant data")

    # Distribute Direct GSP data to other GSPs in the same region
    df_distributed = df_with_regions.groupby("TO_region", group_keys=False).apply(
        distribute_direct_gsp_data
    )

    df_distributed.to_csv(snakemake.output.csv, index=False)

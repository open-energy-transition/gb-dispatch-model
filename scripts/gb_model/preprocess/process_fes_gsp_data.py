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
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import (
    get_scenario_name,
    map_points_to_regions,
    strip_str,
)

logger = logging.getLogger(__name__)


def process_gsp_coordinates(
    gsp_coordinates_path: str, extra_gsp_coordinates: dict
) -> pd.DataFrame:
    """
    Process GSP coordinates data, including filling in extra coordinates and removing duplicates

    Parameters
    ----------
    gsp_coordinates_path: str
        Path to the GSP coordinates file, which contains the latitude and longitude of each GSP
    extra_gsp_coordinates: dict
        A dictionary of extra GSP coordinates to add
    """

    df_gsp_coordinates = pd.read_csv(gsp_coordinates_path)
    df_gsp_coordinates = df_gsp_coordinates.apply(strip_str)
    extra_gsp_coordinates_df = (
        pd.DataFrame.from_dict(extra_gsp_coordinates, orient="index")
        .rename_axis(index="Name")
        .reset_index()
    )
    df_gsp_coordinates = (
        # There are cases of duplicate GSPs where the lat and lon information is the same but the GSP ID and GSP group are slightly different
        pd.concat([df_gsp_coordinates, extra_gsp_coordinates_df])
        .drop_duplicates(subset=["Name", "Latitude", "Longitude"])
        .dropna(subset=["Latitude", "Longitude"])
        .set_index("Name")
        .reset_index()
    )
    if (dups := df_gsp_coordinates.Name.duplicated()).any():
        logger.error(
            f"There are duplicate GSP names with different lat/lons in the GSP coordinates file:\n{df_gsp_coordinates[dups]}"
        )

    logger.info(
        f"Loaded GSP coordinates data with {len(df_gsp_coordinates)} unique GSPs"
    )

    return df_gsp_coordinates


def process_bb1_data(
    bb1_path: str, fes_scenario: str, year_range: list
) -> pd.DataFrame:
    """
    Process FES workbook BB1 data, filtering by scenario and year, and summing duplicates

    Parameters
    ----------
    bb1_path: str
        Path to the extracted BB1 sheet of the FES workbook
    fes_scenario: str
        FES scenario
    year_range: list
        Year range to filter the data by
    """
    df_bb1 = pd.read_csv(bb1_path)
    df_bb1 = df_bb1.apply(strip_str)
    df_bb1_scenario = df_bb1[
        (df_bb1["FES Pathway"].str.lower() == fes_scenario.lower())
        & (df_bb1["year"].between(year_range[0], year_range[1], inclusive="both"))
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

    return df_bb1_scenario_no_dups


def parse_inputs(
    bb1_path: str,
    bb2_path: str,
    es1_path: str,
    manual_gsp_mapping: dict,
    fes_scenario: str,
    year_range: list,
    df_gsp_coordinates: pd.DataFrame,
) -> pd.DataFrame:
    """
    Parse the input data to the required format.

    Args:
        bb1_path (str): path of extracted sheet BB1 of the FES workbook
        bb2_path (str): path of extracted sheet BB2 of the FES workbook
        es1_path (str): path of extracted sheet ES1 of the FES workbook
        df_gsp_coordinates (pd.DataFrame): DataFrame of GSP supply point coordinates
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
        .apply(strip_str)
    )

    df_bb1_scenario_no_dups = process_bb1_data(bb1_path, fes_scenario, year_range)

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
        "Mapping of building blocks between BB1 and BB2 may be incorrect as some units do not match:\n"
        f"{df_bb1_bb2_scenario[~units_match][['Unit', 'Units']]}"
    )

    df_bb1_bb2_scenario = df_bb1_bb2_scenario.drop(columns=["Units"])

    df_bb1_bb2_scenario["GSP"] = df_bb1_bb2_scenario["GSP"].replace(manual_gsp_mapping)

    df_bb1_bb2_with_lat_lon = pd.merge(
        df_bb1_bb2_scenario,
        df_gsp_coordinates,
        left_on="GSP",
        right_on="Name",
    )

    # Missing data checks.
    # We won't raise errors here as we are willing to accept some missing data for now
    missing_lat_lon = df_bb1_bb2_with_lat_lon[
        df_bb1_bb2_with_lat_lon[["Latitude", "Longitude"]].isnull().any(axis=1)
    ].GSP.unique()
    if len(missing_lat_lon) > 0:
        raise ValueError(
            f"The following GSPs are missing latitude and/or longitude information: {missing_lat_lon}.\n"
            "Please update the GSP coordinates file or provide extra coordinates via the `grid-supply_points.fill_lat_lons` configuration option."
        )

    missing_gsps = set(df_bb1_bb2_scenario.GSP).difference(df_bb1_bb2_with_lat_lon.GSP)
    if missing_gsps:
        logger.warning(
            f"The following GSPs are missing from the GSP coordinates file: {missing_gsps}."
            "Their data will be distributed later across other GSPs in the same TO region or across the whole country."
        )
    df_final = pd.concat(
        [df_bb1_bb2_scenario.query("GSP in @missing_gsps"), df_bb1_bb2_with_lat_lon]
    )

    return df_final


def split_technologies(
    df_with_regions: pd.DataFrame,
    df_es1: pd.DataFrame,
    technology_mapping: dict,
    fes_scenario: str,
    year_range: list[int],
    allowed_mismatch: float,
) -> pd.DataFrame:
    """
    To split technologies based on subtypes present in ES1 sheet of the FES workbook

    Parameters
    ----------
    df_with_regions: pd.DataFrame
        Pandas dataframe to modify
    df_es1: pd.DataFrame
        Pandas dataframe of FES workbook ES1 sheet
    technology_mappingL dict[str, list[str]]
        Dictionary to map technologies in BB1 sheet to ES1 sheet
    fes_scenario: str
        FES scenario
    year_range: list[int]
        Year range of the simulation
    allowed_mismatch: float
        Mismatch in total capacity values between the BB1 and ES1 sheet entries for a technology
    """

    # Filter ES1 sheet
    df_es1_reqd = df_es1[
        (df_es1["Pathway"].str.lower() == fes_scenario.lower())
        & (df_es1["year"].between(year_range[0], year_range[1], inclusive="both"))
        & (df_es1["Variable"] == "Capacity (MW)")
    ]

    df_with_regions_updated = df_with_regions.copy()
    # Iterate through the technologies with more subtypes in ES1 sheet
    for tech in technology_mapping.keys():
        df_tech = df_with_regions.query("`Building Block ID Number` == @tech")

        mapped_tech = technology_mapping[tech]

        if not set(mapped_tech).issubset(df_es1_reqd["SubType"]):
            logger.error(
                f"One/more technologies in {mapped_tech} might not match with the SubType technology list in the ES1 workbook sheet."
            )

        df_es1_tech = pd.DataFrame(
            df_es1_reqd.query("SubType in @tech", local_dict={"tech": mapped_tech})
            .groupby(["SubType", "year"])["data"]
            .sum()
        )

        # Calculate %share of each technology subtype for every year
        df_es1_tech["pct"] = (
            df_es1_tech["data"] / df_es1_tech.groupby("year")["data"].sum()
        )

        # Merge the original dataframe indexed from BB1 sheet and the data from ES1 sheet
        df_tech = df_tech.merge(df_es1_tech.reset_index(), on="year")

        # Multiply the regional data with the percentage share of the technology subtype
        df_tech["data"] = df_tech["data_x"].mul(df_tech["pct"])
        df_tech["Technology"] = df_tech["SubType"]

        # Scaling factor to scale the mismatch in total capacity values in BB1 and ES1 sheet
        # The factor scales the values to match the value in the ES1 sheet
        es1_grouped = df_es1_tech.groupby("year")["data"].sum()
        tech_grouped = df_tech.groupby("year")["data"].sum()

        # Calculate % diff in the total capacity of the technology
        df_diff = (es1_grouped - tech_grouped) * 100 / es1_grouped

        if df_diff.mean() > allowed_mismatch:
            logger.warning(
                f"The percentage difference in capacity data for the {tech}, indexed by year in ES1 and BB1 sheet is {df_diff}"
            )

        scaling_factor = es1_grouped / tech_grouped
        scaling_factor.name = "scaling_factor"
        df_tech = df_tech.merge(scaling_factor, on="year")

        df_tech["scaled_data"] = df_tech["data"].mul(df_tech["scaling_factor"])

        df_tech["data"] = df_tech["scaled_data"]

        df_with_regions_updated = pd.concat(
            [
                df_with_regions_updated.query("Technology != @tech"),
                df_tech[df_with_regions_updated.columns],
            ]
        )

    return df_with_regions_updated


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    fes_scenario = get_scenario_name(snakemake)
    gdf_regions = gpd.read_file(snakemake.input.regions)

    df_gsp_coordinates = process_gsp_coordinates(
        gsp_coordinates_path=snakemake.input.gsp_coordinates,
        extra_gsp_coordinates=snakemake.params.fill_gsp_lat_lons,
    )

    df = parse_inputs(
        bb1_path=snakemake.input.bb1_sheet,
        bb2_path=snakemake.input.bb2_sheet,
        es1_path=snakemake.input.es1_sheet,
        df_gsp_coordinates=df_gsp_coordinates,
        manual_gsp_mapping=snakemake.params.manual_gsp_mapping,
        fes_scenario=fes_scenario,
        year_range=snakemake.params.year_range,
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
    if (null_bus := df_with_regions.bus.isnull()).any():
        warning_data = df_with_regions[null_bus][
            ["GSP", "Latitude", "Longitude", "TO_region"]
        ].drop_duplicates()
        logger.warning(
            f"There are GSPs with missing bus/region information after mapping lat/lon to regions:\n{warning_data}"
        )
    logger.info(f"Extracted the {fes_scenario} relevant data")

    df_es1 = pd.read_csv(snakemake.input.es1_sheet)

    df_with_regions_updated = split_technologies(
        df_with_regions=df_with_regions,
        df_es1=df_es1,
        technology_mapping=snakemake.params.bb2_es1_mapping,
        fes_scenario=fes_scenario,
        year_range=snakemake.params.year_range,
        allowed_mismatch=snakemake.params.allowed_mismatch,
    )

    df_with_regions_updated.to_csv(snakemake.output.csv, index=False)
    logger.info(
        f"Exported processed GSP-level powerplant information to {snakemake.output.csv}"
    )

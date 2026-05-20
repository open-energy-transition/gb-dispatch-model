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
import linopy
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


def _optimise_allocation(df_bb1, df_es1, bb_mapping, tolerance):
    bb1_national = df_bb1.groupby(["Building Block ID Number", "year"])["data"].sum()
    es1_national = df_es1.groupby(["SubType", "year"]).data.sum()
    years = sorted(df_bb1["year"].unique())

    mask = (
        pd.Series(bb_mapping)
        .apply(lambda x: pd.Series(1, index=x))
        .fillna(False)
        .astype(bool)
        .rename_axis(index="Building Block ID Number", columns="SubType")
    )
    m = linopy.Model()
    m.add_variables(
        name="allocation",
        lower=0.0,
        mask=mask,
        coords=[mask.index, mask.columns, pd.Index(years, name="year")],
    )
    m_bb1 = m["allocation"].sum("SubType")
    m_es1 = m["allocation"].sum("Building Block ID Number")
    bb1_national_da = bb1_national.to_xarray().sel(m_bb1.coords)
    es1_national_da = es1_national.to_xarray().sel(m_es1.coords)
    m.add_variables(
        name="deviation_bb1",
        lower=0.0,
        mask=mask.any(axis=1),
        coords=[mask.index, pd.Index(years, name="year")],
    )
    m.add_variables(
        name="deviation_es1",
        lower=0.0,
        mask=mask.any(axis=0),
        coords=[mask.columns, pd.Index(years, name="year")],
    )
    m.add_constraints(m_bb1 - m["deviation_bb1"] <= bb1_national_da, name="bb1_upper")
    m.add_constraints(m_es1 - m["deviation_es1"] <= es1_national_da, name="es1_upper")
    m.add_constraints(m_bb1 + m["deviation_bb1"] >= bb1_national_da, name="bb1_lower")
    m.add_constraints(m_es1 + m["deviation_es1"] >= es1_national_da, name="es1_lower")
    m.add_objective(m["deviation_bb1"].sum() + m["deviation_es1"].sum(), sense="min")
    m.solve(solver_name="highs")
    allocation = m["allocation"].solution
    es1_sum = es1_national_da.where(mask.any(axis=0).to_xarray()).sum()
    bb1_sum = bb1_national_da.where(mask.any(axis=1).to_xarray()).sum()
    if (
        abs((allocation.sum() - es1_sum) / es1_sum) > tolerance
        or abs((allocation.sum() - bb1_sum) / bb1_sum) > tolerance
    ):
        message = f"Failed to map FES building block regional data to national technology data within specified tolerance of {tolerance:.1%}."
        raise ValueError(message)
    return allocation


def split_technologies(
    df_with_regions: pd.DataFrame,
    df_es1: pd.DataFrame,
    tech_config: dict,
    fes_scenario: str,
    year_range: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Split building block capacities into technology subtypes using ES1 national totals.

    Three steps:

    1. Aggregate BB1 data to national (BB ID, year) totals, ignoring GSP detail.
    2. Find the optimal (BB ID x SubType) capacity allocation at the national level
       using Iterative Proportional Fitting (IPF/RAS). Row marginals are the BB1
       national totals; column marginals are the ES1 national totals. Trivial cases
       (one SubType per BB, or one BB per SubType) are handled exactly; non-trivial
       cases converge to the minimum-information allocation consistent with both sets
       of marginals.
    3. Distribute each (BB ID, SubType) national allocation regionally using the GSP
       fractional distribution of that BB ID from BB1.

    Only BB IDs present in both `tech_config["building_block_mapping"]` (with a
    non-empty subtype list) and in `df_with_regions` are processed. This allows the
    caller to pre-filter `df_with_regions` to avoid double-counting across configs.

    Parameters
    ----------
    df_with_regions: pd.DataFrame
        DataFrame with columns including 'Building Block ID Number', 'GSP',
        'year', 'data', and all other metadata columns from BB1/BB2.
    df_es1: pd.DataFrame
        DataFrame of FES workbook ES1 sheet with columns 'Pathway', 'year',
        'Variable', 'SubType', 'data'.
    tech_config: dict
        Configuration dict with keys:
        - 'building_block_mapping': {bb_id: [subtype, ...]} mapping
        - 'building_block_mapping_tolerance': fractional tolerance for totals check
    fes_scenario: str
        FES scenario name (matched case-insensitively to 'Pathway' column).
    year_range: list[int]
        Two-element [start, end] year range (inclusive).

    Returns
    -------
    pd.DataFrame
        DataFrame with the same structure as `df_with_regions` but with rows for the relevant BB IDs replaced by rows for each (BB ID, SubType) combination, and 'data' values allocated according to the ES1 national totals and the BB1 regional distribution.
    list[str]
        List of BB IDs that were split into subtypes.
    """
    tolerance = tech_config["building_block_mapping_tolerance"]
    bb_mapping: dict[str, list[str]] = {
        bb_id: subtypes
        for bb_id, subtypes in tech_config["building_block_mapping"].items()
        if subtypes
    }

    bb_list = sorted(bb_mapping.keys())
    subtype_list = sorted(set().union(*bb_mapping.values()))
    # --- Step 1: aggregate BB1 to national (BB ID, year) totals ---
    df_active = df_with_regions[
        df_with_regions["Building Block ID Number"].isin(bb_list)
    ]

    df_es1_reqd = df_es1[
        (df_es1["Pathway"].str.lower() == fes_scenario.lower())
        & (df_es1["year"].between(year_range[0], year_range[1], inclusive="both"))
        & (df_es1["Variable"] == "Capacity (MW)")
        & (df_es1["SubType"].isin(subtype_list))
    ]
    allocation = _optimise_allocation(df_active, df_es1_reqd, bb_mapping, tolerance)

    idx_cols = ["Building Block ID Number", "year", "GSP"]
    bb1_regional = df_active.groupby(idx_cols)["data"].sum().to_xarray()
    bb1_regional_frac = bb1_regional / bb1_regional.sum("GSP")
    es1_regional = (allocation * bb1_regional_frac).to_series().dropna()
    results = df_active.drop_duplicates(idx_cols).merge(
        es1_regional.to_frame("data").reset_index(),
        on=idx_cols,
        suffixes=("", "_y"),
        how="right",
    )
    results = results.assign(data=results["data_y"]).drop(columns=["data_y"])
    return results, bb_list


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

    tech_split, split_bb = split_technologies(
        df_with_regions=df_with_regions,
        df_es1=df_es1,
        tech_config=snakemake.params.technology_mapping,
        fes_scenario=fes_scenario,
        year_range=snakemake.params.year_range,
    )
    # Keep rows for BBs that have no subtype mapping in either config
    df_unsplit = df_with_regions.query("`Building Block ID Number` not in @split_bb")

    df_with_regions_updated = pd.concat([df_unsplit, tech_split], ignore_index=True)

    df_with_regions_updated.to_csv(snakemake.output.csv, index=False)
    logger.info(
        f"Exported processed GSP-level powerplant information to {snakemake.output.csv}"
    )

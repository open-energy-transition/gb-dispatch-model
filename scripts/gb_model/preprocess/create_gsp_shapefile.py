# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


"""
GSP-level data table generator.

This is a script to combine the BB1 sheet with the BB2 (metadata) sheet of the FES workbook.
"""

import logging
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from scripts._helpers import configure_logging, set_scenario_config
from scripts.gb_model._helpers import (
    get_scenario_name,
)
from scripts.gb_model.preprocess.process_fes_gsp_data import (
    process_bb1_data,
    process_gsp_coordinates,
)

logger = logging.getLogger(__name__)


def _merge_gsps(df: pd.DataFrame, gsps: str, key: str) -> pd.DataFrame:
    """
    Merge multiple GSPs into a single GSP, by dissolving the geometries

    Parameters
    ----------
    df: pd.DataFrame
        The dataframe containing the GSPs to merge
    gsps: str
        The GSPs to merge, as a single string with "|" separating the different GSPs (e.g. "GSP1|GSP2|GSP3")s
    key: str
        column to merge the GSPs by
    """

    # All occurrences of the GSPs to merge, i.e. all rows where any of the GSPs to merge are mentioned in the "GSPs" column (there may be multiple rows for each GSP if there are multiple GSPs to merge)
    all_occurrences = df.loc[
        (df[key].str.contains(rf"({gsps})\b", regex=True)) & (df[key].notna())
    ]

    if len(all_occurrences) > 1:
        # Dissolve the rows of geometries matching the GSPs to merge into a single row
        if key == "GSP":
            geo_key = "geometryshape"
        else:
            geo_key = "geometrycoord"
        df.loc[df[key] == gsps, geo_key] = (
            df.loc[all_occurrences.index]
            .set_geometry(geo_key)
            .dissolve()
            .iloc[0][geo_key]
        )

        if key == "GSPs":
            # Assign the GSP name as the concatenation of the GSP names of the merged GSPs, separated by "|"
            df.loc[df[key] == gsps, "GSP"] = all_occurrences[
                all_occurrences[key] != gsps
            ].GSP.str.cat(sep="|")

        # Drop the other rows of the merged GSPs, keeping only the dissolved row
        indices = all_occurrences.index.tolist()
        indices_filter = [
            x
            for x in all_occurrences.index
            if x not in df.loc[df[key] == gsps].index.tolist()
        ]
        if len(indices_filter) == len(indices):
            # For merging the GSPs to combine busbars, the earlier filter will not work
            retain_row = gsps.split("|")[-1]
            indices_filter = [
                x
                for x in all_occurrences.loc[
                    all_occurrences["GSPs"] != retain_row
                ].index.tolist()
            ]
        df.drop(index=indices_filter, inplace=True)

    return df


def create_gsp_shapefile(
    df_gsp_coordinates: pd.DataFrame,
    df_gsp_shapes: gpd.GeoDataFrame,
    df_bb1: pd.DataFrame,
    gsp_mapping: dict,
    combine_gsps: dict,
):
    """
    Create a GSP shapefile by combining FES BB1 sheet data, GSP coordinate data and GSP shape data

    Parameters
    ----------
    df_gsp_coordinates: pd.DataFrame
        The GSP coordinate data dataframe
    df_gsp_shape: gpd.GeoDataFrame
        GSP polygon shape data
    df_bb1: pd.DataFrame
        FES BB1 sheet dataframe
    gsp_mapping: dict
        Manual mapping of GSP names between the FES workbook and the GSP coordinate/shape data
    combine_gsps: dict
        Groups of GSPs to combine
    """

    # Convert the GSP coordinate data to a GeoDataFrame
    gdf_gsps = gpd.GeoDataFrame(
        df_gsp_coordinates,
        geometry=gpd.points_from_xy(
            df_gsp_coordinates.Longitude, df_gsp_coordinates.Latitude
        ),
        crs="EPSG:4326",
    )

    df_bb1_gsp = pd.DataFrame(data=df_bb1.GSP.unique(), columns=["GSP"])
    df_bb1_gsp["GSP"] = df_bb1_gsp["GSP"].replace(gsp_mapping)

    # Join GSP shape data with GSP coordinate data
    gsp_joined = df_gsp_shapes.set_index("GSPs").join(
        gdf_gsps.set_index("GSP ID"), lsuffix="shape", rsuffix="coord", how="outer"
    )

    fes_merged = pd.merge(
        df_bb1_gsp,
        gsp_joined.reset_index(),
        left_on="GSP",
        right_on="Name",
        how="outer",
    )

    # Drop unrequired columns
    fes_merged.drop(
        columns=["GSP Group", "Minor FLOP", "Latitude", "Longitude"], inplace=True
    )

    # Some GSPs have not been matched with shape data as they are part of a geometry shape containing multiple GSPs
    unmatched_gsps = fes_merged.loc[
        (fes_merged.geometrycoord.isna()) & (fes_merged.GSPs.notna())
    ]
    for gsps in unmatched_gsps["GSPs"].tolist():
        fes_merged = _merge_gsps(fes_merged, gsps, "GSPs")
    logger.info(
        "Merged GSP shape and coordinate data for GSPs that are part of a combined geometry shape in the FES workbook"
    )

    # Some busbars are split into multiple GSPs in the FES workbook but represented as a single GSP in shape data
    for key in combine_gsps.keys():
        combine_gsps[key].append(key)
        gsps = "|".join(combine_gsps[key])
        fes_merged = _merge_gsps(fes_merged, gsps, "GSPs")
    logger.info(
        "Merged GSP shape and coordinate data for busbars that are split into multiple GSPs in the FES workbook but represented as a single GSP in shape data"
    )

    # Some GSPs where coordinate data is available but shape data was not matched
    unmatched_gsp_name = fes_merged.loc[
        (fes_merged.geometryshape.isna()) & (fes_merged.GSPs.notna())
    ]
    for gsps in unmatched_gsp_name["GSP"].tolist():
        fes_merged = _merge_gsps(fes_merged, gsps, "GSP")
    logger.info(
        "Merged GSP shape and coordinate data for GSPs where coordinate data is available but shape data was not matched"
    )

    fes_merged = gpd.GeoDataFrame(fes_merged, geometry="geometryshape", crs="EPSG:4326")

    # Merging GSPs with same GSP ID but different GSP groups.
    # Dissolving the shapes into a single GSP as the rows still have distinct geometries though adjoining each other
    fes_merged_notnan = fes_merged[fes_merged["geometryshape"].notna()]
    fes_merged_nan = fes_merged[fes_merged["geometryshape"].isna()]
    fes_merged_notnan = fes_merged_notnan.dissolve(by="GSPs", as_index=False)
    fes_merged = pd.concat([fes_merged_notnan, fes_merged_nan], ignore_index=True)
    logger.info("Merged GSP duplicates due to different GSP groups")

    fes_merged["geometrycoord"] = fes_merged["geometrycoord"].to_wkt()

    if (missing_shapes := fes_merged.geometryshape.isna()).any():
        logger.warning(
            f"There are {missing_shapes.sum()} GSPs with missing shape information after merging the GSP shape and coordinate data. These GSPs will be kept in the output but with null geometry.\n"
            f"{fes_merged[missing_shapes][['GSP', 'GSPs']]}"
        )

    return fes_merged


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem)
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    fes_scenario = get_scenario_name(snakemake)

    df_gsp_coordinates = process_gsp_coordinates(
        gsp_coordinates_path=snakemake.input.gsp_coordinates,
        extra_gsp_coordinates=snakemake.params.fill_gsp_lat_lons,
    )

    df_bb1 = process_bb1_data(
        bb1_path=snakemake.input.bb1_sheet,
        fes_scenario=fes_scenario,
        year_range=snakemake.params.year_range,
    )

    # Read the GSP shapefile
    CRS = 4326
    zip_path = Path(snakemake.input.gsp_shapes)
    shp_filename = [
        x
        for x in zipfile.ZipFile(zip_path).namelist()
        if bool(re.search(rf"Proj_{CRS}/.*_{CRS}_.*\.geojson$", x))
    ][0]
    df_gsp_shapes = gpd.read_file(f"{zip_path}!{shp_filename}")

    shape = create_gsp_shapefile(
        df_gsp_coordinates,
        df_gsp_shapes,
        df_bb1,
        gsp_mapping=snakemake.params.manual_gsp_mapping,
        combine_gsps=snakemake.params.combine_gsps,
    )

    logger.info(f"Exported the GSP shapefile to {snakemake.output.shapefile}")
    shape.to_file(snakemake.output.shapefile, driver="GeoJSON")

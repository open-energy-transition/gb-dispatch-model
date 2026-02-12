# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Preprocessing rules for downloading/extracting key data and preparing the region shapes.
"""

from pathlib import Path


rule download_data:
    message:
        "Download {wildcards.gb_data} GB model data."
    output:
        downloaded="data/gb-model/downloaded/{gb_data}",
    params:
        url=lambda wildcards: config["urls"][Path(wildcards.gb_data).stem],
    log:
        logs("download_{gb_data}.log"),
    localrule: True
    shell:
        "curl -sSLo {output} {params.url}"


rule create_region_shapes:
    input:
        country_shapes=resources("country_shapes.geojson"),
        etys_boundary_lines="data/gb-model/downloaded/gb-etys-boundaries.zip",
        etys_focus_boundary_lines=resources("gb-model/etys_boundary_capabilities.csv"),
    output:
        raw_region_shapes=resources("gb-model/raw_region_shapes.geojson"),
    params:
        pre_filter_boundaries=config["region_operations"][
            "filter_boundaries_using_capabilities"
        ],
        area_loss_tolerance_percent=config["region_operations"][
            "area_loss_tolerance_percent"
        ],
        min_region_area=config["region_operations"]["min_region_area"],
    log:
        logs("raw_region_shapes.log"),
    resources:
        mem_mb=1000,
    script:
        scripts("gb_model/preprocess/create_region_shapes.py")


rule manual_region_merger:
    input:
        raw_region_shapes=rules.create_region_shapes.output.raw_region_shapes,
        country_shapes=resources("country_shapes.geojson"),
    output:
        merged_shapes=resources("gb-model/merged_shapes.geojson"),
    params:
        splits=config["region_operations"]["splits"],
        merge_groups=config["region_operations"]["merge_groups"],
        add_group_to_neighbour=config["region_operations"]["add_group_to_neighbour"],
    log:
        logs("manual_region_merger.log"),
    resources:
        mem_mb=1000,
    script:
        scripts("gb_model/preprocess/manual_region_merger.py")


rule extract_fes_workbook_sheet:
    message:
        "Extract FES workbook sheet {wildcards.fes_sheet} for FES and process into machine-readable, 'tidy' dataframe format according to defined configuration."
    input:
        workbook="data/gb-model/downloaded/fes-workbook.xlsx",
    output:
        csv=resources("gb-model/fes/{fes_sheet}.csv"),
    params:
        sheet_extract_config=lambda wildcards: config["fes-sheet-config"][
            wildcards.fes_sheet
        ],
    log:
        logs("extract_fes_workbook_sheet-{fes_sheet}.log"),
    script:
        scripts("gb_model/preprocess/extract_fes_workbook_sheet.py")


rule unzip_fes_costing_workbook:
    message:
        "Unzip FES costing workbook"
    input:
        "data/gb-model/downloaded/fes-costing-workbook.zip",
    output:
        "data/gb-model/fes-costing-workbook.xlsx",
    shell:
        "unzip -p {input} 'FES20 Costing Workbook (1).xlsx' > {output}"


use rule extract_fes_workbook_sheet as extract_fes_costing_workbook_sheet with:
    message:
        "Extract FES costing workbook sheet {wildcards.fes_sheet} and process into machine-readable, 'tidy' dataframe format according to defined configuration."
    input:
        workbook="data/gb-model/fes-costing-workbook.xlsx",
    output:
        csv=resources("gb-model/fes-costing/{fes_sheet}.csv"),
    params:
        sheet_extract_config=lambda wildcards: config["fes-costing-sheet-config"][
            wildcards.fes_sheet
        ],
    log:
        logs("extract_fes_costing_workbook_sheet-{fes_sheet}.log"),


rule process_fes_eur_data:
    message:
        "Process FES-compatible European scenario workbook."
    params:
        year_range=config["redispatch"]["year_range_incl"],
        countries=config["countries"],
    input:
        eur_data=resources("gb-model/fes/ES2.csv"),
    output:
        csv=resources("gb-model/{fes_scenario}/national_eur_data.csv"),
    log:
        logs("process_fes_eur_data_{fes_scenario}.log"),
    script:
        scripts("gb_model/preprocess/process_fes_eur_data.py")


rule process_dukes_current_capacities:
    message:
        "Assign current capacities to GB model regions and PyPSA-Eur carriers"
    input:
        regions=resources("gb-model/merged_shapes.geojson"),
        regions_offshore=resources("regions_offshore_base_s_clustered.geojson"),
        dukes_data="data/gb-model/downloaded/dukes-5.11.xlsx",
    output:
        csv=resources("gb-model/dukes-current-capacity.csv"),
    log:
        logs("process_dukes_current_capacities.log"),
    params:
        sheet_config=config["dukes-5.11"]["sheet-config"],
        target_crs=config["target_crs"],
    script:
        scripts("gb_model/preprocess/process_dukes_current_capacities.py")


rule process_fes_gsp_data:
    message:
        "Process FES workbook sheet BB1 together with metadata from sheet BB2."
    params:
        year_range=config["redispatch"]["year_range_incl"],
        target_crs=config["target_crs"],
        fill_gsp_lat_lons=config["grid_supply_points"]["fill-lat-lons"],
        manual_gsp_mapping=config["grid_supply_points"]["manual_mapping"],
    input:
        bb1_sheet=resources(f"gb-model/fes/BB1.csv"),
        bb2_sheet=resources(f"gb-model/fes/BB2.csv"),
        gsp_coordinates="data/gb-model/downloaded/gsp-coordinates.csv",
        regions=resources("gb-model/merged_shapes.geojson"),
    output:
        csv=resources("gb-model/{fes_scenario}/regional_gb_data.csv"),
    log:
        logs("process_fes_gsp_data_{fes_scenario}.log"),
    script:
        scripts("gb_model/preprocess/process_fes_gsp_data.py")

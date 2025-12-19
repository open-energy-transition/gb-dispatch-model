# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


import os
import subprocess
from zipfile import ZipFile
from pathlib import Path
import numpy as np


# Rule to download and extract ETYS boundary data
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


rule extract_etys_boundary_capabilities:
    message:
        "Extract boundary capability data from ETYS PDF report"
    input:
        pdf_report="data/gb-model/downloaded/etys.pdf",
        boundaries="data/gb-model/downloaded/gb-etys-boundaries.zip",
    output:
        csv=resources("gb-model/etys_boundary_capabilities.csv"),
    log:
        logs("extract_etys_boundary_capabilities.log"),
    script:
        "../scripts/gb_model/extract_etys_boundary_capabilities.py"


# Rule to create region shapes using create_region_shapes.py
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
        "../scripts/gb_model/create_region_shapes.py"


# Rule to manually merge raw_region_shapes
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
        "../scripts/gb_model/manual_region_merger.py"


# Rule to retrieve generation unit unavailability data from ENTSO-E
rule retrieve_entsoe_unavailability_data:
    message:
        "Retrieve data from ENTSOE API for generator {wildcards.business_type} unavailability in {wildcards.zone} bidding zone"
    output:
        xml_base_dir=directory("data/gb-model/entsoe_api/{zone}/{business_type}"),
    params:
        start_date=config["entsoe_unavailability"]["start_date"],
        end_date=config["entsoe_unavailability"]["end_date"],
        bidding_zones=config["entsoe_unavailability"]["bidding_zones"],
        business_types=config["entsoe_unavailability"]["business_types"],
        max_request_days=config["entsoe_unavailability"]["max_request_days"],
        api_params=config["entsoe_unavailability"]["api_params"],
    log:
        logs("retrieve_entsoe_unavailability_data_{zone}_{business_type}.log"),
    resources:
        mem_mb=1000,
    script:
        "../scripts/gb_model/retrieve_entsoe_unavailability_data.py"


rule process_entsoe_unavailability_data:
    input:
        xml_base_dir="data/gb-model/entsoe_api/{zone}/{business_type}",
    output:
        unavailability=resources(
            "gb-model/{zone}_{business_type}_generator_unavailability.csv"
        ),
    log:
        logs("process_entsoe_unavailability_data_{zone}_{business_type}.log"),
    params:
        business_type_codes=config["entsoe_unavailability"]["api_params"][
            "business_types"
        ],
    resources:
        mem_mb=1000,
    script:
        "../scripts/gb_model/process_entsoe_unavailability_data.py"


rule generator_monthly_availability_fraction:
    input:
        planned=resources("gb-model/{zone}_planned_generator_unavailability.csv"),
        forced=resources("gb-model/{zone}_forced_generator_unavailability.csv"),
        powerplants=resources("powerplants_s_all.csv"),
    params:
        carrier_mapping=config["entsoe_unavailability"]["carrier_mapping"],
        resource_type_mapping=config["entsoe_unavailability"]["resource_type_mapping"],
        start_date=config["entsoe_unavailability"]["start_date"],
        end_date=config["entsoe_unavailability"]["end_date"],
        max_unavailable_days=config["entsoe_unavailability"]["max_unavailable_days"],
    output:
        csv=resources("gb-model/{zone}_generator_monthly_availability_fraction.csv"),
    log:
        logs("{zone}_generator_monthly_availability_fraction.log"),
    script:
        "../scripts/gb_model/generator_monthly_availability_fraction.py"


rule extract_transmission_availability:
    input:
        pdf_report="data/gb-model/downloaded/transmission-availability-{report_year}.pdf",
    output:
        csv=resources("gb-model/transmission-availability-monthly/{report_year}.csv"),
    log:
        logs("extract_transmission_availability_{report_year}.log"),
    script:
        "../scripts/gb_model/extract_transmission_availability.py"


rule process_transmission_availability:
    message:
        "Process {wildcards.transmission_zone} transmission availability stats into timeseries availability fractions."
    input:
        unavailability=expand(
            resources("gb-model/transmission-availability-monthly/{year}.csv"),
            year=config["transmission_availability"]["years"],
        ),
    output:
        csv=resources("gb-model/{transmission_zone}_transmission_availability.csv"),
    params:
        zones=lambda wildcards: config["transmission_availability"][
            wildcards.transmission_zone
        ]["zones"],
        sample_hourly=lambda wildcards: config["transmission_availability"][
            wildcards.transmission_zone
        ]["sample_hourly"],
        random_seeds=config["transmission_availability"]["random_seeds"],
    wildcard_constraints:
        transmission_zone="inter_gb|intra_gb",
    log:
        logs("process_transmission_availability_{transmission_zone}.log"),
    script:
        "../scripts/gb_model/process_transmission_availability.py"


rule extract_fes_workbook_sheet:
    message:
        "Extract FES workbook sheet {wildcards.fes_sheet} for FES-{wildcards.fes_year} and process into machine-readable, 'tidy' dataframe format according to defined configuration."
    input:
        workbook="data/gb-model/downloaded/fes-{fes_year}-workbook.xlsx",
    output:
        csv=resources("gb-model/fes/{fes_year}/{fes_sheet}.csv"),
    params:
        sheet_extract_config=lambda wildcards: config["fes-sheet-config"][
            int(wildcards.fes_year)
        ][wildcards.fes_sheet],
    log:
        logs("extract_fes_workbook_sheet-{fes_year}_{fes_sheet}.log"),
    script:
        "../scripts/gb_model/extract_fes_workbook_sheet.py"


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
        scenario=config["fes"]["eur"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        countries=config["countries"],
    input:
        eur_supply="data/gb-model/downloaded/eur-supply-table.csv",
    output:
        csv=resources("gb-model/national_eur_data.csv"),
    log:
        logs("process_fes_eur_data.log"),
    script:
        "../scripts/gb_model/process_fes_eur_data.py"


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
        "../scripts/gb_model/process_dukes_current_capacities.py"


rule process_fes_gsp_data:
    message:
        "Process FES workbook sheet BB1 together with metadata from sheet BB2."
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        target_crs=config["target_crs"],
        fill_gsp_lat_lons=config["fill-gsp-lat-lons"],
    input:
        bb1_sheet=resources("gb-model/fes/2021/BB1.csv"),
        bb2_sheet=resources("gb-model/fes/2021/BB2.csv"),
        gsp_coordinates="data/gb-model/downloaded/gsp-coordinates.csv",
        regions=resources("gb-model/merged_shapes.geojson"),
    output:
        csv=resources("gb-model/regional_gb_data.csv"),
    log:
        logs("process_fes_gsp_data.log"),
    script:
        "../scripts/gb_model/process_fes_gsp_data.py"


rule create_powerplants_table:
    message:
        "Tabulate powerplant data GSP-wise from FES workbook sheet BB1 and EU supply data"
    params:
        gb_config=config["fes"]["gb"],
        eur_config=config["fes"]["eur"],
        dukes_config=config["dukes-5.11"],
        default_set=config["fes"]["default_set"],
    input:
        gsp_data=resources("gb-model/regional_gb_data.csv"),
        eur_data=resources("gb-model/national_eur_data.csv"),
        dukes_data=resources("gb-model/dukes-current-capacity.csv"),
    output:
        csv=resources("gb-model/fes_powerplants.csv"),
    log:
        logs("create_powerplants_table.log"),
    script:
        "../scripts/gb_model/create_powerplants_table.py"


rule create_interconnectors_table:
    input:
        regions=resources("gb-model/merged_shapes.geojson"),
    output:
        gsp_data=resources("gb-model/interconnectors_p_nom.csv"),
    params:
        interconnector_config=config["interconnectors"],
        year_range=config["fes"]["year_range_incl"],
        target_crs=config["target_crs"],
    log:
        logs("create_interconnectors_table.log"),
    script:
        "../scripts/gb_model/create_interconnectors_table.py"


rule create_hydrogen_demand_table:
    message:
        "Process hydrogen demand data from FES workbook into CSV format"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        fes_demand_sheets=config["fes"]["hydrogen"]["demand"]["annual_demand_sheets"],
        other_sectors_list=config["fes"]["hydrogen"]["demand"]["other_sectors_list"],
    input:
        demand_sheets=lambda wildcards: [
            resources(f"gb-model/fes/{year}/{sheet}.csv")
            for year, sheets in config["fes"]["hydrogen"]["demand"][
                "annual_demand_sheets"
            ].items()
            for sheet in sheets.values()
        ],
    output:
        hydrogen_demand=resources("gb-model/fes_hydrogen_demand.csv"),
    log:
        logs("create_hydrogen_demand_table.log"),
    script:
        "../scripts/gb_model/create_hydrogen_demand_table.py"


rule create_grid_electrolysis_table:
    message:
        "Process hydrogen electrolysis data from FES workbook into CSV format"
    input:
        regional_gb_data=resources("gb-model/regional_gb_data.csv"),
    output:
        csv=resources("gb-model/regional_grid_electrolysis_capacities.csv"),
    log:
        logs("create_grid_electrolysis_table.log"),
    script:
        "../scripts/gb_model/create_grid_electrolysis_table.py"


rule create_hydrogen_supply_table:
    message:
        "Process hydrogen supply data from FES workbook into CSV format"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        fes_supply_sheets=config["fes"]["hydrogen"]["supply"]["supply_sheets"],
        exogeneous_supply_list=config["fes"]["hydrogen"]["supply"][
            "exogeneous_supply_list"
        ],
    input:
        supply_sheets=lambda wildcards: [
            resources(f"gb-model/fes/{year}/{sheet}.csv")
            for year, sheets in config["fes"]["hydrogen"]["supply"][
                "supply_sheets"
            ].items()
            for sheet in sheets.values()
        ],
    output:
        hydrogen_supply=resources("gb-model/fes_hydrogen_supply.csv"),
    log:
        logs("create_hydrogen_supply_table.log"),
    script:
        "../scripts/gb_model/create_hydrogen_supply_table.py"


rule create_off_grid_electrolysis_demand:
    message:
        "Process electricity demand of off-grid electrolysis from FES workbook into CSV format"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        fes_supply_sheets=config["fes"]["hydrogen"]["supply"]["supply_sheets"],
    input:
        supply_sheets=lambda wildcards: [
            resources(f"gb-model/fes/{year}/{sheet}.csv")
            for year, sheets in config["fes"]["hydrogen"]["supply"][
                "supply_sheets"
            ].items()
            for sheet in sheets.values()
        ],
        grid_electrolysis_capacities=resources(
            "gb-model/regional_grid_electrolysis_capacities.csv"
        ),
    output:
        electricity_demand=resources(
            "gb-model/regional_off_grid_electrolysis_electricity_demand.csv"
        ),
    log:
        logs("create_off_grid_electrolysis_demand.log"),
    script:
        "../scripts/gb_model/create_off_grid_electrolysis_demand.py"


rule create_hydrogen_storage_table:
    message:
        "Process hydrogen storage data from FES workbook into CSV format"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        fes_storage_sheets=config["fes"]["hydrogen"]["storage"]["storage_sheets"],
        interpolation_method=config["fes"]["hydrogen"]["storage"][
            "interpolation_method"
        ],
    input:
        storage_sheet=lambda wildcards: [
            resources(f"gb-model/fes/{year}/{sheet}.csv")
            for year, sheets in config["fes"]["hydrogen"]["storage"][
                "storage_sheets"
            ].items()
            for sheet in sheets.values()
        ],
    output:
        csv=resources("gb-model/H2_storage_capacity.csv"),
    log:
        logs("create_hydrogen_storage_table.log"),
    script:
        "../scripts/gb_model/create_hydrogen_storage_table.py"


rule create_demand_table:
    message:
        "Process {wildcards.demand_type} demand from FES workbook into CSV format"
    params:
        technology_detail=config["fes"]["gb"]["demand"]["Technology Detail"],
    input:
        regional_gb_data=resources("gb-model/regional_gb_data.csv"),
    output:
        demand=resources("gb-model/regional_{demand_type}_demand_annual.csv"),
    wildcard_constraints:
        demand_type="|".join(config["fes"]["gb"]["demand"]["Technology Detail"]),
    log:
        logs("create_{demand_type}_demand_table.log"),
    script:
        "../scripts/gb_model/create_demand_table.py"


rule create_flexibility_table:
    message:
        "Process {wildcards.flexibility_type} flexibility from FES workbook into CSV format"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        carrier_mapping=lambda wildcards: config["fes"]["gb"]["flexibility"][
            "carrier_mapping"
        ][wildcards.flexibility_type],
    input:
        flexibility_sheet=resources("gb-model/fes/2021/FLX1.csv"),
    output:
        flexibility=resources("gb-model/{flexibility_type}_flexibility.csv"),
    log:
        logs("create_{flexibility_type}_flexibility_table.log"),
    wildcard_constraints:
        flexibility_type="|".join(config["fes"]["gb"]["flexibility"]["carrier_mapping"]),
    script:
        "../scripts/gb_model/create_flexibility_table.py"


rule create_heat_flexibility_table:
    message:
        "Process residential heat demand flexibility from FES workbook"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        flex_level=config["fes"]["gb"]["flexibility"]["residential_heat_flex_level"],
    input:
        flexibility_sheet=resources("gb-model/fes/2021/FL.10.csv"),
    output:
        csv=resources("gb-model/residential_heat_dsr_flexibility.csv"),
    log:
        logs("create_heat_flexibility_table.log"),
    script:
        "../scripts/gb_model/create_heat_flexibility_table.py"


rule process_regional_flexibility_table:
    message:
        "Process regional {wildcards.flexibility_type} flexibility from FES workbook into CSV format"
    params:
        regional_distribution_reference=config["fes"]["gb"]["flexibility"][
            "regional_distribution_reference"
        ],
    input:
        flexibility=resources("gb-model/{flexibility_type}_flexibility.csv"),
        regional_gb_data=resources("gb-model/regional_gb_data.csv"),
    output:
        regional_flexibility=resources("gb-model/regional_{flexibility_type}.csv"),
    log:
        logs("process_regional_{flexibility_type}_flexibility_table.log"),
    wildcard_constraints:
        flexibility_type="|".join(
            config["fes"]["gb"]["flexibility"]["regional_distribution_reference"]
        ),
    script:
        "../scripts/gb_model/process_regional_flexibility_table.py"


rule cluster_baseline_electricity_demand_timeseries:
    message:
        "Cluster default PyPSA-Eur baseline electricity demand timeseries by bus"
    params:
        scaling_factor=config_provider("load", "scaling_factor"),
    input:
        load=resources("electricity_demand_base_s.nc"),
        busmap=resources("busmap_base_s_clustered.csv"),
    output:
        csv_file=resources("baseline_electricity_demand_s_clustered.csv"),
    log:
        logs("cluster_baseline_electricity_demand_timeseries.log"),
    script:
        "../scripts/gb_model/cluster_baseline_electricity_demand_timeseries.py"


rule process_cop_profiles:
    message:
        "Process COP profile for {wildcards.year} obtained from existing PyPSA-Eur rules"
    params:
        year=lambda wildcards: wildcards.year,
        heat_pump_sources=config["sector"]["heat_pump_sources"],
    input:
        cop_profile=resources("cop_profiles_base_s_clustered_{year}.nc"),
        clustered_pop_layout=resources("pop_layout_base_s_clustered.csv"),
        district_heat_share=resources("district_heat_share.csv"),
    output:
        csv=resources("cop_base_s_clustered_{year}.csv"),
    log:
        logs("process_cop_profiles_clustered_{year}.log"),
    script:
        "../scripts/gb_model/process_cop_profiles.py"


rule process_fes_heating_mix:
    message:
        "Process the share of electrified heating technologies from FES workbook"
    params:
        year=lambda wildcards: wildcards.year,
        electrified_heating_technologies=config["fes"]["gb"]["demand"]["heat"][
            "electrified_heating_technologies"
        ],
        scenario=config["fes"]["gb"]["scenario"],
    input:
        fes_residential_heatmix=resources("gb-model/fes/2021/CV.16.csv"),
        fes_commercial_heatmix=resources("gb-model/fes/2021/CV.55.csv"),
    output:
        csv=resources("gb-model/fes_heating_mix/{year}.csv"),
    log:
        logs("process_fes_heating_mix_{year}.log"),
    script:
        "../scripts/gb_model/process_fes_heating_mix.py"


rule process_heat_demand_shape:
    message:
        "Cluster default PyPSA-Eur heat demand shape by bus"
    params:
        year=lambda wildcards: wildcards.year,
    input:
        demand=resources("hourly_heat_demand_total_base_s_clustered.nc"),
        cop_profile=resources("cop_base_s_clustered_{year}.csv"),
        heating_mix=resources("gb-model/fes_heating_mix/{year}.csv"),
    output:
        residential_csv_file=resources(
            "gb-model/residential_heat_demand_shape/{year}.csv"
        ),
        #Industry load is not generated in PyPSA-Eur, hence the same profile as services is considered to be applicable for c&i
        commercial_csv_file=resources("gb-model/iandc_heat_demand_shape/{year}.csv"),
    log:
        logs("heat_demand_s_clustered_{year}.log"),
    script:
        "../scripts/gb_model/process_heat_demand_shape.py"


rule process_demand_shape:
    message:
        "Process {wildcards.demand_sector} demand profile shape into CSV format"
    input:
        pypsa_eur_demand_timeseries=resources("{demand_sector}_demand_s_clustered.csv"),
    output:
        demand_shape=resources("gb-model/{demand_sector}_demand_shape.csv"),
    log:
        logs("process_demand_shape_{demand_sector}.log"),
    script:
        "../scripts/gb_model/process_demand_shape.py"


rule process_ev_demand_shape:
    message:
        "Process EV demand profile shape into CSV format"
    params:
        snapshots=config_provider("snapshots"),
        drop_leap_day=config_provider("enable", "drop_leap_day"),
        plug_in_offset=config["ev"]["plug_in_offset"],
        charging_duration=config["ev"]["charging_duration"],
    input:
        clustered_pop_layout=resources("pop_layout_base_s_clustered.csv"),
        traffic_data_KFZ="data/bundle/emobility/KFZ__count",
    output:
        demand_shape=resources("gb-model/ev_demand_shape.csv"),
    log:
        logs("process_ev_demand_shape.log"),
    script:
        "../scripts/gb_model/process_ev_demand_shape.py"


rule create_ev_v2g_storage_table:
    message:
        "Process EV V2G storage data from FES workbook into CSV format"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        carrier_mapping=config["fes"]["gb"]["flexibility"]["carrier_mapping"]["ev_v2g"],
    input:
        storage_sheet=resources("gb-model/fes/2021/FL.14.csv"),
        flexibility_sheet=resources("gb-model/fes/2021/FLX1.csv"),
    output:
        storage_table=resources("gb-model/ev_v2g_storage.csv"),
    log:
        logs("create_ev_v2g_storage_table.log"),
    script:
        "../scripts/gb_model/create_ev_v2g_storage_table.py"


rule create_ev_peak_charging_table:
    message:
        "Process EV unmanaged charging demand from FES workbook into CSV format"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
    input:
        unmanaged_charging_sheet=resources("gb-model/fes/2021/FL.11.csv"),
    output:
        csv=resources("gb-model/ev_peak.csv"),
    log:
        logs("create_ev_peak_charging_table.log"),
    script:
        "../scripts/gb_model/create_ev_peak_charging_table.py"


rule process_regional_ev_data:
    message:
        "Process regional EV {wildcards.ev_data_type} data into CSV format"
    input:
        input_csv=resources("gb-model/ev_{ev_data_type}.csv"),
        reference_data=lambda wildcards: {
            "peak": resources("gb-model/regional_ev_demand_annual.csv"),
            "v2g_storage": resources("gb-model/regional_ev_v2g.csv"),
        }[wildcards.ev_data_type],
    output:
        regional_output=resources("gb-model/regional_ev_{ev_data_type}.csv"),
    log:
        logs("process_regional_ev_{ev_data_type}.log"),
    wildcard_constraints:
        ev_data_type="v2g_storage|peak",
    script:
        "../scripts/gb_model/process_regional_ev_data.py"


rule process_regional_battery_storage_capacity:
    message:
        "Process national storage data from FES workbook into CSV format"
    params:
        scenario=config["fes"]["gb"]["scenario"],
        year_range=config["fes"]["year_range_incl"],
        carrier_mapping=config["fes"]["gb"]["flexibility"]["carrier_mapping"]["battery"],
    input:
        flexibility_sheet=resources("gb-model/fes/2021/FLX1.csv"),
        regional_data=resources("gb-model/fes_powerplants.csv"),
    output:
        csv=resources("gb-model/regional_battery_storage_capacity_inc_eur.csv"),
    log:
        logs("process_regional_battery_storage_capacity.log"),
    script:
        "../scripts/gb_model/process_regional_battery_storage_capacity.py"


rule process_H2_demand:
    message:
        "Get net H2 demand"
    input:
        demand=resources("gb-model/fes_hydrogen_demand.csv"),
        fixed_supply=resources("gb-model/fes_hydrogen_supply.csv"),
    output:
        csv=resources("gb-model/H2_demand_annual.csv"),
    log:
        logs("process_H2_demand.log"),
    script:
        "../scripts/gb_model/process_H2_demand.py"


rule process_regional_H2_data:
    message:
        "Process {wildcards.data_type} H2 data into GB regions format"
    input:
        to_regionalise=resources("gb-model/{data_type}.csv"),
        regional_distribution=resources(
            "gb-model/regional_grid_electrolysis_capacities.csv"
        ),
    params:
        param_name=lambda wildcards: {
            "H2_demand_annual": "p_set",
            "H2_storage_capacity": "e_nom",
        }[wildcards.data_type],
    output:
        csv=resources("gb-model/regional_{data_type}.csv"),
    wildcard_constraints:
        data_type="H2_demand_annual|H2_storage_capacity",
    log:
        logs("process_regional_H2_data_{data_type}.log"),
    script:
        "../scripts/gb_model/process_regional_H2_data.py"


rule distribute_eur_demands:
    message:
        "Distribute total European neighbour annual demands into base electricity, heating, and transport"
    input:
        eur_data=resources("gb-model/national_eur_data.csv"),
        energy_totals=resources("energy_totals.csv"),
        electricity_demands=expand(
            resources("gb-model/regional_{demand_type}_demand_annual.csv"),
            demand_type=config["fes"]["gb"]["demand"]["Technology Detail"].keys(),
        ),
        extra_demands=[],
    params:
        totals_to_demands=config["fes"]["eur"]["totals_to_demand_groups"],
        base_year=config["energy"]["energy_totals_year"],
    output:
        csv=resources("gb-model/eur_demand_annual.csv"),
    log:
        logs("distribute_eur_demands.log"),
    script:
        "../scripts/gb_model/distribute_eur_demands.py"


def _ref_demand_type(w):
    return config["fes"]["eur"]["add_data_reference"][w.dataset]


rule add_eur_H2_demand:
    message:
        "Add European H2 demand based on historical data combined with TYNDP future scenario demands"
    input:
        gb_demand=resources("gb-model/regional_H2_demand_annual.csv"),
        eur_demand_tyndp="data/gb-model/tyndp_h2_demand.csv",
        eur_demand_today="data/gb-model/downloaded/eur_H2_demand_today.xlsx",
    params:
        countries=config["countries"],
    output:
        csv=resources("gb-model/regional_H2_demand_annual_inc_eur.csv"),
    log:
        logs("add_eur_H2_demand.log"),
    script:
        "../scripts/gb_model/add_eur_H2_demand.py"


rule synthesise_eur_H2_data:
    message:
        "Synthesise European H2 {wildcards.dataset} data using GB data"
    input:
        h2_demand=resources("gb-model/regional_H2_demand_annual_inc_eur.csv"),
        gb_only_dataset=resources("gb-model/regional_{dataset}.csv"),
    output:
        csv=resources("gb-model/regional_{dataset}_inc_eur.csv"),
    log:
        logs("synthesise_eur_H2_data_{dataset}.log"),
    wildcard_constraints:
        dataset="off_grid_electrolysis_electricity_demand|H2_storage_capacity|grid_electrolysis_capacities",
    script:
        "../scripts/gb_model/synthesise_eur_H2_data.py"


rule synthesise_eur_data:
    message:
        "Create a regional {wildcards.dataset} dataset including European neighbours based on GB data and relative annual demand"
    input:
        gb_demand_annual=lambda wildcards: resources(
            f"gb-model/regional_{_ref_demand_type(wildcards)}_demand_annual.csv"
        ),
        eur_demand_annual=resources("gb-model/eur_demand_annual.csv"),
        gb_only_dataset=resources("gb-model/regional_{dataset}.csv"),
    params:
        demand_type=_ref_demand_type,
    output:
        csv=resources("gb-model/regional_{dataset}_inc_eur.csv"),
    log:
        logs("synthesise_eur_data_{dataset}.log"),
    wildcard_constraints:
        dataset="|".join(config["fes"]["eur"]["add_data_reference"].keys()),
    script:
        "../scripts/gb_model/synthesise_eur_data.py"


rule scaled_demand_profile:
    message:
        "Generate {wildcards.demand_type} demand profile for model year {wildcards.year}"
    input:
        gb_demand_annual=resources("gb-model/regional_{demand_type}_demand_annual.csv"),
        eur_demand_annual=resources("gb-model/eur_demand_annual.csv"),
        demand_shape=resources("gb-model/{demand_type}_demand_shape.csv"),
    output:
        csv=resources("gb-model/{demand_type}_demand/{year}.csv"),
    log:
        logs("scaled_demand_profile_{demand_type}_{year}.log"),
    wildcard_constraints:
        demand_type="baseline_electricity",
    script:
        "../scripts/gb_model/scaled_demand_profile.py"


use rule scaled_demand_profile as scaled_heat_demand_profile with:
    input:
        gb_demand_annual=resources("gb-model/regional_{demand_type}_demand_annual.csv"),
        eur_demand_annual=resources("gb-model/eur_demand_annual.csv"),
        demand_shape=resources("gb-model/{demand_type}_demand_shape/{year}.csv"),
    wildcard_constraints:
        demand_type="residential_heat|iandc_heat",


use rule scaled_demand_profile as scaled_ev_demand_profile with:
    input:
        gb_demand_annual=resources("gb-model/regional_{demand_type}_demand_annual.csv"),
        eur_demand_annual=resources("gb-model/eur_demand_annual.csv"),
        demand_shape=resources("gb-model/{demand_type}_demand_shape.csv"),
        gb_demand_peak=resources("gb-model/regional_{demand_type}_peak.csv"),
    params:
        scaling_params=config["ev"]["ev_demand_profile_transformation"],
    wildcard_constraints:
        demand_type="ev",


rule create_chp_p_min_pu_profile:
    message:
        "Create CHP minimum operation profiles linked to heat demand"
    params:
        heat_to_power_ratio=config["chp"]["heat_to_power_ratio"],
        min_operation_level=config["chp"]["min_operation_level"],
        shutdown_threshold=config["chp"]["shutdown_threshold"],
    input:
        regions=resources("gb-model/merged_shapes.geojson"),
        heat_demand=resources("hourly_heat_demand_total_base_s_clustered.nc"),
    output:
        chp_p_min_pu=resources("gb-model/chp_p_min_pu.csv"),
    log:
        logs("create_chp_p_min_pu_profile.log"),
    script:
        "../scripts/gb_model/create_chp_p_min_pu_profile.py"


rule process_CfD_strike_prices:
    message:
        "get strike price for low carbon contracts"
    params:
        carrier_mapping=config["low_carbon_register"]["carrier_mapping"],
        end_year=config["fes"]["year_range_incl"][0],
    input:
        register="data/gb-model/downloaded/low-carbon-contracts.csv",
    output:
        csv=resources("gb-model/CfD_strike_prices.csv"),
    log:
        logs("process_CfD_strike_prices.log"),
    script:
        "../scripts/gb_model/process_CfD_strike_prices.py"


rule assign_costs:
    message:
        "Prepares costs file from technology-data of PyPSA-Eur and FES and assigns to {wildcards.data_file}"
    params:
        costs_config=config["costs"],
        fes_scenario=config["fes"]["gb"]["scenario"],
    input:
        tech_costs=lambda w: resources(
            f"costs_{config_provider('costs', 'year')(w)}.csv"
        ),
        fes_power_costs=resources("gb-model/fes-costing/AS.1 (Power Gen).csv"),
        fes_carbon_costs=resources("gb-model/fes-costing/AS.7 (Carbon Cost).csv"),
        fes_powerplants=resources("gb-model/{data_file}.csv"),
    output:
        enriched_powerplants=resources("gb-model/{data_file}_inc_tech_data.csv"),
    log:
        logs("assign_costs_{data_file}.log"),
    wildcard_constraints:
        data_file="fes_powerplants|regional_H2_storage_capacity_inc_eur|regional_grid_electrolysis_capacities_inc_eur",
    script:
        "../scripts/gb_model/assign_costs.py"


rule compose_network:
    params:
        countries=config["countries"],
        costs_config=config["costs"],
        electricity=config["electricity"],
        clustering=config["clustering"],
        renewable=config["renewable"],
        enable_chp=config["chp"]["enable"],
        dsr_hours_dict=config["fes"]["gb"]["flexibility"]["dsr_hours"],
        load_bus_suffixes=config["fes"]["gb"]["demand"]["bus_suffix"],
        flex_carrier_suffixes=config["fes"]["gb"]["flexibility"]["carrier_suffix"],
    input:
        unpack(input_profile_tech),
        demands=expand(
            resources("gb-model/{demand_type}_demand/{{year}}.csv"),
            demand_type=config["fes"]["gb"]["demand"]["Technology Detail"].keys(),
        ),
        dsr=expand(
            resources("gb-model/regional_{sector}_dsr_inc_eur.csv"),
            sector=["residential", "iandc", "iandc_heat", "ev", "residential_heat"],
        ),
        ev_data=expand(
            resources("gb-model/regional_ev_{ev_data}_inc_eur.csv"),
            ev_data=["v2g_storage", "v2g"],
        )
        + [resources("avail_profile_s_clustered.csv")],
        network=resources("networks/base_s_clustered.nc"),
        powerplants=resources("gb-model/fes_powerplants_inc_tech_data.csv"),
        tech_costs=lambda w: resources(
            f"costs_{config_provider('costs', 'year')(w)}.csv"
        ),
        hydro_capacities=ancient("data/hydro_capacities.csv"),
        chp_p_min_pu=resources("gb-model/chp_p_min_pu.csv"),
        interconnectors_p_nom=resources("gb-model/interconnectors_p_nom.csv"),
        interconnectors_availability=resources(
            "gb-model/inter_gb_transmission_availability.csv"
        ),
        generator_availability=resources(
            "gb-model/GB_generator_monthly_availability_fraction.csv"
        ),
        battery_e_nom=resources(
            "gb-model/regional_battery_storage_capacity_inc_eur.csv"
        ),
        H2_data=[
            resources("gb-model/regional_H2_demand_annual_inc_eur.csv"),
            resources(
                "gb-model/regional_off_grid_electrolysis_electricity_demand_inc_eur.csv"
            ),
            resources(
                "gb-model/regional_H2_storage_capacity_inc_eur_inc_tech_data.csv"
            ),
            resources(
                "gb-model/regional_grid_electrolysis_capacities_inc_eur_inc_tech_data.csv"
            ),
        ],
        intermediate_data=[
            # TODO: calculate intra_gb availability per line/boundary before this point (currently only per TO)
            resources("gb-model/intra_gb_transmission_availability.csv"),
        ],
    output:
        network=resources("networks/composed_{clusters}/{year}.nc"),
    log:
        logs("compose_network_{clusters}_{year}.log"),
    resources:
        mem_mb=4000,
    wildcard_constraints:
        # We only accept clustered clusters
        clusters="clustered",
    script:
        "../scripts/gb_model/compose_network.py"


year_range = config["fes"]["year_range_incl"]


rule compose_networks:
    input:
        expand(
            resources("networks/composed_clustered/{year}.nc"),
            **config["scenario"],
            run=config["run"]["name"],
            year=list(np.arange(year_range[0], year_range[1])),
        ),


rule prepare_unconstrained:
    message:
        "Prepare network for unconstrained optimization"
    input:
        network=resources("networks/composed_clustered/{year}.nc"),
    output:
        network=resources("networks/unconstrained_clustered/{year}.nc"),
    log:
        logs("prepare_unconstrained_network/{year}.log"),
    script:
        "../scripts/gb_model/prepare_unconstrained_network.py"


rule solve_unconstrained:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=[],
    input:
        network=resources("networks/unconstrained_clustered/{year}.nc"),
    output:
        network=RESULTS + "networks/unconstrained_clustered/{year}.nc",
        config=RESULTS + "configs/config.unconstrained_clustered/{year}.yaml",
    log:
        solver=normpath(
            RESULTS + "logs/solve_network/unconstrained_clustered/{year}_solver.log"
        ),
        memory=RESULTS + "logs/solve_network/unconstrained_clustered/{year}_memory.log",
        python=RESULTS + "logs/solve_network/unconstrained_clustered/{year}_python.log",
    benchmark:
        (RESULTS + "benchmarks/solve_network/unconstrained_clustered/{year}")
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    script:
        "../scripts/solve_network.py"


rule solve_constrained:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=input_custom_extra_functionality,
        etys_boundaries_to_lines=config["region_operations"]["etys_boundaries_lines"],
        etys_boundaries_to_links=config["region_operations"]["etys_boundaries_links"],
    input:
        network=resources("networks/constrained_clustered_{year}.nc"),
        etys_caps=resources("gb-model/etys_boundary_capabilities.csv"),
    output:
        network=RESULTS + "networks/constrained_clustered_{year}.nc",
        config=RESULTS + "configs/config.constrained_clustered_{year}.yaml",
    log:
        solver=normpath(
            RESULTS + "logs/solve_network/constrained_clustered_{year}_solver.log"
        ),
        memory=RESULTS + "logs/solve_network/constrained_clustered_{year}_memory.log",
        python=RESULTS + "logs/solve_network/constrained_clustered_{year}_python.log",
    benchmark:
        (RESULTS + "benchmarks/solve_network/constrained_clustered_{year}")
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    script:
        "../scripts/solve_network.py"


rule prepare_constrained_network:
    message:
        "Prepare network for constrained optimization"
    params:
        bids_and_offers=config_provider("redispatch"),
    input:
        network=resources("networks/composed_clustered/{year}.nc"),
        unconstrained_result=RESULTS + "networks/unconstrained_clustered/{year}.nc",
        renewable_payment_profile=resources(
            "gb-model/renewable_payment_profile/{year}.csv"
        ),
    output:
        network=resources("networks/constrained_clustered/{year}.nc"),
    log:
        logs("prepare_constrained_network/{year}.log"),
    script:
        "../scripts/gb_model/prepare_constrained_network.py"


rule get_renewable_payment_profile:
    message:
        "Compute the difference in market rate and strike prices as a profile for renewable generators"
    input:
        unconstrained_result=RESULTS + "networks/unconstrained_clustered/{year}.nc",
        strike_prices=resources("gb-model/CfD_strike_prices.csv"),
    output:
        csv=resources("gb-model/renewable_payment_profile/{year}.csv"),
    log:
        logs("get_renewable_payment_profile/{year}.log"),
    script:
        "../scripts/gb_model/get_renewable_payment_profile.py"

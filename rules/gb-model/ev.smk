# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Electric vehicle demand and V1G / V2G rules.
"""


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
        traffic_data_KFZ=Path(MOBILITY_PROFILES_DATASET["folder"]) / "kfz.csv",
    output:
        demand_shape=resources("gb-model/ev_demand_shape.csv"),
    log:
        logs("process_ev_demand_shape.log"),
    script:
        scripts("gb_model/ev/process_ev_demand_shape.py")


rule create_ev_peak_charging_table:
    message:
        "Process EV unmanaged charging demand from FES workbook into CSV format"
    params:
        year_range=config["redispatch"]["year_range_incl"],
    input:
        unmanaged_charging_sheet=resources("gb-model/fes/ED5.csv"),
    output:
        csv=resources("gb-model/{fes_scenario}/ev_peak_demand.csv"),
    log:
        logs("create_ev_peak_charging_table_{fes_scenario}.log"),
    script:
        scripts("gb_model/ev/create_ev_peak_charging_table.py")


rule create_ev_v2g_storage_table:
    message:
        "Process EV unmanaged charging demand from FES workbook into CSV format"
    params:
        v2g_multiplier=config["fes"]["gb"]["flexibility"][
            "v2g_storage_to_capacity_ratio"
        ],
    input:
        v2g_cap=resources("gb-model/{fes_scenario}/regional_ev_v2g.csv"),
    output:
        csv=resources("gb-model/{fes_scenario}/regional_ev_v2g_storage.csv"),
    log:
        logs("create_ev_v2g_storage_table_{fes_scenario}.log"),
    script:
        scripts("gb_model/ev/create_ev_v2g_storage_table.py")


rule scaled_ev_demand_profile:
    message:
        "Generate EV demand profile for model year {wildcards.year}"
    input:
        gb_demand_annual=resources(
            "gb-model/{fes_scenario}/regional_ev_demand_annual.csv"
        ),
        eur_demand_annual=resources("gb-model/{fes_scenario}/eur_demand_annual.csv"),
        demand_shape=resources("gb-model/ev_demand_shape.csv"),
        gb_demand_peak=resources("gb-model/{fes_scenario}/regional_ev_peak_demand.csv"),
    params:
        scaling_params=config["ev"]["ev_demand_profile_transformation"],
    output:
        csv=resources("gb-model/{fes_scenario}/ev_demand/{year}.csv"),
    log:
        logs("scaled_demand_profile_{fes_scenario}_ev_{year}.log"),
    script:
        scripts("gb_model/ev/scaled_demand_profile.py")

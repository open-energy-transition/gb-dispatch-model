# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Generator component rules.
"""


rule retrieve_entsoe_unavailability_data:
    output:
        unavailability=resources("gb-model/{zone}_generator_unavailability.csv"),
    log:
        logs("process_entsoe_unavailability_data_{zone}.log"),
    params:
        start_date=config["entsoe_unavailability"]["start_date"],
        end_date=config["entsoe_unavailability"]["end_date"],
        max_request_days=config["entsoe_unavailability"]["max_request_days"],
    resources:
        mem_mb=1000,
    script:
        scripts("gb_model/generators/retrieve_entsoe_unavailability_data.py")


rule generator_monthly_availability_fraction:
    message:
        "Combine outage data with DUKES current technology capacities to get monthly outage fractions per carrier"
    input:
        outages=resources("gb-model/{zone}_generator_unavailability.csv"),
        dukes_data=resources("gb-model/dukes-current-capacity.csv"),
    params:
        entsoe_carrier_mapping=config["entsoe_unavailability"]["carrier_mapping"],
        start_date=config["entsoe_unavailability"]["start_date"],
        end_date=config["entsoe_unavailability"]["end_date"],
        max_unavailable_days=config["entsoe_unavailability"]["max_unavailable_days"],
        dukes_config=config["dukes-5.11"],
        default_set=config["fes"]["default_set"],
    output:
        csv=resources("gb-model/{zone}_generator_monthly_availability_fraction.csv"),
    log:
        logs("{zone}_generator_monthly_availability_fraction.log"),
    script:
        scripts("gb_model/generators/generator_monthly_availability_fraction.py")


rule create_powerplants_table:
    message:
        "Tabulate powerplant data GSP-wise from FES workbook sheet BB1 and EU supply data"
    params:
        gb_config=config["fes"]["gb"],
        eur_config=config["fes"]["eur"],
        dukes_config=config["dukes-5.11"],
        default_set=config["fes"]["default_set"],
    input:
        gsp_data=resources("gb-model/{fes_scenario}/regional_gb_data.csv"),
        eur_data=resources("gb-model/{fes_scenario}/national_eur_data.csv"),
        dukes_data=resources("gb-model/dukes-current-capacity.csv"),
    output:
        csv=resources("gb-model/{fes_scenario}/fes_powerplants.csv"),
    log:
        logs("create_powerplants_table_{fes_scenario}.log"),
    script:
        scripts("gb_model/generators/create_powerplants_table.py")


rule assign_costs:
    message:
        "Prepares costs file from technology-data of PyPSA-Eur and FES and assigns to {wildcards.data_file}"
    params:
        costs_config=config["fes_costs"],
    input:
        tech_costs=Path(COSTS_DATASET["folder"])
        / f"costs_{config['scenario']['planning_horizons'][0]}.csv",
        fes_power_costs=resources("gb-model/fes-costing/AS.1 (Power Gen).csv"),
        fes_carbon_costs=resources("gb-model/fes-costing/AS.7 (Carbon Cost).csv"),
        fes_powerplants=resources("gb-model/{fes_scenario}/{data_file}.csv"),
    output:
        enriched_powerplants=resources(
            "gb-model/{fes_scenario}/{data_file}_inc_tech_data.csv"
        ),
    log:
        logs("assign_costs_{fes_scenario}_{data_file}.log"),
    wildcard_constraints:
        data_file="fes_powerplants|regional_H2_storage_capacity_inc_eur|regional_grid_electrolysis_capacities_inc_eur",
    script:
        scripts("gb_model/generators/assign_costs.py")


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
        scripts("gb_model/generators/create_chp_p_min_pu_profile.py")

# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Hydrogen subsystem (demand, storage, electrolysis, fuel cells / turbine) rules.
"""


rule create_hydrogen_data_tables:
    message:
        "Process net hydrogen demand data, off-grid electrolysis electricity demand, and storage demand from FES workbook into CSV format"
    params:
        year_range=config["redispatch"]["year_range_incl"],
        data_selection=config["fes"]["hydrogen"]["data_selection"],
    input:
        whole_system_data=resources("gb-model/fes/WS1.csv"),
    output:
        hydrogen_demand=resources("gb-model/{fes_scenario}/H2_demand_annual.csv"),
        electricity_demand=resources(
            "gb-model/{fes_scenario}/non_networked_electrolysis_demand_annual.csv"
        ),
        storage=resources("gb-model/{fes_scenario}/H2_storage_capacity.csv"),
        electrolysis_efficiency=resources(
            "gb-model/{fes_scenario}/electrolysis_efficiency.csv"
        ),
    log:
        logs("create_hydrogen_data_tables_{fes_scenario}.log"),
    script:
        scripts("gb_model/hydrogen/create_hydrogen_data_tables.py")


rule create_grid_electrolysis_table:
    message:
        "Process hydrogen electrolysis data from FES workbook into CSV format"
    input:
        regional_gb_data=resources("gb-model/{fes_scenario}/regional_gb_data.csv"),
    output:
        csv=resources(
            "gb-model/{fes_scenario}/regional_grid_electrolysis_capacities.csv"
        ),
    log:
        logs("create_grid_electrolysis_table_{fes_scenario}.log"),
    script:
        scripts("gb_model/hydrogen/create_grid_electrolysis_table.py")


rule add_eur_H2_demand:
    message:
        "Add European H2 demand based on historical data combined with TYNDP future scenario demands"
    input:
        gb_demand=resources("gb-model/{fes_scenario}/regional_H2_demand_annual.csv"),
        eur_demand_tyndp="data/gb-model/tyndp_h2_demand.csv",
        eur_demand_today="data/gb-model/downloaded/eur_H2_demand_today.xlsx",
    params:
        countries=config["countries"],
    output:
        csv=resources("gb-model/{fes_scenario}/regional_H2_demand_annual_inc_eur.csv"),
    log:
        logs("add_eur_H2_demand_{fes_scenario}.log"),
    script:
        scripts("gb_model/hydrogen/add_eur_H2_demand.py")


rule synthesise_eur_H2_data:
    message:
        "Synthesise European H2 {wildcards.dataset} data using GB data"
    input:
        h2_demand=resources(
            "gb-model/{fes_scenario}/regional_H2_demand_annual_inc_eur.csv"
        ),
        gb_only_dataset=resources("gb-model/{fes_scenario}/regional_{dataset}.csv"),
    output:
        csv=resources("gb-model/{fes_scenario}/regional_{dataset}_inc_eur.csv"),
    log:
        logs("synthesise_eur_H2_data_{fes_scenario}_{dataset}.log"),
    wildcard_constraints:
        dataset="non_networked_electrolysis_demand_annual|H2_storage_capacity|grid_electrolysis_capacities",
    script:
        scripts("gb_model/hydrogen/synthesise_eur_H2_data.py")

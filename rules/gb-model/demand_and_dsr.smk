# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Baseline and generalised demand and demand-side response (flexibility) rules.
"""


rule create_demand_table:
    input:
        regional_gb_data=resources("gb-model/{fes_scenario}/regional_gb_data.csv"),
    output:
        demand=resources(
            "gb-model/{fes_scenario}/regional_{demand_type}_demand_annual.csv"
        ),
    log:
        logs("create_{demand_type}_demand_table_{fes_scenario}.log"),
    wildcard_constraints:
        demand_type="|".join(config["fes"]["gb"]["demand"]["Technology Detail"]),
    params:
        technology_detail=config["fes"]["gb"]["demand"]["Technology Detail"],
    message:
        "Process {wildcards.demand_type} demand from FES workbook into CSV format"
    script:
        scripts("gb_model/demand_and_dsr/create_demand_table.py")


rule cluster_baseline_electricity_demand_timeseries:
    input:
        load=resources("electricity_demand_base_s.nc"),
        busmap=resources("busmap_base_s_clustered.csv"),
    output:
        csv_file=resources(
            "gb-model/historical_baseline_electricity_demand_profile.csv"
        ),
    log:
        logs("cluster_baseline_electricity_demand_timeseries.log"),
    params:
        scaling_factor=config_provider("load", "scaling_factor"),
    message:
        "Cluster default PyPSA-Eur baseline electricity demand timeseries by bus"
    script:
        scripts(
            "gb_model/demand_and_dsr/cluster_baseline_electricity_demand_timeseries.py"
        )


rule process_baseline_demand_shape:
    input:
        historical_profile=resources(
            "gb-model/historical_baseline_electricity_demand_profile.csv"
        ),
        historical_resistive_heater_demand=resources(
            "gb-model/{fes_scenario}/historical_resistive_heater_demand/{year}.csv"
        ),
    output:
        demand_shape=resources(
            "gb-model/{fes_scenario}/baseline_electricity_demand_shape/{year}.csv"
        ),
    log:
        logs("process_baseline_demand_shape_{fes_scenario}_{year}.log"),
    message:
        "Process baseline electricity demand profile shape into CSV format for {wildcards.year}"
    script:
        scripts("gb_model/demand_and_dsr/process_baseline_demand_shape.py")


rule create_flexibility_table:
    input:
        flexibility_sheet=resources(f"gb-model/fes/FLX1.csv"),
    output:
        flexibility=resources("gb-model/{fes_scenario}/{flexibility_type}.csv"),
    log:
        logs("create_flexibility_table_{fes_scenario}_{flexibility_type}.log"),
    wildcard_constraints:
        flexibility_type="|".join(config["fes"]["gb"]["flexibility"]["carrier_mapping"]),
    params:
        year_range=config["redispatch"]["year_range_incl"],
        carrier_mapping=lambda wildcards: config["fes"]["gb"]["flexibility"][
            "carrier_mapping"
        ][wildcards.flexibility_type],
    message:
        "Process {wildcards.flexibility_type} flexibility from FES workbook into CSV format"
    script:
        scripts("gb_model/demand_and_dsr/create_flexibility_table.py")


rule synthesise_gb_regional_data:
    input:
        national_gb_data=resources("gb-model/{fes_scenario}/{regional_reference}.csv"),
        regional_gb_data=resources("gb-model/{fes_scenario}/regional_gb_data.csv"),
    output:
        csv=resources("gb-model/{fes_scenario}/regional_{regional_reference}.csv"),
    log:
        logs("synthesise_gb_regional_data_{fes_scenario}_{regional_reference}.log"),
    wildcard_constraints:
        regional_reference="|".join(
            config["fes"]["gb"]["regional_distribution_reference"].keys()
        ),
    params:
        regional_distribution_reference=lambda wildcards: config["fes"]["gb"][
            "regional_distribution_reference"
        ][wildcards.regional_reference],
    message:
        "Process regional {wildcards.regional_reference} data from FES workbook into CSV format"
    script:
        scripts("gb_model/demand_and_dsr/synthesise_gb_regional_data.py")


rule distribute_eur_demands:
    input:
        eur_data=resources("gb-model/{fes_scenario}/national_eur_data.csv"),
        energy_totals=resources("energy_totals.csv"),
        electricity_demands=expand(
            resources(
                "gb-model/{{fes_scenario}}/regional_{demand_type}_demand_annual.csv"
            ),
            demand_type=config["fes"]["gb"]["demand"]["Technology Detail"].keys(),
        ),
        extra_demands=[],
    output:
        csv=resources("gb-model/{fes_scenario}/eur_demand_annual.csv"),
    log:
        logs("distribute_eur_demands_{fes_scenario}.log"),
    params:
        totals_to_demands=config["fes"]["eur"]["totals_to_demand_groups"],
        base_year=config["energy"]["energy_totals_year"],
    message:
        "Distribute total European neighbour annual demands into base electricity, heating, and transport"
    script:
        scripts("gb_model/demand_and_dsr/distribute_eur_demands.py")


def _ref_demand_type(w):
    return config["fes"]["eur"]["add_data_reference"][w.dataset]


rule synthesise_eur_data:
    input:
        gb_demand_annual=lambda wildcards: resources(
            f"gb-model/{{fes_scenario}}/regional_{_ref_demand_type(wildcards)}_demand_annual.csv"
        ),
        eur_demand_annual=resources("gb-model/{fes_scenario}/eur_demand_annual.csv"),
        gb_only_dataset=resources("gb-model/{fes_scenario}/regional_{dataset}.csv"),
    output:
        csv=resources("gb-model/{fes_scenario}/regional_{dataset}_inc_eur.csv"),
    log:
        logs("synthesise_eur_data_{fes_scenario}_{dataset}.log"),
    wildcard_constraints:
        dataset="|".join(config["fes"]["eur"]["add_data_reference"].keys()),
    params:
        demand_type=_ref_demand_type,
    message:
        "Create a regional {wildcards.dataset} dataset including European neighbours based on GB data and relative annual demand"
    script:
        scripts("gb_model/demand_and_dsr/synthesise_eur_data.py")


rule scaled_demand_profile:
    input:
        gb_demand_annual=resources(
            "gb-model/{fes_scenario}/regional_baseline_electricity_demand_annual.csv"
        ),
        resistive_heater_demand=resources(
            "gb-model/{fes_scenario}/future_resistive_heater_demand/{year}.csv"
        ),
        eur_demand_annual=resources("gb-model/{fes_scenario}/eur_demand_annual.csv"),
        demand_shape=resources(
            "gb-model/{fes_scenario}/baseline_electricity_demand_shape/{year}.csv"
        ),
    output:
        csv=resources("gb-model/{fes_scenario}/baseline_electricity_demand/{year}.csv"),
    log:
        logs("scaled_demand_profile_{fes_scenario}_baseline_electricity_{year}.log"),
    message:
        "Generate baseline_electricity demand profile for model year {wildcards.year}"
    script:
        scripts("gb_model/demand_and_dsr/scaled_demand_profile.py")


rule add_extra_demands:
    input:
        fes_total_electricity_demands=resources("gb-model/fes/ED1.csv"),
        electricity_demands=expand(
            resources("gb-model/{{fes_scenario}}/{demand_type}_demand/{{year}}.csv"),
            demand_type=[
                "baseline_electricity",
                "iandc_heat",
                "ev",
                "residential_heat",
            ],
        ),
        h2_demand=resources("gb-model/{fes_scenario}/regional_H2_demand_annual.csv"),
        non_networked_h2_demand=resources(
            "gb-model/{fes_scenario}/regional_non_networked_electrolysis_demand_annual.csv"
        ),
        electrolysis_efficiency=resources(
            "gb-model/{fes_scenario}/electrolysis_efficiency.csv"
        ),
    output:
        csv=resources("gb-model/{fes_scenario}/additional_demand/{year}.csv"),
    log:
        logs("add_extra_demands_{fes_scenario}_{year}.log"),
    params:
        total=config_provider("fes", "gb", "demand", "expected_total_data_item"),
        losses=config_provider("fes", "gb", "demand", "losses_data_item"),
    message:
        "Add leftover {wildcards.year} {wildcards.fes_scenario} electricity demand not already accounted for in other profiles (direct transmission demand; T&D losses)"
    script:
        scripts("gb_model/demand_and_dsr/add_extra_demands.py")

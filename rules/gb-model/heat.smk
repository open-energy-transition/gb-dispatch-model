# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Building heat demand and demand-side response rules.
"""


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
        csv=resources("gb-model/cop/{year}.csv"),
    log:
        logs("process_cop_profiles_clustered_{year}.log"),
    script:
        scripts("gb_model/heat/process_cop_profiles.py")


rule process_fes_heat_technologies:
    message:
        "Process the share of electrified heating technologies from FES workbook"
    params:
        year_range=config["redispatch"]["year_range_incl"],
        electrified_heating_technologies=config["fes"]["gb"]["demand"]["heat"][
            "electrified_heating_technologies"
        ],
    input:
        fes_heat_technology_data=resources(f"gb-model/fes/ED3.csv"),
    output:
        residential=resources(
            "gb-model/{fes_scenario}/residential_heat_demand_annual.csv"
        ),
        services=resources("gb-model/{fes_scenario}/iandc_heat_demand_annual.csv"),
    log:
        logs("process_fes_heat_technologies_{fes_scenario}.log"),
    script:
        scripts("gb_model/heat/process_fes_heat_technologies.py")


rule resistive_heater_demand_profile:
    message:
        "Process historical and future resistive heat demand profile for {wildcards.year}"
    input:
        energy_totals=resources("pop_weighted_energy_totals_s_clustered.csv"),
        heat_demand_shape=resources("hourly_heat_demand_total_base_s_clustered.nc"),
        residential_heat_demand_annual=resources(
            "gb-model/{fes_scenario}/regional_residential_heat_demand_annual_inc_eur.csv"
        ),
        services_heat_demand_annual=resources(
            "gb-model/{fes_scenario}/regional_iandc_heat_demand_annual_inc_eur.csv"
        ),
    output:
        historical=resources(
            "gb-model/{fes_scenario}/historical_resistive_heater_demand/{year}.csv"
        ),
        future=resources(
            "gb-model/{fes_scenario}/future_resistive_heater_demand/{year}.csv"
        ),
    log:
        logs("resistive_heater_demand_profile_{fes_scenario}_{year}.log"),
    script:
        scripts("gb_model/heat/resistive_heater_demand_profile.py")


rule heat_demand_electricity_load_profile:
    message:
        "Scale PyPSA-Eur {wildcards.sector} heat demand profiles to create heat bus electricity load profiles for future year {wildcards.year}"
    input:
        demand=resources("hourly_heat_demand_total_base_s_clustered.nc"),
        cop_profile=resources("gb-model/cop/{year}.csv"),
        heating_mix=resources(
            "gb-model/{fes_scenario}/regional_{sector}_heat_demand_annual_inc_eur.csv"
        ),
        energy_totals=resources("pop_weighted_energy_totals_s_clustered.csv"),
    output:
        csv=resources("gb-model/{fes_scenario}/{sector}_heat_demand/{year}.csv"),
    log:
        logs("heat_demand_s_clustered_{fes_scenario}_{sector}_{year}.log"),
    script:
        scripts("gb_model/heat/heat_demand_electricity_load_profile.py")

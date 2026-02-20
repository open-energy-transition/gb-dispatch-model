# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Main network composition rule for the GB model.
"""


include: "gb-model/preprocess.smk"
include: "gb-model/generators.smk"
include: "gb-model/transmission.smk"
include: "gb-model/storage.smk"
include: "gb-model/demand_and_dsr.smk"
include: "gb-model/ev.smk"
include: "gb-model/heat.smk"
include: "gb-model/hydrogen.smk"
include: "gb-model/dispatch.smk"
include: "gb-model/redispatch.smk"


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
        time_aggregation=config["time_aggregation"],
    input:
        unpack(input_profile_tech),
        demands=expand(
            resources("gb-model/{{fes_scenario}}/{demand_type}_demand/{{year}}.csv"),
            demand_type=config["fes"]["gb"]["demand"]["Technology Detail"].keys(),
        ),
        dsr=expand(
            resources("gb-model/{{fes_scenario}}/regional_{sector}_dsr_inc_eur.csv"),
            sector=["residential", "iandc", "iandc_heat", "ev", "residential_heat"],
        ),
        ev_data=expand(
            resources("gb-model/{{fes_scenario}}/regional_ev_{ev_data}_inc_eur.csv"),
            ev_data=["v2g_storage", "v2g"],
        )
        + [resources("avail_profile_s_clustered.csv")],
        network=resources("networks/base_s_clustered.nc"),
        powerplants=resources(
            "gb-model/{fes_scenario}/fes_powerplants_inc_tech_data.csv"
        ),
        tech_costs=Path(COSTS_DATASET["folder"])
        / f"costs_{config['scenario']['planning_horizons'][0]}.csv",
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
            "gb-model/{fes_scenario}/regional_battery_storage_capacity_inc_eur.csv"
        ),
        H2_data=[
            resources("gb-model/{fes_scenario}/regional_H2_demand_annual_inc_eur.csv"),
            resources(
                "gb-model/{fes_scenario}/regional_non_networked_electrolysis_demand_annual_inc_eur.csv"
            ),
            resources(
                "gb-model/{fes_scenario}/regional_H2_storage_capacity_inc_eur_inc_tech_data.csv"
            ),
            resources(
                "gb-model/{fes_scenario}/regional_grid_electrolysis_capacities_inc_eur_inc_tech_data.csv"
            ),
        ],
        intermediate_data=[
            # TODO: calculate intra_gb availability per line/boundary before this point (currently only per TO)
            resources("gb-model/intra_gb_transmission_availability.csv"),
        ],
    output:
        network=resources("networks/{fes_scenario}/composed_{clusters}/{year}.nc"),
    log:
        logs("compose_network_{clusters}_{fes_scenario}_{year}.log"),
    resources:
        mem_mb=4000,
    wildcard_constraints:
        # We only accept clustered clusters
        clusters="clustered",
    script:
        scripts("gb_model/compose_network.py")


rule gb_all:
    input:
        expand(
            RESULTS + "constraint_costs/{fes_scenario}.csv",
            fes_scenario=config["fes"]["scenario_mapping"].keys(),
        ),


rule gb_compose_all:
    input:
        expand(
            resources("networks/{fes_scenario}/composed_clustered/{year}.nc"),
            fes_scenario=config["fes"]["scenario_mapping"].keys(),
            year=range(
                config["redispatch"]["year_range_incl"][0],
                config["redispatch"]["year_range_incl"][1] + 1,
            ),
        ),

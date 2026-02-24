# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Boundary constrained redispatch run rules.
"""


localrules:
    fetch_bid_offer_data_elexon,


rule process_CfD_strike_prices:
    message:
        "get strike price for low carbon contracts"
    params:
        carrier_mapping=config["low_carbon_register"]["carrier_mapping"],
        fes_year=config["fes"]["fes_year"],
    input:
        register="data/gb-model/downloaded/low-carbon-contracts.csv",
    output:
        csv=resources("gb-model/CfD_strike_prices.csv"),
    log:
        logs("process_CfD_strike_prices.log"),
    script:
        scripts("gb_model/redispatch/process_CfD_strike_prices.py")


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
        scripts("gb_model/redispatch/extract_etys_boundary_capabilities.py")


rule prepare_future_etys_caps:
    message:
        "Prepare future ETYS boundary capabilities for {wildcards.fes_scenario} scenario"
    params:
        sheet_name=config_provider("etys", "future_capacities_sheet_name"),
        year_range=config_provider("redispatch", "year_range_incl"),
    input:
        current_caps=resources("gb-model/etys_boundary_capabilities.csv"),
        future_caps="data/gb-model/downloaded/etys_chart_data.xlsx",
    output:
        csv=resources("gb-model/{fes_scenario}/future_etys_boundary_capabilities.csv"),
    log:
        logs("prepare_future_etys_caps_{fes_scenario}.log"),
    script:
        scripts("gb_model/redispatch/prepare_future_etys_caps.py")


rule fetch_bid_offer_data_elexon:
    message:
        "Get bid/offer data from Elexon"
    params:
        technology_mapping=config_provider("redispatch", "elexon", "technology_mapping"),
        api_bmu_fuel_map=config_provider("redispatch", "elexon", "api_bmu_fuel_map"),
        max_concurrent_requests=config_provider(
            "redispatch", "elexon", "max_concurrent_requests"
        ),
    input:
        bmu_fuel_map_path="data/gb-model/BMUFuelType.xlsx",
    output:
        csv=resources("gb-model/bids_and_offers/Elexon/{bod_year}.csv"),
    resources:
        # Used to avoid the same rule running simultaneously (and exceeding max concurrent requests).
        parallel_elexon_download=1,
    log:
        logs("fetch_bid_offer_data_elexon_{bod_year}.log"),
    script:
        scripts("gb_model/redispatch/fetch_bid_offer_data_elexon.py")


rule calculate_bid_offer_multipliers:
    message:
        "Calculate bid / offer multipliers for conventional generators"
    params:
        costs_config=config["costs"],
        technology_mapping=config_provider("redispatch", "elexon", "technology_mapping"),
    input:
        fes_power_costs=resources("gb-model/fes-costing/AS.1 (Power Gen).csv"),
        fes_carbon_costs=resources("gb-model/fes-costing/AS.7 (Carbon Cost).csv"),
        tech_costs=Path(COSTS_DATASET["folder"])
        / f"costs_{config['scenario']['planning_horizons'][0]}.csv",
        bid_offer_data=expand(
            resources("gb-model/bids_and_offers/Elexon/{bod_year}.csv"),
            bod_year=config["redispatch"]["elexon"]["years"],
        ),
    output:
        csv=resources("gb-model/{fes_scenario}/bid_offer_multipliers.csv"),
    log:
        logs("calculate_bid_offer_multipliers_{fes_scenario}.log"),
    script:
        scripts("gb_model/redispatch/calculate_bid_offer_multipliers.py")


rule calc_interconnector_bid_offer_profile:
    message:
        "Calculate interconnector bid/offer profiles"
    input:
        bids_and_offers=resources("gb-model/{fes_scenario}/bid_offer_multipliers.csv"),
        unconstrained_result=RESULTS
        + "networks/{fes_scenario}/unconstrained_clustered/{year}.nc",
    output:
        bid_offer_profile=resources(
            "gb-model/{fes_scenario}/interconnector_bid_offer_profile/{year}.csv"
        ),
    log:
        logs("calc_interconnector_bid_offer_profile_{fes_scenario}_{year}.log"),
    script:
        scripts("gb_model/redispatch/calc_interconnector_bid_offer_profile.py")


rule prepare_constrained_network:
    message:
        "Prepare network for constrained optimization"
    params:
        unconstrain_lines_and_links=config["redispatch"]["unconstrain_lines_and_links"],
    input:
        network=resources("networks/{fes_scenario}/composed_clustered/{year}.nc"),
        unconstrained_result=RESULTS
        + "networks/{fes_scenario}/unconstrained_clustered/{year}.nc",
        renewable_strike_prices=resources("gb-model/CfD_strike_prices.csv"),
        interconnector_bid_offer=resources(
            "gb-model/{fes_scenario}/interconnector_bid_offer_profile/{year}.csv"
        ),
        bids_and_offers=resources("gb-model/{fes_scenario}/bid_offer_multipliers.csv"),
    output:
        network=resources("networks/{fes_scenario}/constrained_clustered/{year}.nc"),
    log:
        logs("prepare_constrained_network_{fes_scenario}_{year}.log"),
    script:
        scripts("gb_model/redispatch/prepare_constrained_network.py")


rule solve_constrained:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=Path(workflow.snakefile).parent
        / scripts("gb_model/redispatch/custom_constraints.py"),
        etys_boundaries_to_lines=config_provider("etys", "boundaries_lines"),
        etys_boundaries_to_links=config_provider("etys", "boundaries_links"),
        manual_future_etys_caps=(
            config_provider("etys", "manual_future_capacities")
            if config["etys"]["use_future_capacities"]
            else {}
        ),
    input:
        network=resources("networks/{fes_scenario}/constrained_clustered/{year}.nc"),
        current_etys_caps=resources("gb-model/etys_boundary_capabilities.csv"),
        future_etys_caps=(
            resources("gb-model/{fes_scenario}/future_etys_boundary_capabilities.csv")
            if config["etys"]["use_future_capacities"]
            else []
        ),
    output:
        network=RESULTS + "networks/{fes_scenario}/constrained_clustered/{year}.nc",
        config=RESULTS
        + "configs/{fes_scenario}/config.constrained_clustered/{year}.yaml",
    log:
        solver=normpath(
            RESULTS
            + "logs/solve_network/{fes_scenario}/constrained_clustered/{year}_solver.log"
        ),
        memory=RESULTS
        + "logs/solve_network/{fes_scenario}/constrained_clustered/{year}_memory.log",
        python=RESULTS
        + "logs/solve_network/{fes_scenario}/constrained_clustered/{year}_python.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_network/{fes_scenario}/constrained_clustered/{year}"
        )
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    script:
        scripts("solve_network.py")


rule calculate_constraint_costs:
    message:
        "Calculate total constraint cost across all years for `{wildcards.fes_scenario}` scenario"
    params:
        constraint_cost_extra_years=config["redispatch"]["constraint_cost_extra_years"],
    input:
        networks=expand(
            RESULTS + "networks/{{fes_scenario}}/constrained_clustered/{year}.nc",
            year=range(
                config["redispatch"]["year_range_incl"][0],
                config["redispatch"]["year_range_incl"][1] + 1,
            ),
        ),
    output:
        csv=RESULTS + "constraint_costs/{fes_scenario}.csv",
    script:
        scripts("gb_model/redispatch/calculate_constraint_costs.py")

# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Unconstrained dispatch run rules.
"""


rule prepare_unconstrained:
    message:
        "Prepare network for unconstrained optimization"
    input:
        network=resources("networks/{fes_scenario}/composed_clustered/{year}.nc"),
    output:
        network=resources("networks/{fes_scenario}/unconstrained_clustered/{year}.nc"),
    params:
        load_shedding_cost_above_marginal=config["fes"]["eur"][
            "load_shedding_cost_above_marginal"
        ],
    log:
        logs("prepare_unconstrained_network_{fes_scenario}_{year}.log"),
    script:
        scripts("gb_model/dispatch/prepare_unconstrained_network.py")


rule solve_unconstrained:
    params:
        solving=config_provider("solving"),
        foresight=config_provider("foresight"),
        co2_sequestration_potential=config_provider(
            "sector", "co2_sequestration_potential", default=200
        ),
        custom_extra_functionality=Path(workflow.snakefile).parent
        / scripts("gb_model/dispatch/custom_constraints.py"),
        nuclear_max_annual_capacity_factor=config["conventional"]["nuclear"][
            "max_annual_capacity_factor"
        ],
        nuclear_min_annual_capacity_factor=config["conventional"]["nuclear"][
            "min_annual_capacity_factor"
        ],
    input:
        network=resources("networks/{fes_scenario}/unconstrained_clustered/{year}.nc"),
    output:
        network=RESULTS + "networks/{fes_scenario}/unconstrained_clustered/{year}.nc",
        config=RESULTS
        + "configs/{fes_scenario}/config.unconstrained_clustered/{year}.yaml",
    log:
        solver=normpath(
            RESULTS
            + "logs/solve_network/{fes_scenario}/unconstrained_clustered/{year}_solver.log"
        ),
        memory=RESULTS
        + "logs/solve_network/{fes_scenario}/unconstrained_clustered/{year}_memory.log",
        python=RESULTS
        + "logs/solve_network/{fes_scenario}/unconstrained_clustered/{year}_python.log",
    benchmark:
        (
            RESULTS
            + "benchmarks/solve_network/{fes_scenario}/unconstrained_clustered/{year}"
        )
    threads: solver_threads
    resources:
        mem_mb=config_provider("solving", "mem_mb"),
        runtime=config_provider("solving", "runtime", default="6h"),
    shadow:
        shadow_config
    script:
        scripts("solve_network.py")

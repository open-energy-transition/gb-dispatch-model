<!-- SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur> -->
<!-- SPDX-FileCopyrightText: Contributors to gb-dispatch-model -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Introduction {#gb_intro}

The GB dispatch model represents Great Britain's future energy system for use in market, constraint cost and resource adequacy assessments.
It uses the NESO Future Energy Scenarios (FES) to define the future system and various public data sources to fill data gaps not provided by the published FES outputs.

This model is built using a reproducible [Snakemake](https://snakemake.github.io/) workflow from entirely public and open-access data sources.
It is distributed over a collection of PyPSA networks representing different future years.
Each network is solved separately and then the results are aggregated to understand system impacts over the lifetime of energy assets.

Here, we will briefly describe some of the core concepts of the workflow and practicalities of navigating the project directory.
For more information on the methodology we follow see our [system representation pages](system_overview.md#system-overview_gb).

## Quick guide

To prepare for and run the workflow, follow these steps:

1. create your working environment by following our [installation steps](installation.md#installation_gb).
2. [Run the workflow](run.md#run_gb) using default configuration options and `pixi` tasks.
3. Analyse the results of the solved PyPSA networks to understand the system dispatch decisions and redispatch requirements.
4. If desired, modify the default [configuration](configuration.md#model_config_gb) to meet your needs and re-run the workflow.

## Scenarios, Configuration and Modification

The GB dispatch model can be used to run multiple scenarios using the `snakemake` [wildcards feature](https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards).
We use wildcards to produce multiple files with one rule, with each file following a [regular expression](https://en.wikipedia.org/wiki/Regular_expression) pattern.
One can think of a wildcard as a parameter that shows up in the input/output file names and thereby determines which rules to run, what data to retrieve and what files to produce.
Details are explained in [Wildcards](wildcards.md#wildcards_gb).

The model also has several further configuration options collected in the `config/config.default.gb.yaml` file.
Options are explained in [Configuration](configuration.md#model_config_gb).

## Folder Structure

- `scripts`: Includes all the Python scripts executed by the `snakemake` rules.
- `scripts/gb_model`: Includes the Python scripts executed by the `snakemake` rules that are specific to the GB dispatch model.
- `rules`: Includes all the `snakemake` rules loaded in the `Snakefile`.
- `rules/gb-model`: Includes all the `snakemake` rules loaded in the `Snakefile` that are specific to the GB dispatch model.
- `envs`: The files in this directory are not used as part of the GB dispatch model workflow.
- `data`: Includes input data that is shipped with the repository by default or is retrieved from public sources automatically by the workflow.
- `data/gb-model`: Includes auto-generated and manually pre-prepared input data that is specific to the GB dispatch model.
- `cutouts`: Stores raw weather data cutouts from `atlite`.
- `resources`: Stores intermediate results of the workflow which can be picked up again by subsequent rules.
- `resources/<run-name>/gb-model`: Stores intermediate results of the workflow specific to the GB dispatch model.
- `results`: Stores the solved PyPSA network data, summary files and plots.
- `logs`: Stores log files.
- `benchmarks`: Stores `snakemake` benchmarks.
- `doc`: Includes the documentation of PyPSA-Eur.
- `docker`: Includes some optional Docker environments.

## System Requirements

Building the model with the scripts in this repository runs on a regular computer.
You will need around 40-50GB of disk space to store input, intermediate, and result data, and at least 8GB of RAM to run the pre-processing steps (16-32GB if running several workflow jobs in parallel).
Optimising for dispatch and redispatch decisions at an hourly resolution requires a strong interior-point solver like [Gurobi](http://www.gurobi.com/) or [CPLEX](https://www.ibm.com/analytics/cplex-optimizer) with at least 16 GB of RAM per parallel job.
Open-source solvers like [HiGHS](https://highs.dev) can also be used for smaller problems, particularly if using the `HiPO` solver option.

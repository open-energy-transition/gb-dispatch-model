<!-- SPDX-FileCopyrightText: gb-dispatch-model contributors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Frequently Asked Questions {#faq_gb}

## Installation & Setup

??? question "Q: I'm getting \"pixi: command not found\" when trying to install. What should I do?"

    First, install pixi from https://pixi.sh/latest/ or by using `conda` (`conda install pixi`).
    Once installed, you can install your pixi environment in a terminal by running `pixi install -e gb-model`.

??? question "Q: What are the system requirements to run gb-dispatch-model?"

    The model can run on a modern laptop but will require distributed resources to benefit from parallelisation to reduce overall runtime:

    - **For data processing only**: 3GB+ RAM, a few hours of CPU time
    - **For full optimisation**: 7-20GB RAM (full temporal resolution) and 0.5 - 1 hour per optimisation run (default 160 optimisation runs across all scenarios and years)
    - **Disk space**: ~35GB for input and intermediate files; ~35GB for output files

    The workflow *can* be run on Windows.
    However, we recommend using a Linux machine or macOS, or Windows Subsystem for Linux.

??? question "Q: Can I run this on a laptop?"

    Yes, but we recommend you simplify the networks being solved if doing so.
    Use `config/config.gb.time-segment.yaml` to reduce the time resolution and minimize the optimisation problem size.

??? question "Q: Snakemake is failing to run. What should I do?"

    Check the error message carefully. Common issues include:

    - Missing pixi installation (see Q1)
    - Missing environment variables defining API keys for data retrieval.
      You should have set `ENTSO_E_API_KEY=<your-api-key>` in a project-level `.env` file.
    - Running from an incorrect directory.
      You should be running commands in the terminal from within the `gb-dispatch-model` directory.
    - You are requesting a file that cannot be generated.
      If you are requesting that a specific intermediate file is generated, ensure you have the full filepath correctly defined, given your current configuration.
      E.g., `pixi run -e gb-model snakemake resources/<run-name>/gb-model/<scenario>/resistive_heater_demand/<year>.csv` includes three configurable items:
        - `<run-name>` is defined in the configuration under `run.name`;
        - `scenario` is one of the scenarios defined in the configuration under `fes.scenario_mapping`;
        - `year` is one of the years defined in the configuration under `redispatch.year_range_incl`.

      The final filepath with the default configuration could be `pixi run -e gb-model snakemake resources/GB/gb-model/HT/resistive_heater_demand/2040.csv`
    - Data retrieval from a remote resource failed.
      This is usually resolved by running the workflow again.
      If it persists, you can find options for limiting request rates within the [configuration](configuration.md#model_config_gb).

## Running the Model

??? question "Q: How do I run the full model?"

    Use the command:

    ```bash
    pixi run model
    ```

    This runs all data processing, network composition, and optimisation steps sequentially for all model years and FES scenarios.

??? question "Q: Can I run only the data processing steps?"

    Yes, use:

    ```bash
    pixi run compose_networks
    ```

    This processes all input data and creates network files without running optimisation.
    It's useful for checking input data or preparing PyPSA networks to share.

??? question "Q: Can I run specific workflow steps?"

    Yes, use the standard Snakemake syntax. For example:

    ```bash
    pixi run -e gb-model snakemake --target rule_name -v
    ```

    Or:

    ```bash
    pixi run -e gb-model snakemake /path/to/output.ext
    ```

??? question "Q: Can I run dispatch+redispatch for only a single year and scenario?"

    Yes, use the standard Snakemake syntax to request the output file for the given scenario and model year.
    The file is of the form `results/<run-name>/networks/<scenario>/constrained_clustered/<year>.nc`
    For example:

    ```bash
    pixi run -e gb-model snakemake "results/GB/networks/HT/constrained_clustered/2030.nc"
    ```

??? question "Q: How long does the workflow take to run?"

    This depends on configuration and available solvers:

    - **Data processing only**: 3 - 4 hours with no parallelisation (approximately 1 hour with 4 parallel jobs).
    - **Optimisation (full temporal resolution, 4 cores, gurobi solver)**: approximately 30 - 45 minutes per model year for dispatch optimisation, 20 - 30 minutes per model year for redispatch optimisation.
    - **Optimisation (full temporal resolution, 4 cores, HiGHS HiPO solver)**: approximately 2.5 - 3 hours per model year for dispatch optimisation, 1.5 - 2 hours per model year for redispatch optimisation.
    - **Optimisation (x4 temporal resolution reduction, 4 cores, gurobi solver)**: approximately 10 - 15 minutes for dispatch optimisation, 5 - 10 minutes per model year for redispatch optimisation.

    Use `pixi run model --configfile 'config/config.gb.time-segment.yaml'` to optimise with x4 temporal resolution reduction.

    When testing that the workflow runs as expected, you can also [limit your runs to a single FES scenario](#faq_gb_single_scenario) to reduce overall runtime.

??? question "Q: What's the difference between \"dispatch\" and \"redispatch\"?"

    See the [detailed documentation](system_dispatch_redispatch.md#system-dispatch_redispatch).
    Briefly:

    - **Dispatch**: Initial optimisation with relaxed constraints, reflecting the GB day-ahead market.
    - **Redispatch**: Optimisation considering realistic physical network constraints, reflecting the GB balancing market.

    Both are computed when running the full model.

??? question "Q: What effect does running each future year separately have?"

    To enable parallelisation of the optimisation step, and to keep memory requirements low for individual runs, we separate out each future year into its own model and optimise it independently.
    This does mean that long-duration storage devices can have a mismatch of stored energy on the boundary between two years.
    For instance, there might be 10GWh of hydrogen stored at the end of 2030 but only 2GWh stored at the beginning of 2031.
    Some energy has been destroyed between these two year.
    Similarly, it is possible that some energy is created between years.

    We recognise this problem but believe it has a limited or even negligible impact on model results and therefore deem it a reasonable sacrifice of model accuracy for a gain in speed.
    From tests, recorded in [a GitHub issue](https://github.com/open-energy-transition/gb-dispatch-model/issues/218), we found that pumped hydro storage creates/destroys energy equivalent to less than a day of European load.
    In the worst year (2030, when it is first introduced into the system), hydrogen storage creates energy equivalent to 6 days of European load.
    This is clearly more impactful, but given our cyclic storage requirements, the hydrogen storage must also have this level of energy stored at the end of the year, so it cannot use this energy in favour of generation.

    The alternatives are:

    1. Run all years in one model.
       Runs would then have a very high RAM and time footprint, at least 20x higher than running a single year in isolation.
       Inevitably, hardware limits would require aggressive temporal aggregation, which is likely to reduce result accuracy more than the storage energy created/destroyed on the year boundary.
    2. Run all years in series and pass the energy content of stored devices onto the next year.
       This will slow the time to solve but would not *require* temporal aggregation since RAM requirements would not increase.
       It is a feasible solution to consider, but the change in capacity between years would then need to be accounted for.
    3. Run all years together in a rolling-horizon optimisation.
       Since we are not optimising capacity, we could run a rolling-horizon optimisation for the full time horizon (e.g., week-long rolling windows over 20 years).

    We would recommend using the third option, alongside a representation of the [water value](https://docs.pypsa.org/latest/examples/water-value/) from annual perfect foresight runs.

## Configuration & Scenarios

??? question "Q: What configuration files should I use?"

    Start with the default FES2024 configuration, which you can find in `config/config.gb.2024.yaml`.

    - To use regions more closely aligned with the ETYS boundaries, use `config/config.gb.etys-subset.yaml`.
    - For faster testing, use `config/config.gb.time-segment.yaml` to coarsen the time dimension.
    - To create a new OSM build release configured for use with this project, use `config/config.gb.osm-release.yaml`

    These configuration files will need to be called explicitly to invoke their content, e.g.:

    ```bash
    pixi run model --configfile="config/config.gb.etys-subset.yaml"
    ```

    Or create your own overrides to the default configuration by placing it in `config/config.yaml`.
    You do not need to explicitly reference `config/config.yaml`, its content will be automatically considered by the workflow.

    Each configuration file will selectively override the existing configuration, including overrides made by previously applied files.
    The content of `config/config.yaml` will be applied after all other configuration.

??? question "Q: What is the FES and how do I change the FES year?"

    The FES (Future Energy Scenarios) are developed by the GB National Energy System Operator (NESO).
    They are a large-scale modelling effort to define possible pathways for GB's future energy system.
    The output of the FES, which includes supply-side data (e.g., generation and storage technology capacities) and demand-side data (e.g., heat pump / EV uptake), is used as the input to our dispatch modelling.

    The FES is published annually.
    Our workflow should support different FES years (≥FES2023) and scenarios (a.k.a. "pathways", e.g. Holistic Transition / HT).
    By default, we support FES2024 and all its scenarios.
    To model a different FES year, you will need to create your own version of the configuration file with appropriate updates on input data (under the `urls` configuration option) and data table processing (e.g., all `sheet-config` entries).
    This is not trivial and we expect that it may require changes to be in the workflow scripts themselves.

<a id="faq_gb_single_scenario"></a>

??? question "Q: How do I run the workflow for just one FES scenario?"

    By default, the workflow runs over all scenarios.
    To limit it to a subset of scenario, provide a configuration override which updates the `fes.scenarios` list.

??? question "Q: How do I change the temporal resolution?"

    You can request the application of [PyPSA time aggregation methods](https://docs.pypsa.org/latest/api/networks/cluster/#pypsa.Network.cluster.temporal) using the `time_aggregation` configuration item.
    See `config/config.gb.time-segment.yaml` for an example.

??? question "Q: What does each configuration option do?"

    You can find some details about each configuration option in our [documentation](https://gb-dispatch-model.readthedocs.io/en/latest/gb-model/configuration.html).
    You will likely find it useful to use this resource together with our FES2024 configuration in `config/config.gb.2024.yaml` to better understand some of the nested options that are empty by default.

??? question "Q: What are some of the key configuration options I might want to modify?"

    For the most part, you will not need to edit the configuration options unless you update the input data source (e.g., if updating to a different FES year).
    Some options that you may wish to modify with the default FES year include:

    - The weather year used to generate hourly load and renewable energy capacity factor profiles.
      This can be modified together under the `atlite`, `snapshots` and `energy.energy_totals_year` keys.
    - The countries that are considered.
      By default, we model all available countries (limited by the FES European data).
      You can reduce the number that are modelled by defining a list with fewer items under the `countries` key.
    - The FES scenarios to model, set under `fes.scenarios`.
    - You may wish to update the solver and the solver options used, both set under `solving.solver`.
      If you do not have access to a commercial solver, we recommend you use the `highs` solver.
    - CHP, nuclear, and EV operating characteristics (defined under `chp`, `conventional.nuclear`, and `ev` respectively) are based on our own assumptions, and may benefit from being tweaked to achieve expected operation of these technology groups.
    - The GB interconnector investment plan, defined under `interconnectors.plan`, is based on a review of limited information from the FES.
      The total annual interconnector capacity per scenario roughly match those from the FES, but there are many possible alternatives with the same capacity that could meet the same capacity totals.
    - For some datasets, we collate historical data across multiple past years.
      The selected past years are configurable in these cases, and can be extended or restricted as preferred.
      This includes the transmission availability reports (`transmission_availability.years`), the ENTSO-E API generator availability data (`entsoe_unavailability.<start|end>_date`), and the Elexon API bid/offer data (`redispatch.elexon.years`).

## Data & Input Files

??? question "Q: Where does the model get its input data from?"

    See [data sources documentation](data_sources.md#data_sources_gb). Key sources include:

    - **Weather/resource data**: ERA5 and SARAH-3 via atlite.
    - **Electricity demand profiles**: ENTSO-E Transparency Platform.
    - **Power plant characteristics**: powerplantmatching.
    - **Network topology**: OpenStreetMap (via PyPSA-Eur).
    - **Costs**: FES datasets and PyPSA-Eur defaults.
    - **GB-specific data**: various NESO and DUKES datasets and reports.

??? question "Q: What assumptions have been made where data is unavailable?"

    Not all data required to build and run the model is available from public sources.
    In these cases, we have made assumptions to fill gaps.
    Where possible, we have made these assumptions configurable so that you can choose to override them with your own, if desired.

    To find out more about the assumptions, check the "key assumptions" section of each system sub-page.
    For instance, key assumptions of the hydrogen subsystem can be found [here](system_hydrogen.md#system-hydrogen-assumptions).

??? question "Q: How much do you rely on the PyPSA-Eur workflow to prepare input data?"

    Although most of the data used in this workflow is from separate sources to the "parent" PyPSA-Eur workflow, we still use several parts of it to prepare our PyPSA network and to fill data gaps:

    - **Spatial representation**: The spatial representation of GB and neighbouring countries and of the transmission network is based on PyPSA-Eur data retrieval and processing steps.
      This includes transmission network reduction once the final model regions have been set by the GB workflow.
    - **Timeseries data**: Hourly profiles are derived from the PyPSA-Eur workflow.
      This includes renewable generator capacity factors, demand profiles, and heat pump COP profiles.
    - **European data gap filling**: When mapping total annual demand to sectoral demands in European countries, we use historical sectoral demands in all countries derived from the PyPSA-Eur workflow.
    - **Technology characteristics**: Where unavailable in the FES data (e.g., operating efficiency), technology characteristics are derived from the nearest equivalent technology type from the PyPSA-Eur workflow.

    To better understand when we are relying on the PyPSA-Eur workflow, refer to the "Data Processing Workflow" section of each system component / subsystem page.
    For instance, the data processing workflow of the hydrogen subsystem can be found [here](system_hydrogen.md#system-hydrogen-workflow).

??? question "Q: How frequently is the input data updated?"

    This input data is fixed for a given FES year, so will be updated annually in line with new FES releases.

??? question "Q: Can I use my own data instead of the defaults?"

    Yes, you can, although it can prove to be quite involved.

    If you want to modify a dataset linked to the base PyPSA-Eur workflow, you can add an entry to `data/versions.csv` and then make reference to that entry in your configuration under the `data` key.

    If you want to modify a gb-dispatch-model dataset, referenced under the `urls` key in the configuration, you can provide a new URL.
    However, you should be aware that other configuration options may need to be updated to process this new entry into the correct format.
    For instance, a new DUKES table 5.11 dataset may need a different set of options passed to the CSV reader to capture the data in the sheet of interest (under `dukes-5.11.sheet-config` configuration option).

## Results & Output

??? question "Q: Where are the results stored?"

    In the `results/` directory:

    - `networks/`: Solved PyPSA network files.
    - `constraint_costs/`: CSV files with constraint costs per scenario.

??? question "Q: How do I interpret the results?"

    Results include:

    - **Optimal dispatch**: Generator & storage dispatch at each time step
    - **Network flows**: Power flows on transmission lines
    - **Cost breakdown**: Total system cost and component costs

    See [the PyPSA documentation](https://docs.pypsa.org/latest/user-guide/statistics/) for detailed guidance on generating appropriate statistics to interpret results.

??? question "Q: Can I export results in different formats?"

    The main results are stored as `.nc` (NetCDF) files.
    You can read these with PyPSA and then export them to a different format (CSVs, Excel, etc.) if you desire.
    See the [PyPSA import/export documentation](https://docs.pypsa.org/latest/user-guide/import-export/) for more information.

## Solvers & Optimisation

??? question "Q: What solvers can I use?"

    The model supports and solvers that are also supported by [linopy](https://linopy.readthedocs.io/en/latest/).
    We recommend that you use a commercial or GPU-accelerated solver to ensure reasonable runtimes.

    Specify the solver in your configuration under `solving.solver.name`.

??? question "Q: The solver is running very slowly. What can I do?"

    Options include:

    1. Use timeseries aggregation.
       We recommend time segmentation (a.k.a. chronological typical hours): `pixi run model --configfile='config/config.gb.time-segment.yaml'`
    2. Use a commercial solver: Gurobi will reach a solution faster than HiGHS
    3. Allocate more CPU cores: This can be set under `solving.solver_options`, for the solver option group defined in `solving.solver.options`.

??? question "Q: What does \"infeasible\" or \"unbounded\" mean?"

    These indicate solver errors:

    - **Infeasible**: No solution exists satisfying all constraints (physically impossible scenario)
    - **Unbounded**: Solver found no optimal solution (usually indicates a model error)

    Infeasibilities can occur when using aggressive timeseries aggregation.
    Loosening some technology constraints in the configuration can help (e.g., under `conventional.nuclear`, `chp`, and `fes.gb.dsr_hours`).

??? question "Q: Can I run the optimisation with my own solver settings?"

    Yes, modify the `solving.solver_options` in your configuration file, for the solver option group you have defined under `solving.solver.options`.
    See existing solver option group settings for the settings we tend to consider worth modifying.

## Troubleshooting

??? question "Q: I'm getting out-of-memory errors. What can I do?"

    The model requires substantial RAM for optimisation:

    1. Use timeseries aggregation to reduce the problem size
    2. Run on a machine with more memory
    3. Run fewer / no parallel jobs.
       To run jobs in series, set `--cores 1`.

??? question "Q: Snakemake says a rule failed. How do I debug it?"

    Check the log file in `logs/`. For more detail:

    1. Re-run with verbose output: `pixi run snakemake -v`
    2. Check the specific rule logs in `logs/` directory.
       They are usually named after the rule they are logging, plus any specific wildcards that were used: `logs/<rule-name>_<wildcards>`
    3. Examine the input files for a specific rule.
       You can identify inputs by inspecting the terminal output when dry-running snakemake `pixi run snakemake <target-rule-or-file> --dry-run`, e.g.:

       ```console
       rule identify_boundary_crossings:
          input: resources/GB/networks/base.nc, resources/GB/networks/base_s_clustered.nc, resources/GB/linemap_base_s_clustered.csv, data/gb-model/downloaded/gb-etys-boundaries.zip, resources/GB/gb-model/merged_shapes.geojson, resources/GB/gb-model/etys_boundary_capabilities.csv
          output: resources/GB/gb-model/etys_boundary_crossings.csv
          log: logs/GB/identify_boundary_crossings.log
          jobid: 0
          reason: Input files updated by another job: resources/GB/linemap_base_s_clustered.csv, resources/GB/networks/base_s_clustered.nc
          resources: tmpdir=/var/folders/4n/5q9kdtps1334qh_fqx5jl2xw0000gn/T
       ```

??? question "Q: How do I update the model code from PyPSA-Eur?"

    This repository is a soft-fork of PyPSA-Eur.
    You can merge in changes from upstream as follows:

    ```console
    git checkout master
    git remote add upstream git@github.com:open-energy-transition/pypsa-eur.git
    git fetch upstream
    git merge upstream/master
    ```

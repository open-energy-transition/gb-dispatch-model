<!-- SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur> -->
<!-- SPDX-FileCopyrightText: gb-dispatch-model contributors -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Configuration {#model_config_gb}

gb-dispatch-model has several additional configuration options above and beyond those set by PyPSA-Eur, which are documented in this section.
For base PyPSA-Eur configuration, see references [below](#base_config).

## Configuration Files

Any gb-dispatch-model configuration can be set in a `.yaml` file.
The default configuration `config/config.default.gb.yaml` is maintained in the repository and covers all the options that are used / can be set.
The configuration `config/config.gb.2024.yaml` is an opinionated configuration we maintain that uses FES2024 data and is always applied on top of `config/config.default.gb.yaml`.

To pass your own configuration, you can create a new file, e.g. `my_config.yaml`, and specify the options you want to change.
They will override the default settings and options which are not set, will be inherited from the defaults above.

Another way is to use the `config/config.yaml` file, which does not exist in the repository and is also not tracked by git.
But snakemake will always use this file if it exists.
This way you can run snakemake with a custom config without having to specify the config file each time.

Configuration order of precedence is as follows:
1. Command line options specified with `--config` (optional)
2. Custom configuration file specified with `--configfile` (optional)
3. The `config/config.yaml` file (optional)
4. The default configuration files `config/config.default.yaml` and `config/plotting.default.yaml`

To use your custom configuration file, you need to pass it to the `snakemake` command using the `--configfile` option:

```console
$ snakemake -call --configfile my_config.yaml
```

## `fes_costs` {#fes_costs_cf}

{{ schema_table("fes_costs", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("fes_costs", source="config.default.gb.yaml") }}
```

## `low_carbon_register` {#low_carbon_register_cf}

{{ schema_table("low_carbon_register", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("low_carbon_register", source="config.default.gb.yaml") }}
```

## `chp` {#chp_cf}

{{ schema_table("chp", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("chp", source="config.default.gb.yaml") }}
```

## `urls` {#urls_cf}

{{ schema_table("urls", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("urls", source="config.default.gb.yaml") }}
```

## `target_crs` {#target_crs_cf}

Target coordinate reference system (e.g., `EPSG:27700`).

- **Type:** string
- **Default:** `EPSG:27700`

**YAML Syntax**

```yaml
{{ yaml_section("target_crs", source="config.default.gb.yaml") }}
```

## `region_operations` {#region_operations_cf}

{{ schema_table("region_operations", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("region_operations", source="config.default.gb.yaml") }}
```

## `etys` {#etys_cf}

{{ schema_table("etys", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("etys", source="config.default.gb.yaml") }}
```

## `entsoe_unavailability` {#entsoe_unavailability_cf}

{{ schema_table("entsoe_unavailability", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("entsoe_unavailability", source="config.default.gb.yaml") }}
```

## `transmission_availability` {#transmission_availability_cf}

{{ schema_table("transmission_availability", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("transmission_availability", source="config.default.gb.yaml") }}
```

## `dukes-5.11` {#dukes_5_11_cf}

{{ schema_table("dukes-5.11", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("dukes-5.11", source="config.default.gb.yaml") }}
```

## `grid_supply_points` {#grid_supply_points_cf}

{{ schema_table("grid_supply_points", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("grid_supply_points", source="config.default.gb.yaml") }}
```

## `fes` {#fes_cf}

{{ schema_table("fes", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("fes", source="config.default.gb.yaml") }}
```

## `ev` {#ev_cf}

{{ schema_table("ev", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("ev", source="config.default.gb.yaml") }}
```

## `interconnectors` {#interconnectors_cf}

{{ schema_table("interconnectors", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("interconnectors", source="config.default.gb.yaml") }}
```

## `redispatch` {#redispatch_cf}

{{ schema_table("redispatch", source="schema.default.gb.json") }}

**YAML Syntax**

```yaml
{{ yaml_section("redispatch", source="config.default.gb.yaml") }}
```

## `time_aggregation`

List of time aggregation configurations to apply sequentially. If multiple configurations are provided, they will be applied in the order they appear in the list. See [PyPSA documentation](https://docs.pypsa.org/latest/api/networks/cluster/#pypsa.Network.cluster.temporal) for details on the available time aggregation methods and their parameters.

- **Type:** list of `{method, parameters}`
- **Default:** `[]`

**YAML Syntax**

```yaml
{{ yaml_section("time_aggregation", source="config.default.gb.yaml") }}
```

## Base PyPSA-Eur config {#base_config}

Follow the links below to get more information about the base PyPSA-Eur configuration.

- [version](../configuration.md#version_cf)
- [tutorial](../configuration.md#tutorial_cf)
- [logging](../configuration.md#logging_cf)
- [remote](../configuration.md#remote_cf)
- [run](../configuration.md#run_cf)
- [foresight](../configuration.md#foresight_cf)
- [snapshots](../configuration.md#snapshots_cf)
- [enable](../configuration.md#enable_cf)
- [CO2_budget](../configuration.md#CO2_budget_cf)
- [electricity](../configuration.md#electricity_cf)
- [atlite](../configuration.md#atlite_cf)
- [renewable](../configuration.md#renewable_cf)
- [conventional](../configuration.md#conventional_cf)
- [lines](../configuration.md#lines_cf)
- [links](../configuration.md#links_cf)
- [transmission_projects](../configuration.md#transmission_projects_cf)
- [transformers](../configuration.md#transformers_cf)
- [load](../configuration.md#load_cf)
- [energy](../configuration.md#energy_cf)
- [biomass](../configuration.md#biomass_cf)
- [solar_thermal](../configuration.md#solar_thermal_cf)
- [existing_capacities](../configuration.md#existing_capacities_cf)
- [sector](../configuration.md#sector_cf)
- [industry](../configuration.md#industry_cf)
- [costs](../configuration.md#costs_cf)
- [clustering](../configuration.md#clustering_cf)
- [adjustments](../configuration.md#adjustments_cf)
- [solving](../configuration.md#solving_cf)
- [data](../configuration.md#data_cf)
- [overpass_api](../configuration.md#overpass_api_cf)
- [secrets](../configuration.md#secrets_cf)
- [plotting](../configuration.md#plotting_cf)

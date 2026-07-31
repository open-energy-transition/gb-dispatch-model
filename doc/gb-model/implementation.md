<!-- SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model> -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Technical Implementation {#implementation}

This section provides technical details about GB-specific implementation features.

## FES Powerplants Data {#powerplants-data}

### Overview

The GB model uses powerplant capacity data from the Future Energy Scenarios (FES) workbook, enriched with technical and cost parameters to create a complete dataset ready for PyPSA network composition. This data replaces the default PyPSA-Eur powerplants dataset with GB-specific capacity projections.

### Data Pipeline

The powerplants data flows through two main stages:

1. **Capacity Aggregation** (`create_powerplants_table.py`)

   - Processes FES workbook data (GB regions)
   - Processes European supply data (neighboring countries)
   - Maps technology names to PyPSA carriers
   - Aggregates capacities by bus, year, carrier, and set

2. **Cost Enrichment** (`create_powerplants_table.py`)

   - Joins technology cost data (efficiency, VOM, fuel costs, etc.)
   - Calculates marginal costs
   - Fills missing values with sensible defaults
   - Creates unique generator indices

### Output Format

The resulting `fes_powerplants.csv` contains complete generator data:

**Core Attributes**:

- `bus` - Network bus ID (string)
- `year` - Planning year (integer)
- `carrier` - Technology type (CCGT, nuclear, onwind, etc.)
- `set` - Generator classification (PP, CHP, Store)
- `p_nom` - Nominal capacity in MW (float)
- `build_year` - Year of installation (integer)

**Technical Parameters**:

- `efficiency` - Energy conversion efficiency (0-1)
- `lifetime` - Asset lifetime in years (float)

**Economic Parameters**:

- `VOM` - Variable O&M cost (GBP/MWh)
- `FOM` - Fixed O&M cost (€/MW/year)
- `capital_cost` - Investment cost (€/MW)
- `fuel` - Fuel cost (GBP/MWh_thermal)
- `marginal_cost` - Total variable cost (GBP/MWh_el)

**Index Format**: `"{bus} {carrier}-{year}-{counter}"`

Example: `"GB0 CCGT-2030-0"`, `"GB0 CCGT-2030-1"`

### Marginal Cost Calculation

Marginal cost combines variable O&M and fuel costs:

```
marginal_cost = VOM + fuel / efficiency
```

Where:

- `VOM` - Variable operations and maintenance (GBP/MWh_el)
- `fuel` - Fuel cost (GBP/MWh_thermal)
- `efficiency` - Conversion efficiency (MWh_el / MWh_thermal)

Example for CCGT with efficiency=0.55, VOM=2.5, fuel=25.0:

```
marginal_cost = 2.5 + 25.0/0.55 = 47.95 GBP/MWh
```

### Default Values

When cost data is unavailable for specific carriers, defaults are applied:

- `efficiency`: 1 (100% conversion efficiency)
- `capital_cost`: 0.0 €/MW
- `lifetime`: 25.0 years
- `marginal_cost`: 0.0 GBP/MWh

These defaults prevent missing data from blocking network composition while logging warnings for review.

### Integration with compose_network

The `compose_network` rule loads the enriched powerplants data directly:

```
ppl = pd.read_csv(powerplants_path, index_col=0, dtype={"bus": "str"})
```

This data is then used by:

- `attach_conventional_generators()` - Adds fossil/nuclear generators
- `attach_wind_and_solar()` - Adds renewable generators
- `attach_hydro()` - Adds hydro and storage units
- `attach_chp_constraints()` - Applies CHP heat demand constraints

No additional preprocessing is required; all necessary attributes are present in the CSV.

### Configuration

The `create_powerplants_table` rule requires:

**Inputs**:

- `gsp_data` - GB regional capacity data from FES workbook
- `eur_data` - European national capacity data
- `tech_costs` - Technology cost assumptions

**Parameters**:

- `gb_config` - GB technology mapping configuration
- `eur_config` - European technology mapping configuration
- `default_set` - Default generator classification (typically "PP")

**Output**:

- `fes_powerplants.csv` - Complete powerplants dataset

### File Location

```
scripts/gb_model/
└── create_powerplants_table.py   # Data processing and enrichment

rules/
└── gb-model.smk                  # Snakemake rule definitions

results/{run}/resources/gb-model/
└── fes_powerplants.csv           # Generated output
```

..
  SPDX-FileCopyrightText: Contributors to gb-open-market-model <https://github.com/open-energy-transition/gb-open-market-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _implementation:

##########################################
Technical Implementation
##########################################

This section provides technical details about GB-specific implementation features.

.. _powerplants-data:

===============================================
FES Powerplants Data
===============================================

Overview
========

The GB model uses powerplant capacity data from the Future Energy Scenarios (FES) workbook, enriched with technical and cost parameters to create a complete dataset ready for PyPSA network composition. This data replaces the default PyPSA-Eur powerplants dataset with GB-specific capacity projections.

Data Pipeline
=============

The powerplants data flows through two main stages:

1. **Capacity Aggregation** (``create_powerplants_table.py``)

   - Processes FES workbook data (GB regions)
   - Processes European supply data (neighboring countries)
   - Maps technology names to PyPSA carriers
   - Aggregates capacities by bus, year, carrier, and set

2. **Cost Enrichment** (``create_powerplants_table.py``)

   - Joins technology cost data (efficiency, VOM, fuel costs, etc.)
   - Calculates marginal costs
   - Fills missing values with sensible defaults
   - Creates unique generator indices

Output Format
=============

The resulting ``fes_powerplants.csv`` contains complete generator data:

**Core Attributes**:

- ``bus`` - Network bus ID (string)
- ``year`` - Planning year (integer)
- ``carrier`` - Technology type (CCGT, nuclear, onwind, etc.)
- ``set`` - Generator classification (PP, CHP, Store)
- ``p_nom`` - Nominal capacity in MW (float)
- ``build_year`` - Year of installation (integer)

**Technical Parameters**:

- ``efficiency`` - Energy conversion efficiency (0-1)
- ``lifetime`` - Asset lifetime in years (float)

**Economic Parameters**:

- ``VOM`` - Variable O&M cost (€/MWh)
- ``FOM`` - Fixed O&M cost (€/MW/year)
- ``capital_cost`` - Investment cost (€/MW)
- ``fuel`` - Fuel cost (€/MWh_thermal)
- ``marginal_cost`` - Total variable cost (€/MWh_el)

**Index Format**: ``"{bus} {carrier}-{year}-{counter}"``

Example: ``"GB0 CCGT-2030-0"``, ``"GB0 CCGT-2030-1"``

Marginal Cost Calculation
==========================

Marginal cost combines variable O&M and fuel costs::

    marginal_cost = VOM + fuel / efficiency

Where:

- ``VOM`` - Variable operations and maintenance (€/MWh_el)
- ``fuel`` - Fuel cost (€/MWh_thermal)
- ``efficiency`` - Conversion efficiency (MWh_el / MWh_thermal)

Example for CCGT with efficiency=0.55, VOM=2.5, fuel=25.0::

    marginal_cost = 2.5 + 25.0/0.55 = 47.95 €/MWh

Default Values
==============

When cost data is unavailable for specific carriers, defaults are applied:

- ``efficiency``: 0.4 (40% conversion efficiency)
- ``capital_cost``: 0.0 €/MW
- ``lifetime``: 25.0 years
- ``marginal_cost``: 0.0 €/MWh

These defaults prevent missing data from blocking network composition while logging warnings for review.

Integration with compose_network
=================================

The ``compose_network`` rule loads the enriched powerplants data directly::

    ppl = pd.read_csv(powerplants_path, index_col=0, dtype={"bus": "str"})

This data is then used by:

- ``attach_conventional_generators()`` - Adds fossil/nuclear generators
- ``attach_wind_and_solar()`` - Adds renewable generators
- ``attach_hydro()`` - Adds hydro and storage units
- ``attach_chp_constraints()`` - Applies CHP heat demand constraints

No additional preprocessing is required; all necessary attributes are present in the CSV.

Configuration
=============

The ``create_powerplants_table`` rule requires:

**Inputs**:

- ``gsp_data`` - GB regional capacity data from FES workbook
- ``eur_data`` - European national capacity data
- ``tech_costs`` - Technology cost assumptions

**Parameters**:

- ``gb_config`` - GB technology mapping configuration
- ``eur_config`` - European technology mapping configuration
- ``default_set`` - Default generator classification (typically "PP")

**Output**:

- ``fes_powerplants.csv`` - Complete powerplants dataset

File Location
=============

::

    scripts/gb_model/
    └── create_powerplants_table.py   # Data processing and enrichment

    rules/
    └── gb-model.smk                  # Snakemake rule definitions

    results/{run}/resources/gb-model/
    └── fes_powerplants.csv           # Generated output

.. _chp-implementation:

===============================================
Simplified CHP Implementation
===============================================

Overview
========

Combined Heat and Power (CHP) plants generate both electricity and heat simultaneously, typically for district heating networks or industrial processes. In electricity markets, CHPs must often maintain minimum electricity generation to meet their heat supply obligations, even when electricity prices are low. This creates operational constraints that significantly affect electricity market dispatch and pricing.

The GB model implements a simplified CHP representation suitable for electricity-only market modeling. Rather than explicitly modeling heat networks and sector coupling (as in PyPSA-Eur's full sector network), CHPs are represented as electricity generators with time-varying minimum generation constraints (``p_min_pu``) derived from heat demand profiles. This approach captures the key market impact—CHPs' reduced flexibility due to heat obligations—while avoiding the complexity of multi-sector optimization.

**Key Simplification**: Instead of modeling CHPs as multi-output Links with explicit heat buses and loads, they are modeled as Generators with minimum operation levels that implicitly represent heat demand obligations. This maintains electricity market realism while keeping the model computationally efficient.

Implementation Components
=========================

CHP Identification
------------------

CHPs are identified from powerplants data using the ``Set`` column::

    chp_plants = powerplants[powerplants["set"] == "CHP"]

**Data Source**: The ``Set`` column in powerplants CSV distinguishes:

- ``PP`` - Regular power plants
- ``CHP`` - Combined Heat and Power plants
- ``Store`` - Storage facilities (hydro)

Heat Demand Processing
----------------------

Heat demand is loaded from the hourly heat demand NetCDF file.

**Data Structure**:

- **Dimensions**: ``snapshots`` (time), ``node`` (buses)
- **Variables**:

  - ``residential water``
  - ``residential space``
  - ``services water``
  - ``services space``

**Processing Steps**:

1. Load heat demand dataset
2. Sum all heat types to get total heat demand per bus
3. Normalize by peak demand to create 0-1 profile

Minimum Operation Calculation
------------------------------

The minimum power output (``p_min_pu``) is calculated using:

.. math::

    p\_min\_pu = \frac{total\_heat\_demand / peak\_heat\_demand}{heat\_to\_power\_ratio}

**Key Parameters**:

- **heat_to_power_ratio** (default: 1.5)

  - Represents the back-pressure coefficient (:math:`c_b`)
  - Typical values: 1.0-2.0 for gas CHPs
  - Higher values = more heat per unit electricity
  - Determines how much CHP must run to meet heat demand

- **min_operation_level** (default: 0.3)

  - Minimum operation when CHP is running (30% of capacity)
  - Technical constraint - CHPs cannot operate below this level
  - Applied as an upper bound on p_min_pu

**Special Handling**:

- When heat demand < 10% of peak, CHPs can shut down completely (p_min_pu = 0)
- This prevents forcing CHPs to run during very low heat demand periods

Application to Generators
--------------------------

After conventional generators are attached to the network:

1. Identify CHP generators by matching:

   - Bus location with CHP powerplants
   - Carrier type (typically CCGT)

2. Apply time series constraint (vectorized)::

    # For all CHP generators at once
    n.generators_t.p_min_pu[chp_gen_indices] = p_min_pu_for_gens

3. Result: CHPs must run at least at p_min_pu level when operational

Configuration
=============

Add to ``config/config.gb.default.yaml``::

    chp:
      enable: true                  # Enable CHP constraints
      heat_to_power_ratio: 1.5     # Heat/power ratio (c_b)
      min_operation_level: 0.3     # Min operation (30%)

Configuration Options
---------------------

- **enable**: Set to ``false`` to disable CHP constraints (treat as regular generators)
- **heat_to_power_ratio**: Adjust based on CHP technology

  - Gas turbine CHPs: 1.0-1.5
  - Steam turbine CHPs: 1.5-2.5
  - Reciprocating engines: 0.8-1.2

- **min_operation_level**: Technical minimum (typically 0.2-0.4)

File Structure
==============

::

    scripts/gb_model/
    ├── chp_utils.py           # CHP utility functions
    └── compose_network.py     # Integration point

    config/
    └── config.gb.default.yaml # CHP configuration

Methodology
===========

Heat-to-Power Ratio
-------------------

The ``heat_to_power_ratio`` parameter represents the back-pressure coefficient :math:`c_b = \eta_{heat} / \eta_{el}`, where :math:`\eta_{el}` and :math:`\eta_{heat}` are the electrical and thermal efficiencies. For typical gas CHPs with :math:`\eta_{el}` ≈ 0.4 and :math:`\eta_{heat}` ≈ 0.5, :math:`c_b` ≈ 1.25-1.5.

Minimum Operation Logic
-----------------------

For each time step, the minimum generation constraint is:

.. code-block:: python

    if heat_profile[bus, t] < 0.1:  # < 10% of peak heat demand
        p_min_pu[bus, t] = 0.0      # CHP can shut down
    else:
        p_min_pu[bus, t] = min(
            heat_profile[bus, t] / heat_to_power_ratio,
            min_operation_level  # Technical minimum (e.g., 0.3)
        )

This ensures:

1. CHPs follow heat demand when significant
2. CHPs respect technical minimum operation level
3. CHPs can shut down during very low heat demand

Limitations
===========

This simplified approach does not model:

- Heat network topology and transmission
- Thermal storage for decoupling heat and power
- Variable heat-to-power operating modes
- Heat provision for ancillary services

For applications requiring detailed heat sector modeling, use PyPSA-Eur's full sector-coupling approach with explicit heat buses and Links.

.. _boundary-capability-scaling:

===============================================
Transmission Boundary Capability Scaling
===============================================

Overview
========

The GB model scales transmission line capacities (``s_nom``) to match the boundary capabilities defined in NESO's Electricity Ten Year Statement (ETYS) report.
This ensures that inter-regional power transfer limits in the model align with real-world transmission constraints.

PyPSA-Eur's base network, derived from OpenStreetMap (OSM) transmission infrastructure, provides detailed line-by-line representation of the GB transmission system.
However, these individual line ratings may not aggregate to match the official boundary transfer capabilities that result from detailed power flow modelling.
The scaling process adjusts line capacities to match baseline capabilities defined in the ETYS while respecting the physical network topology.

Problem Statement
=================

**ETYS Boundaries**: NESO defines transmission boundaries—interfaces between major regions of the GB network—with specific power transfer capabilities in MW.
These represent operational limits for managing power flows across regions.

**PyPSA-Eur Lines**: The base network contains many individual transmission lines crossing each boundary, each with its own ``s_nom`` (nominal capacity) derived from OSM voltage data and standard assumptions.

**Mismatch**: The sum of PyPSA line capacities across a boundary rarely matches the ETYS boundary capability exactly, due to:

- Simplified assumptions in OSM data processing
- Network topology differences (actual vs. modeled)
- Operational vs. nominal rating differences
- Missing or incomplete OSM infrastructure data

**Solution**: Scale individual line capacities using ``s_max_pu`` multipliers to achieve target boundary capabilities while maintaining relative line importance.

Methodology
===========

Boundary-to-Line Mapping
-------------------------

ETYS boundaries are mapped to PyPSA network buses in the configuration::

    region_operations:
      etys_boundaries:
        B6:                    # Boundary name from ETYS report
        - bus0: 23             # PyPSA bus IDs
          bus1: 21
        B7a:
        - bus0: 20
          bus1: 15
        - bus0: 20             # Multiple line groups can compose
          bus1: 16             # a single ETYS boundary
        - bus0: 16
          bus1: 21

Each boundary is defined by one or more bus pairs.
All transmission lines connecting those bus pairs contribute to the boundary's total capacity.

Capacity Calculation
--------------------

For each ETYS boundary, the baseline PyPSA capacity is calculated by summing all contributing lines::

    boundary_capacity = sum(line.s_nom for line in boundary_lines)

This is compared against the ETYS target capability extracted from the ETYS PDF report (see ``extract_etys_boundary_capabilities.py``).

Optimization Problem
--------------------

The scaling process finds ``s_max_pu`` values (capacity multipliers) for each line by solving an optimization problem:

**Objective**: Minimize total relative error across all boundaries

.. math::

    \min_{x_i} \sqrt{\sum_{b \in \text{boundaries}} \left( \frac{\sum_{i \in b} s_{nom,i} \cdot x_i - C_b}{C_b} \right)^2}

Where:

- :math:`x_i` = ``s_max_pu`` for line :math:`i`
- :math:`s_{nom,i}` = nominal capacity of line :math:`i`
- :math:`C_b` = ETYS target capability for boundary :math:`b`
- :math:`i \in b` = lines that contribute to boundary :math:`b`

**Constraints**: Each line's ``s_max_pu`` is bounded by all boundaries it belongs to

.. math::

    \min_{b: i \in b}(\text{scaling}_b) \leq x_i \leq \max_{b: i \in b}(\text{scaling}_b)

Where :math:`\text{scaling}_b = C_b / \sum_{i \in b} s_{nom,i}` is the simple scaling factor for boundary :math:`b`.

This ensures:

1. Lines belonging to multiple boundaries satisfy constraints from all
2. No line is scaled beyond what any of its boundaries require
3. Relative line importance within boundaries is preserved

**Method**: Scipy's ``minimize``

**Initial Guess**: Midpoint of min/max range for each line

Special Cases
-------------

**Line Pruning**: Some lines are manually removed (set ``s_max_pu = 0``) because they represent:

- Spurious connections from network building errors
- Stubs that should have been removed earlier (but we didn't want to accidentally remove stubs that represent real connections)

These are defined in configuration::

    region_operations:
      prune_lines:
      - bus0: 10
        bus1: 11
      - bus0: 26
        bus1: 29

File Structure
==============

::

    scripts/gb_model/
    └── scale_boundary_capabilities.py    # Optimization implementation

    rules/
    └── gb-model.smk                       # Snakemake rule definition

    config/
    └── config.gb.default.yaml             # Boundary mappings

    results/{run}/resources/gb-model/
    ├── line_s_max_pu.csv                  # Output scaling factors
    └── line_s_nom_compare.html            # Validation visualization

Limitations
===========

- **Topology Constraints**: Some boundary definitions may not perfectly match network topology, requiring approximations (noted in config comments)
- **Optimization Convergence**: Highly constrained problems (many overlapping boundaries) may not achieve perfect matches for all boundaries simultaneously
- **Static Capacities**: Scaling factors are constant; seasonal or dynamic ratings are not modeled
- **Bus Numbering**: Requires manual mapping between ETYS geographical boundaries and PyPSA bus IDs
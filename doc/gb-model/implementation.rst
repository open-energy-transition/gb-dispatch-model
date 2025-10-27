..
  SPDX-FileCopyrightText: Contributors to gb-open-market-model <https://github.com/open-energy-transition/gb-open-market-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _implementation:

##########################################
Technical Implementation
##########################################

This section provides technical details about GB-specific implementation features.

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

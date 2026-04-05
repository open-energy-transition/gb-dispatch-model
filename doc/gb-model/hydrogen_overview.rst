..
  SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: gb-dispatch-model contributors

  SPDX-License-Identifier: CC-BY-4.0

.. _hydrogen-overview:

##########################################
Overview & Components
##########################################

This page introduces the hydrogen subsystem, its data sources, and the components that model hydrogen in the electricity network.

Overview
========

The hydrogen subsystem in gb-dispatch-model represents the interaction between electricity and hydrogen systems in Great Britain and connected European countries.
Unlike other loads in the model which represent electrical demand from various sectors, hydrogen is modelled as a distinct energy carrier with its own bus, storage, and conversion technologies.

The hydrogen system comprises four main components:

- **Hydrogen demand**: Net hydrogen consumption for non-power system uses (e.g., industrial processes, hydrogen boilers, transport)
- **Grid-connected electrolysis**: Conversion of electricity to hydrogen via electrolysers connected to the electricity grid
- **Hydrogen storage**: Underground or surface storage for temporal balancing of hydrogen supply and demand
- **Hydrogen-to-electricity conversion**: Fuel cells and hydrogen turbines that can generate electricity from stored hydrogen

Additionally, the model accounts for **non-networked electrolysis**—off-grid hydrogen production that appears only as additional electricity demand and does not interact with the modeled hydrogen system.

This modelling approach allows hydrogen to act as both a flexible load on the electricity system (through electrolysis) and a potential backup generation source (through fuel cells and turbines).

.. graphviz::

   digraph {
      rankdir=LR;
      node [shape=box, style=filled];

      // Main components
      AC_bus [label="AC Bus", fillcolor="#B3D9FF", shape=ellipse, width=2, height=1.5, fixedsize=true];
      H2_bus [label="H2 Bus", fillcolor="#CCFFFF", shape=ellipse, width=2, height=1.5, fixedsize=true];
      
      // Hydrogen system components
      electrolyser [label="Grid\nElectrolysis", fillcolor="#90EE90"];
      fuel_cell [label="Fuel Cell", fillcolor="#FFB6C1"];
      h2_turbine [label="H2 Turbine", fillcolor="#FFB6C1"];
      h2_store [label="H2 Storage", fillcolor="#FFFACD"];
      h2_demand [label="H2 Demand", fillcolor="#FFDAB9"];
      non_grid_elec [label="Non-networked\nElectrolysis", fillcolor="#DDA0DD"];
      
      // Connections
      AC_bus -> electrolyser -> H2_bus [label="Grid-connected"];
      H2_bus -> fuel_cell -> AC_bus;
      H2_bus -> h2_turbine -> AC_bus;
      H2_bus -> h2_store [dir=both];
      H2_bus -> h2_demand;
      AC_bus -> non_grid_elec [label="Off-grid demand"];
   }


.. _hydrogen-data-sources:

Data Sources
============

**Great Britain Data**
----------------------

All GB hydrogen system data is derived from the Future Energy Scenarios (FES) workbook, primarily from:

- **WS1 (Whole System)**: Annual hydrogen demand, supply, and storage capacity data filtered by fuel type, scenario pathway, and year
- **BB1 (Building Block Data)**: Regional grid-connected electrolysis capacities assigned to Grid Supply Points (GSPs)

The FES 2024 provides annual data for multiple scenarios (Electric Engagement, Hydrogen Evolution, Holistic Transition, Counterfactual, Five Year Forecast) covering:

- **Hydrogen production**: Grid-connected and non-networked electrolysis capacities by region and year (in TWh/year and MW)
- **Hydrogen consumption**: Demand across different sectors (transport, heating, industry, power generation) in TWh/year
- **Hydrogen supply**: Non-electrolysis hydrogen sources (e.g., steam methane reforming with CCUS, biomass gasification, imports) in TWh/year
- **Storage capacity**: Required hydrogen storage infrastructure by year (in TWh)

All FES energy data is provided in TWh and converted to MWh for PyPSA (multiply by 1,000,000).

**European Data**
-----------------

For European countries outside GB, we synthesise hydrogen data by scaling GB patterns according to:

1. **TYNDP (Ten-Year Network Development Plan)**: Future hydrogen demand projections for European countries
2. **Hydrogen Europe annual report**: Current European hydrogen consumption by country

   - Data provided in tonnes per year (T/Y)
   - Converted to MWh using lower heating value: 1 tonne H₂ = 33.33 MWh
   - Formula: ``MWh = tonnes × 33.33 × 1000`` (the 1000 factor converts from tonnes to kg, with LHV ≈ 33.33 kWh/kg)

This synthesis approach maintains relative patterns from GB FES data (ratios between demand, storage, and electrolysis) while matching European-specific total demand projections.


.. _hydrogen-components:

System Components
=================

.. _hydrogen-demand:

Hydrogen Demand
---------------

**PyPSA Component**: ``Load`` attached to the hydrogen bus

Net hydrogen demand represents the total hydrogen consumption minus total hydrogen supply from all sources except grid electrolysis. Grid electrolysis is modelled separately and endogenously, so it does not appear in the supply data.

The hydrogen demand includes:

- Industrial processes (ammonia, steel, refining, chemicals)
- Residential and commercial building heating (hydrogen boilers)
- Road transport (hydrogen fuel cell vehicles)
- Rail transport (hydrogen trains)
- Shipping (hydrogen-fueled vessels)
- Aviation (hydrogen aircraft)
- Power generation (hydrogen fuel cells and turbines for electricity)

The demand is calculated as:

.. math::

   \text{Net H}_2 \text{ Demand} = \sum \text{All Demand} - \sum \text{All Unmodelled Supply}

where supply sources might include hydrogen from steam methane reforming, biomass gasification, or imports, but exclude grid electrolysis which is optimized by the model.

**Data Processing**:

The :ref:`create_hydrogen_data_tables` rule extracts demand and supply data from FES WS1 sheet, filters by fuel type ("hydrogen"), scenario, and year range, then aggregates to create annual net demand in MWh.

**Temporal Profile**:

Hydrogen demand is assumed to be flat across the year.
While in reality, hydrogen demand would vary (e.g., heating demand in winter), the FES does not provide sufficient granularity to estimate this profile.
This assumption means that any temporal mismatch between hydrogen production and consumption must be managed by hydrogen storage.

**Regional Distribution**:

For GB, national hydrogen demand data is disaggregated to regions using the :ref:`regional distribution process <regional_hydrogen_distribution>`.
Regional distribution is based on the spatial pattern of hydrogen electrolysis capacity from the FES BB1 sheet.
For European countries, demand is assigned at the country level based on TYNDP and historical consumption data.


.. _hydrogen-electrolysis:

Hydrogen Electrolysis
---------------------

Electrolysis converts electricity to hydrogen using water:

.. math::

   2H_2O + \text{electricity} \rightarrow 2H_2 + O_2

Grid-connected Electrolysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyPSA Component**: ``Link`` from AC bus to hydrogen bus

Grid-connected electrolysers are directly connected to the electricity network and can be optimally dispatched in the model.
This provides flexibility to produce hydrogen when electricity is cheap or abundant (e.g., during high renewable generation periods).

**Capacity**:

Electrolyser capacities are extracted from the FES Building Block Data (BB1) for each GB region and year.
Where regional assignments are missing, capacities are distributed proportionally based on the existing regional distribution of electrolysis capacity.

**Efficiency**:

Electrolyser efficiency is derived from `PyPSA technology-data <https://github.com/PyPSA/technology-data>`_ (2035 cost year, around 63.74%), representing the energy conversion from electricity to hydrogen (lower heating value basis).

**Operation**:

Grid-connected electrolysers are fully flexible with no minimum load constraints, meaning they can be dispatched at any level from 0% to 100% of capacity in any timestep.

.. _hydrogen-electrolysis-non-grid:

Non-networked Electrolysis
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyPSA Component**: Additional electricity ``Load`` on the AC bus

Non-networked (or "off-grid") electrolysis represents hydrogen production from dedicated renewable generators that are not connected to the main electricity grid.
These could be remote wind farms or solar installations directly coupled to electrolysers.

From the electricity system perspective, this appears as an inflexible electricity demand.
The hydrogen produced does not appear in the model's hydrogen system but is accounted for in the supply side of the net hydrogen demand calculation.

**Calculation**:

Non-networked electrolysis electricity demand is calculated by dividing the hydrogen production capacity by an assumed electrolysis efficiency (configurable, default 70%):

.. math::

   \text{Electricity Demand} = \frac{\text{H}_2 \text{ Production}}{\text{Efficiency}}

For GB, this national electricity demand is :ref:`disaggregated to regions <regional_hydrogen_distribution>` based on the spatial pattern of hydrogen electrolysis capacity, then added to the baseline electricity load in each region.


.. _hydrogen-storage:

Hydrogen Storage
----------------

**PyPSA Component**: ``Store`` attached to the hydrogen bus

Hydrogen storage provides temporal flexibility to balance hydrogen production and demand.
Storage can be in underground caverns (salt caverns, depleted gas fields), surface tanks, or other infrastructure.

**Capacity**:

Storage capacity (energy capacity in MWh) is derived from FES data and increases over time as the hydrogen economy develops.
For GB, national storage capacity is :ref:`disaggregated to regions <regional_hydrogen_distribution>` based on the spatial pattern of hydrogen electrolysis capacity.
The model assumes that storage can be charged and discharged without capacity constraints beyond the bus energy balance (i.e., available hydrogen must be produced or drawn from storage).

**Dispatch Constraints**:

The storage level at the end of each modelled year is set to equal to the level at the start of that year.
This cyclic constraint prevents unrealistic depletion of storage within each year, but may create artificial constraints on storage use in the final timesteps of each year (see :doc:`faq`).

.. _hydrogen-conversion:

Hydrogen to Electricity Conversion
-----------------------------------

Conversion of hydrogen back to electricity provides dispatchable backup generation that can support the grid during periods of low renewable generation or high demand.

.. _hydrogen-fuel-cells:

Fuel Cells
^^^^^^^^^^

**PyPSA Component**: ``Link`` from hydrogen bus to AC bus

Fuel cells convert hydrogen to electricity through an electrochemical reaction without combustion:

.. math::

   2H_2 + O_2 \rightarrow 2H_2O + \text{electricity}

**Characteristics**:

- High electrical efficiency (derived from `PyPSA technology-data <https://github.com/PyPSA/technology-data>`_ 2035 cost year, typically 40-60%)
- Can provide fast ramping and flexible operation
- No direct CO₂ emissions
- Can be distributed (small scale) or centralized

**Capacity**:

.. note::
   FES 2024 BB1 contains **zero fuel cell capacity for all years**.
   Fuel cells are not included in the FES hydrogen infrastructure projections.


.. _hydrogen-turbines:

Hydrogen Turbines
^^^^^^^^^^^^^^^^^

**PyPSA Component**: ``Link`` from hydrogen bus to AC bus

Hydrogen gas turbines are modified natural gas turbines that can burn pure hydrogen or hydrogen blends.
They combust hydrogen to drive a turbine and generate electricity:

.. math::

   2H_2 + O_2 \rightarrow 2H_2O + \text{heat} \rightarrow \text{electricity}

**Characteristics**:

- Lower electrical efficiency than fuel cells (derived from `PyPSA technology-data <https://github.com/PyPSA/technology-data>`_ 2035 cost year, assumed 50% in this workflow)
- Can provide larger-scale, centralized generation
- Fast start-up times for grid balancing
- No direct CO₂ emissions

**Capacity**:

Like fuel cells, hydrogen turbine capacity is based on FES projections.

..
  SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: gb-dispatch-model contributors

  SPDX-License-Identifier: CC-BY-4.0

.. _hydrogen:

##########################################
Hydrogen System
##########################################

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

   \text{Net H}_2 \text{ Demand} = \sum \text{All Demand} - \sum \text{All Supply}

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


.. _hydrogen-workflow:

Data Processing Workflow
=========================

The hydrogen system is built through a multi-stage data processing pipeline:

.. graphviz::

   digraph {
      rankdir=TB;
      node [shape=box, style="filled,rounded"];
      
      // Input data
      fes_ws1 [label="FES WS1\n(Whole System)", fillcolor="#E6F3FF", shape=folder];
      fes_bb1 [label="FES BB1\n(Building Blocks)", fillcolor="#E6F3FF", shape=folder];
      tyndp [label="TYNDP Data", fillcolor="#FFE6E6", shape=folder];
      eur_today [label="EU H2 Today", fillcolor="#FFE6E6", shape=folder];
      
      // GB national processing
      h2_tables [label="create_hydrogen_data_tables\n• National H2 demand\n• National storage\n• National off-grid elec", fillcolor="#D4EDDA"];
      
      // GB regional processing
      grid_elec [label="create_grid_electrolysis_table\n• Regional grid electrolysis", fillcolor="#D4EDDA"];
      regional_dist [label="synthesise_gb_regional_data\n• Distributes demand\n• Distributes storage\n• Distributes off-grid elec", fillcolor="#D4EDDA"];
      
      // EUR processing
      eur_demand [label="add_eur_H2_demand\n• Combine GB + EUR", fillcolor="#FFF3CD"];
      eur_synth [label="synthesise_eur_H2_data\n• Scale storage/elec", fillcolor="#FFF3CD"];
      
      // Outputs (intermediate)
      out_demand [label="regional_H2_demand\n_annual_inc_eur.csv", fillcolor="#D1ECF1", shape=note];
      out_elec [label="regional_grid_electrolysis\n_capacities_inc_eur.csv", fillcolor="#D1ECF1", shape=note];
      out_storage [label="regional_H2_storage\n_capacity_inc_eur.csv", fillcolor="#D1ECF1", shape=note];
      out_offgrid [label="regional_non_networked\n_electrolysis_..._inc_eur.csv", fillcolor="#D1ECF1", shape=note];
      
      // Cost assignment
      assign_costs [label="assign_costs\n• Add tech-data costs", fillcolor="#FFC7CE"];
      
      // Final outputs with costs
      out_elec_costs [label="..._grid_electrolysis\n_capacities_inc_eur\n_inc_tech_data.csv", fillcolor="#C9E4DE", shape=note];
      out_storage_costs [label="..._H2_storage\n_capacity_inc_eur\n_inc_tech_data.csv", fillcolor="#C9E4DE", shape=note];
      
      // Final integration
      network [label="PyPSA Network\nComposition", fillcolor="#E7D4F0"];
      
      // Connections
      fes_ws1 -> h2_tables;
      fes_bb1 -> grid_elec;
      fes_bb1 -> regional_dist [label="reference:\nH2 electrolysis", fontsize=10];
      h2_tables -> regional_dist [label="national data", fontsize=10];
      
      regional_dist -> eur_demand [label="demand", fontsize=10];
      tyndp -> eur_demand;
      eur_today -> eur_demand;
      
      eur_demand -> eur_synth;
      grid_elec -> eur_synth;
      regional_dist -> eur_synth [label="storage,\noff-grid", fontsize=10];
      
      eur_demand -> out_demand;
      eur_synth -> out_elec;
      eur_synth -> out_storage;
      eur_synth -> out_offgrid;
      
      out_elec -> assign_costs;
      out_storage -> assign_costs;
      
      assign_costs -> out_elec_costs;
      assign_costs -> out_storage_costs;
      
      out_demand -> network;
      out_elec_costs -> network;
      out_storage_costs -> network;
      out_offgrid -> network;
   }

**Workflow Stages**:

1. **GB Data Extraction** (``create_hydrogen_data_tables``):
   
   - Reads FES WS1 whole system data
   - Filters for hydrogen fuel type and selected scenario
   - Applies user-defined data selection filters
   - Calculates net demand, storage needs, and off-grid electricity demand
   - Outputs: GB-level annual data

2. **Regional Distribution**:
   
   - ``create_grid_electrolysis_table``: Processes FES BB1 building block data for grid-connected electrolysis, assigns capacities to GB regions (Grid Supply Points), redistributes unmapped capacities proportionally
   - ``synthesise_gb_regional_data``: Distributes national H2 demand, storage, and non-networked electrolysis data to regions based on hydrogen electrolysis capacity patterns
   - Outputs: Regional GB data by GSP for all hydrogen components

3. **European Integration** (``add_eur_H2_demand``):
   
   - Combines TYNDP projections with current consumption
   - Interpolates/extrapolates for all model years
   - Merges with GB regional data
   - Outputs: Combined GB+EUR demand

4. **European Synthesis** (``synthesise_eur_H2_data``):
   
   - Normalizes GB data by GB demand
   - Scales European data by European country demands
   - Applies ratios to storage and electrolysis datasets
   - Outputs: Complete regional datasets for all components

5. **Cost Assignment** (``assign_costs``):
   
   - Enriches H2 storage and grid electrolysis capacity data with technology costs
   - Applies costs from `PyPSA technology-data <https://github.com/PyPSA/technology-data>`_ (2035 cost year)
   - Includes capital costs, efficiency, lifetime, and other technical parameters
   - Appends ``_inc_tech_data`` suffix to filenames
   - Outputs: Cost-enriched capacity files ready for network integration

6. **Network Integration**:
   
   - Network composition rules read CSV outputs
   - Creates PyPSA buses, loads, links, and stores
   - Assigns temporal profiles and technical parameters
   - Results: Complete hydrogen subsystem in PyPSA network

See :doc:`implementation` for details on how these data are integrated into the network model.


.. _hydrogen-implementation:

Implementation Details
======================

The hydrogen system is implemented through several Snakemake rules defined in ``rules/gb-model/hydrogen.smk`` that process FES data and prepare it for integration with the PyPSA network.

.. _create_hydrogen_data_tables:

Data Table Creation
-------------------

**Rule**: ``create_hydrogen_data_tables``

**Location**: `rules/gb-model/hydrogen.smk <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/rules/gb-model/hydrogen.smk#L10-L28>`_

**Script**: `scripts/gb_model/hydrogen/create_hydrogen_data_tables.py <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/scripts/gb_model/hydrogen/create_hydrogen_data_tables.py>`_

This rule processes the FES WS1 sheet to extract three key hydrogen datasets:

1. **Net hydrogen demand** (``H2_demand_annual.csv``): Total consumption minus all supply except grid-electrolysis supply
2. **Non-networked electrolysis electricity demand** (``non_networked_electrolysis_demand_annual.csv``): Off-grid electrolysis converted to electricity demand
3. **Hydrogen storage capacity** (``H2_storage_capacity.csv``): Required storage infrastructure

**Process**:

1. Filter WS1 data for hydrogen fuel type, selected FES scenario, and model years
2. Apply user-defined data selection filters to categorise hydrogen data (demand, supply, storage, non-networked supply)
3. Aggregate filtered data by year
4. Calculate net demand and convert non-networked electrolysis to electricity demand using efficiency factor
5. Convert from TWh (FES units) to MWh (model units)
6. Export to CSV files

**Configuration**:

The data selection filters define which rows from the FES WS1 sheet are included in each category.
Filters are dictionaries matching FES column names to values (case-insensitive).

From ``config/config.gb.2024.yaml``:

.. literalinclude:: ../../config/config.gb.2024.yaml
   :language: yaml
   :lines: 906-938


.. _create_grid_electrolysis:

Grid Electrolysis Capacity
---------------------------

**Rule**: ``create_grid_electrolysis_table``

**Location**: `rules/gb-model/hydrogen.smk <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/rules/gb-model/hydrogen.smk#L31-L43>`_

**Script**: `scripts/gb_model/hydrogen/create_grid_electrolysis_table.py <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/scripts/gb_model/hydrogen/create_grid_electrolysis_table.py>`_

This rule extracts regional grid-connected electrolyser capacities from the FES Building Block Data.

**Process**:

1. Filter regional GB data for "hydrogen electrolysis" technology
2. Sum capacities by bus (region) and year
3. Calculate regional distribution of mapped capacities
4. Identify unmapped capacities (no regional assignment)
5. Redistribute unmapped capacities according to existing regional distribution
6. Export combined capacities (mapped + redistributed) to CSV

This approach ensures that all FES electrolysis capacity is allocated to regions, even when the source data lacks complete geographic detail.


.. _regional_hydrogen_distribution:

Regional Distribution of National Data
---------------------------------------

**Rule**: ``synthesise_gb_regional_data``

**Location**: `rules/gb-model/demand_and_dsr.smk <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/rules/gb-model/demand_and_dsr.smk#L89-L106>`_

**Script**: `scripts/gb_model/demand_and_dsr/synthesise_gb_regional_data.py <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/scripts/gb_model/demand_and_dsr/synthesise_gb_regional_data.py>`_

The national-level hydrogen data produced by ``create_hydrogen_data_tables`` must be disaggregated to GB regions before network integration.
This rule distributes three national hydrogen datasets to regional level:

- **H2 demand** (``H2_demand_annual.csv`` → ``regional_H2_demand_annual.csv``)
- **H2 storage capacity** (``H2_storage_capacity.csv`` → ``regional_H2_storage_capacity.csv``)
- **Non-networked electrolysis demand** (``non_networked_electrolysis_demand_annual.csv`` → ``regional_non_networked_electrolysis_demand_annual.csv``)

**Distribution Method**:

Regional distribution is based on a reference technology pattern from the FES Building Block Data.
For hydrogen datasets, the reference is **"hydrogen electrolysis"** capacity distribution.

**Process**:

1. Load national annual data (indexed by year)
2. Load regional GB data containing the reference technology (hydrogen electrolysis)
3. Calculate the regional distribution pattern from reference technology:
   
   .. math::
   
      \text{Regional Share}_{bus,year} = \frac{\text{Reference Capacity}_{bus,year}}{\sum_{buses} \text{Reference Capacity}_{bus,year}}

4. Apply regional shares to national data:
   
   .. math::
   
      \text{Regional Data}_{bus,year} = \text{National Data}_{year} \times \text{Regional Share}_{bus,year}

5. Export regional data with MultiIndex (bus, year)

**Configuration**:

The reference technology mapping is defined in ``config/config.gb.2024.yaml`` under ``fes.gb.regional_distribution_reference`` (lines 843-851):

.. literalinclude:: ../../config/config.gb.2024.yaml
   :language: yaml
   :lines: 843-851

**Rationale**:

Using hydrogen electrolysis capacity as the distribution reference is justified because:

- Regions with higher electrolysis capacity in FES projections are likely to have higher hydrogen infrastructure development
- Industrial hydrogen demand tends to co-locate with production facilities
- Storage infrastructure is typically sized proportionally to regional production/consumption
- This approach leverages the detailed regional planning in the FES Building Block Data

**Output**:

Regional CSV files with columns ``bus``, ``year``, and data values (``p_set`` for demand/power, ``e_nom`` for storage energy capacity).


.. _european_hydrogen_data:

European Hydrogen Data Integration
-----------------------------------

Integration of European hydrogen data involves two rules that extend GB-only datasets to include European countries.

.. _add_eur_H2_demand:

Adding European Hydrogen Demand
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Rule**: ``add_eur_H2_demand``

**Location**: `rules/gb-model/hydrogen.smk <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/rules/gb-model/hydrogen.smk#L46-L60>`_

**Script**: `scripts/gb_model/hydrogen/add_eur_H2_demand.py <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/scripts/gb_model/hydrogen/add_eur_H2_demand.py>`_

**Input Data**:

- GB regional H2 demand from FES processing
- TYNDP future H2 demand projections (``data/gb-model/tyndp_h2_demand.csv``)
- Current European H2 consumption (``data/gb-model/downloaded/eur_H2_demand_today.xlsx``)

**Process**:

1. Convert European country names to ISO2 codes for consistency
2. Combine current consumption data (converted from tonnes/year to MWh/year) with TYNDP future projections
3. Interpolate/extrapolate to fill missing years between present day and furthest modelled year
4. Filter for countries included in the model configuration
5. Combine with GB demand data to create unified demand dataset
6. Export with MultiIndex (bus, year)

**Assumptions**:

- Linear interpolation between known data points is reasonable for annual aggregation
- Extrapolation beyond TYNDP horizon maintains recent trends
- Country-level aggregation is appropriate (no sub-national detail for European countries)


.. _synthesise_eur_H2_data:

Synthesising Other European Hydrogen Data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Rule**: ``synthesise_eur_H2_data``

**Location**: `rules/gb-model/hydrogen.smk <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/rules/gb-model/hydrogen.smk#L63-L78>`_

**Script**: `scripts/gb_model/hydrogen/synthesise_eur_H2_data.py <https://github.com/open-energy-transition/gb-dispatch-model/blob/master/scripts/gb_model/hydrogen/synthesise_eur_H2_data.py>`_

This rule creates European data for:

- Non-networked electrolysis electricity demand
- Hydrogen storage capacity  
- Grid-connected electrolysis capacity

**Process**:

1. Calculate the ratio between GB-specific dataset and GB hydrogen demand (normalisation)
2. Apply the average ratio for each year to European countries
3. Scale by European country H2 demand to get absolute values
4. Fill any remaining gaps with year-average values
5. Export combined GB + European dataset

**Rationale**:

Since detailed European hydrogen system data equivalent to FES is not available, we assume that:

- The relationship between hydrogen demand and system components (storage, electrolysis) in European countries follows similar patterns to GB
- Using GB ratios is more defensible than arbitrary assumptions
- Country-specific demand scaling ensures absolute values are reasonable

**Limitations**:

- Does not account for country-specific hydrogen system designs
- May overestimate storage needs in countries with steadier renewable generation
- Ignores existing European hydrogen infrastructure and industry


.. _hydrogen-outputs:

Output Files
============

The hydrogen processing rules generate several intermediate data files in the ``resources/gb-model/{fes_scenario}/`` directory:

**Core Hydrogen Data**
----------------------

- ``H2_demand_annual.csv``: Annual net hydrogen demand by year (MWh/year)
  
  - Columns: ``year``, ``data``
  - Data: Total GB hydrogen consumption minus all supply except grid-electrolysis supply

- ``H2_storage_capacity.csv``: Annual hydrogen storage capacity requirements (MWh)
  
  - Columns: ``year``, ``data``
  - Data: Energy storage capacity needed for hydrogen balancing

- ``non_networked_electrolysis_demand_annual.csv``: Off-grid electrolysis electricity demand (MWh/year)
  
  - Columns: ``year``, ``data``
  - Data: Electricity consumed by non-networked electrolysers (added to baseline load)

**Regional Hydrogen Data**
--------------------------

These files are produced by the :ref:`regional distribution process <regional_hydrogen_distribution>`, which disaggregates national GB data to regions:

- ``regional_H2_demand_annual.csv``: Regional hydrogen demand distribution for GB (MWh/year)
  
  - Index: ``bus``, ``year``
  - Column: Demand values
  - Source: National ``H2_demand_annual.csv`` distributed by hydrogen electrolysis capacity

- ``regional_grid_electrolysis_capacities.csv``: Grid-connected electrolyser capacity by region and year (MW)
  
  - Index: ``bus``, ``year``
  - Column: ``p_nom`` (nominal power capacity)
  - Source: FES BB1 regional data aggregated and redistributed

- ``regional_H2_storage_capacity.csv``: Regional hydrogen storage capacity (MWh)
  
  - Index: ``bus``, ``year``
  - Column: ``e_nom`` (energy storage capacity)
  - Source: National ``H2_storage_capacity.csv`` distributed by hydrogen electrolysis capacity

- ``regional_non_networked_electrolysis_demand_annual.csv``: Regional off-grid electrolysis electricity demand (MWh/year)
  
  - Index: ``bus``, ``year``
  - Column: ``p_set`` (electricity demand)
  - Source: National ``non_networked_electrolysis_demand_annual.csv`` distributed by hydrogen electrolysis capacity

**European-Integrated Data**
----------------------------

- ``regional_H2_demand_annual_inc_eur.csv``: Combined GB and European hydrogen demand
  
  - Index: ``year``, ``bus``
  - Column: ``p_set`` (demand in MWh/year)
  - Buses: GB regions (``GB 1`` - ``GB 30``) plus European country codes (``AT``, ``DE``, ``FR``, etc.)

- ``regional_grid_electrolysis_capacities_inc_eur.csv``: Combined regional electrolysis capacities (without costs)
- ``regional_non_networked_electrolysis_demand_annual_inc_eur.csv``: Combined off-grid demand
- ``regional_H2_storage_capacity_inc_eur.csv``: Combined storage capacity requirements (without costs)

**Cost-Enriched Data**
----------------------

These files are produced by the ``assign_costs`` rule, which enriches capacity data with technology costs from `PyPSA technology-data <https://github.com/PyPSA/technology-data>`_ (2035 cost year):

- ``regional_grid_electrolysis_capacities_inc_eur_inc_tech_data.csv``: Grid electrolysis with capital costs, efficiency, and lifetime
- ``regional_H2_storage_capacity_inc_eur_inc_tech_data.csv``: H2 storage with capital costs and technical parameters

These cost-enriched files are used in the PyPSA network composition stage.


.. seealso::

   **Related Documentation**:
   
   - :ref:`system-hydrogen` - Overview of hydrogen in system representation
   - :ref:`gb_data_sources` - FES and other data sources used
   - :doc:`configuration` - Full configuration options reference
   - :doc:`dispatch_redispatch` - How hydrogen interacts with dispatch optimization
   - :doc:`data_cleaning` - Data processing pipeline details
   
   **FAQ and Troubleshooting**:
   
   - :doc:`faq` - Common questions about storage constraints and model behavior
   
   **External Resources**:
   
   - `FES 2024 Data Workbook <https://www.neso.energy/publications/future-energy-scenarios-fes>`_ - Primary data source
   - `TYNDP 2024 <https://tyndp.entsoe.eu/>`_ - European hydrogen demand projections
   - `PyPSA Documentation <https://pypsa.readthedocs.io/>`_ - Core modelling framework
   - `PyPSA technology-data <https://github.com/PyPSA/technology-data>`_ - Technology costs (2035 cost year used for all hydrogen components)

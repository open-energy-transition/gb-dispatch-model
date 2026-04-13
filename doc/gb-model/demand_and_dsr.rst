..
  SPDX-FileCopyrightText: gb-dispatch-model contributors
  SPDX-License-Identifier: CC-BY-4.0

.. _demand_and_dsr:

##########################
Demand and DSR
##########################

Overview
========

The GB dispatch model incorporates demand-side response (DSR) capabilities across multiple demand sectors to provide flexibility in the electricity system. DSR allows for the shifting of electricity consumption from peak to off-peak periods, helping to balance supply and demand while reducing the need for additional generation capacity.

Demand Sectors
==============

The model includes the following demand sectors with DSR capabilities:

1. **Baseline Electricity (Residential)** - Residential electricity demand excluding heat pumps
2. **Baseline Electricity (I&C)** - Industrial and Commercial electricity demand
3. **Residential Heat** - Electricity demand for residential heating systems
4. **I&C Heat** - Electricity demand for industrial and commercial heating systems
5. **EV Charging** - Electric vehicle charging demand

Data Sources
============

Demand and flexibility data are sourced from the Future Energy Scenario FES 2024 workbooks. The model processes FES data to extract:

- Annual electricity demand by sector and region
- DSR capacity and flexibility potential
- Temporal demand profiles
- Technology-specific demand characteristics

Data Processing Workflow
========================

The demand and DSR implementation follows a multi-step workflow:

1. **Demand Table Creation** - Extract annual demand data from FES workbook by technology and region
2. **Flexibility Table Creation** - Process DSR capacity data from FES flexibility sheets
3. **Regional Distribution** - Distribute national data to regional PyPSA buses
4. **Demand Profile Scaling** - Apply temporal profiles to annual demands
5. **Network Composition** - Add DSR components to the PyPSA network

DSR Implementation
==================

DSR is implemented using PyPSA's Store and Link components to model energy storage and power flow capabilities. For each demand sector with DSR, the model creates:

- **DSR Storage Bus** - A dedicated bus for storing shifted energy
- **Shift Link** - Allows energy to be moved from the main demand bus to storage during off-peak periods
- **Reverse Link** - Allows energy to be returned from storage to the main demand bus during peak periods
- **Store Component** - Represents the energy storage capacity with time-dependent constraints

Model Representation
--------------------

The DSR implementation for each sector follows this structure:

.. graphviz::

  digraph DSR_Flow {
      rankdir=LR;

      node [shape=circle,width=1.5, height=1.5, fontsize=12];

      AC [label="AC Bus"]
      Demand [label="Sector Demand"]
      DSR [label="Sector DSR Storage"]
      X [style=invis, width=0, height=0, label=""];

      AC -> Demand [label="Unmanaged Load", fontsize=12];
      Demand -> DSR [label="DSR Shift", fontsize=12];
      DSR -> Demand [label="DSR Reverse", fontsize=12];

      // downward arrow from Demand
      Demand -> X [style=invis];
      Demand:s -> X:n [constraint=false, xlabel="Sector Demand", fontsize=12];
  }

DSR Parameters
==============

DSR operation is controlled by several key parameters:

**DSR Hours**
  Time windows during which DSR can operate. Configured per sector:

  - Residential baseline electricity: 17:00-20:00 (5pm-8pm)
  - I&C baseline electricity: 17:00-20:00 (5pm-8pm)
  - Residential heat: 00:00-22:00 (all day)
  - I&C heat: 00:00-22:00 (all day)
  - EV charging: 08:00-06:00 (8am-6am next day)

**Storage Capacity**
  Calculated as DSR power capacity × duration hours. Represents the maximum energy that can be shifted.

**Availability Profiles**
  Time-dependent constraints on when DSR can operate. For EV DSR, this is linked to vehicle availability patterns.

**Efficiency**
  DSR links operate at 100% efficiency (no losses in shifting energy).

Configuration
=============

DSR is configured in the model configuration files under the ``fes.gb.flexibility`` section:

.. code-block:: yaml

   flexibility:
     carrier_mapping:
       ev_dsr:
         Detail: "V2G impact at peak"
       residential_dsr:
         Detail: "residential DSR impact at peak"
       residential_heat_dsr:
         Detail: "residential heat DSR impact at peak"
       iandc_dsr:
         Detail: "i&c dsr impact at peak"
       iandc_heat_dsr:
         Detail: "i&c heat DSR impact at peak"

     dsr_hours:
       residential_heat_dsr: [0, 22]
       residential_dsr: [17, 20]
       iandc_dsr: [17, 20]
       iandc_heat_dsr: [0, 22]
       ev_dsr: [8, 6]

     load_bus_suffixes:
       residential: " Baseline Electricity (Residential) DSR"
       iandc: " Baseline Electricity (I&C) DSR"
       ev: " EV DSR"
       residential_heat: " Residential Heat DSR"
       iandc_heat: " I&C Heat DSR"

European Integration
====================

For European neighbour countries included in the model, demand and DSR data are synthesized by:

1. Using GB data as a reference for technology mixes
2. Scaling based on relative annual demand totals from energy balance data
3. Applying appropriate time zone shifts for DSR operation windows

This ensures consistent representation of cross-border demand flexibility effects.

Data Processing Pipeline
=========================

The demand and DSR workflow is implemented through Snakemake rules and Python scripts.

Demand Table Creation
----------------------

**Rule**: `create_demand_table`

Input: Annual demand data from FES workbook indexed by scenario, GSP and year
Output: Annual demand data for a particular scenario indexed by GB bus and year
Method: Converts demand from GWh to MWh and outputs regional demand by bus and year.

Baseline Demand Profile Processing
----------------------------------

**Rule**: `cluster_baseline_electricity_demand_timeseries`

Input:
- Baseline electricity demand profiles from PyPSA-Eur workflow
- gb-model bus cluatering
Output: Baseline electricity demand profiles for each bus in the gb-model network
Method: Clusters PyPSA-Eur baseline electricity demand to the gb-model bus layout, producing a historic baseline electricity demand profile for each bus in GB and rest of Europe.

**Rule**: `process_baseline_demand_shape`

Input:
   - Historical baseline electricity demand profiles for each bus in the gb-model network (from `cluster_baseline_electricity_demand_timeseries`)
   - Resistive heater demand profiles for each bus in the gb-model network (from `resistive_heater_demand_profile`)
Output: Normalized demand profile shapes for each network bus for a given year
Method: Removes future resistive heating demand from historical baseline electricity demand to get the net electrified heat demand and then normalizes the resulting demand profiles for each bus.

DSR Flexibility Processing
--------------------------

**Rule**: `create_flexibility_table`

Input: FES flexibility data for each scenario, year and flexibility type (residential, I&C, residential heat, I&C heat, EV)
Output: DSR capacity for each demand sector by region and year
Method: Extracts DSR capacity from FES data, converts to MW and outputs flexibility tables for each flexibility type.


Regional Demand Synthesis
-------------------------

**Rule**: `synthesise_gb_regional_data`

Input: 
 - Annual demand data from FES workbook indexed by scenario, GSP and year
 - DSR capacity for each demand sector by region and year
Output: Annual demand and DSR capacity data for a particular scenario indexed by network bus and year
Method: Clusters demand and DSR capacity from GSP to network bus level

**Rule**: `distribute_eur_demands`

Input: 
- Annual demand data from FES workbook indexed by EU scenario, country and year
- Energy totals weighted by population for each GB node in the network (from PyPSA-Eur workflow)
- Annual demand data for GB (from `create_demand_table`) 
Output: Annual demand data for European neighbours indexed by demand type, network bus and year
Method: Allocates European neighbour annual demand totals across the model. Uses energy totals and regional FES demand data to create consistent demand inputs.

**Rule**: `synthesise_eur_data`

Input: 
 - Annual demand data for each demand type
 - Annual EUR demand data
 - GB DSR capacity data
Output: Annual demand and DSR capacity data for a particular scenario indexed by network bus and year
Method: Synthesizes demand and DSR data for European neighbours using GB data as a reference, scaling based on relative annual demand totals and applying time zone shifts for DSR operation windows.

Demand Profile Scaling
----------------------

**Rule**: `scaled_demand_profile`

Input: 
 - Annual GB demand (from `create_demand_table`)
 - Annual EUR demand (from `distribute_eur_demands`)
 - Normalized demand profile shapes for each network bus for a given year (from `process_baseline_demand_shape`)
Output: Hourly demand profiles for each demand type, network bus and year
Method: Scales normalized demand profile shapes to match annual regional demand totals, producing hourly demand profiles at each network bus for each demand type.
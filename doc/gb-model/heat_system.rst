..
  SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _heating_overview:

#####################
Heat system overview
#####################

Overview
========

The electrified heating system in the GB dispatch model is represented with three primary technologies:

- **Resistive heating** (direct electric heating)
- **Air source heat pump (ASHP)**
- **Ground source heat pump (GSHP)**

Various heating technologies such as district heating, hybrid systems (ASHP with hydrogen boiler, biofuel boiler or resistive heater), and storage heating are consolidated and mapped to one of these three primary technologies for model representation. This approach allows the model to capture the essential electrification pathways.
The technology splits are sourced from the **Future Energy Scenario (FES)** workbook. The configuration maps FES heating technology categories to model representations:

.. _electrified_heating_technologies:

.. code-block:: yaml

   electrified_heating_technologies:
     Electric resistive: resistive
     Electric storage: resistive
     Hybrid (ASHP + Electric resistive): [ASHP, resistive]
     ASHP: ASHP
     DH: [ASHP, GSHP]
     Hybrid (ASHP + Hydrogen boiler): ASHP
     Hybrid (ASHP + Biofuel boiler): ASHP
     GSHP: GSHP


Architecture
============

Heat System Structure
---------------------

The heat system is organized by two main demand sectors as follows:

1. **Residential heat** - Space heating and domestic hot water for households
2. **Industrial & Commercial (I&C) heat** - Space heating and process heat for commercial and industrial buildings (referred to by the services sector in PyPSA-Eur)

Each sector is further subdivided by geographic location type:

- **Urban central** - Urban areas with potential for district heating networks
- **Urban decentral** - Urban areas with individual heating systems
- **Rural** - rural areas with decentralized heating

Model representation
--------------------

The heat system is represented in the model with the following components:

.. graphviz::


  digraph Flow {
      rankdir=LR;   // Left to Right

      node [shape=circle,width=1.5, height=1.5, fontsize=12];

      A [label="AC bus"]
      B [label="Sector Heat"]
      C [label="Sector Heat DSR"]
      X [style=invis, width=0, height=0, label=""];

      A -> B [label="Sector Heat \n unmanaged load", fontsize=12];
      B -> C [label="Sector Heat DSR shift", fontsize=12];
      C -> B [label="Sector Heat DSR reverse",labelangle=-90, fontsize=12];

      // downward arrow from B
      B -> X [style=invis];
      B:s -> X:n [constraint=false, xlabel="Sector Heat demand", fontsize=12];
  }


The Sector refers to either `Residential heat` or `I&C heat` depending on the demand sector being modeled. The AC bus represents the electrical grid connection for the heat system, while the Sector Heat and Sector Heat DSR represent the heat demand and demand-side response capabilities respectively.



Heat Pump Sources and COP Profiling
====================================

Heat pump source availability is configured per geographic location type:

.. code-block:: yaml

   heat_pump_sources:
     urban central: [air]
     urban decentral: [air]
     rural: [air, ground]


Data Processing Pipeline
=========================

The heat system relies on several key processing steps:

Coefficient of Performance (COP) Processing
--------------------------------------------

**Rule**: `process_cop_profiles`

Processes baseline COP profiles from PyPSA-Eur workflow:

- Input: 
      1. COP profiles for the target year, 
      2. clustered population layout,
      3. district heating share
- Output: Hourly COP profiles for each node in the network and heat sources (ASHP, GSHP)
- Method: Calculate COP profiles for ASHP and GSHP weighted on population distribution.

Note: Resistive heating is assumed to have a constant COP of 1 and does not require processing.

Future Energy Scenario Processing
----------------------------------

**Rule**: `process_fes_heat_technologies`

Extracts technology penetration rates from FES workbook:

- Input: 
    1. FES ED3 data with technology shares by sector
    2. :ref:`electrified_heating_technologies` 
- Output: Year-by-year technology consumption curves for:
  - Residential sector heat technologies
  - Industrial & Commercial sector heat technologies
- Method: Parses FES categories and maps to model technology representations

Resistive Heater Demand Profiling
----------------------------------

**Rule**: `resistive_heater_demand_profile`

Creates technology-specific demand profiles:

- Input: 
    1. Energy totals weighted by population for each node in the network (from PyPSA-Eur workflow)
    2. Hourly heat demand shapes (from PyPSA-Eur workflow)
    3. FES technology consumption splits (residential and I&C)
- Output: Hourly resistive heater demand profiles by node and year
- Method: Removes future resistive heating demand from historical electrified heat demand to get the net electrified heat demand.


Heat Demand Shape Processing
-----------------------------

**Rule**: `process_heat_demand_shape`

Generates demand profiles for different heat sources and sectors:

- Input:
    1. Total hourly heat demand (from PyPSA-Eur workflow)
    2. COP profiles generated (from `process_cop_profiles`)
    3. Technology heating mix by sector (from `process_fes_heat_technologies`)
    4. Population-weighted energy totals (from PyPSA-Eur workflow)
- Output: Hourly heat demand profiles by technology and sector for each node in the network
- Method: 
  - Associates demand with appropriate heat source (resistive, ASHP, GSHP)
  - Incorporates COP variations for heat pump systems

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

The heat system is organized by two main demand sectors and follow the structure as in the PyPSA-Eur model:

1. **Residential heat** - Space heating and domestic hot water for households
2. **Industrial & Commercial (I&C) heat** - Space heating and process heat for commercial and industrial buildings

Each sector is further subdivided by geographic location type:

- **Urban central** - Cities with potential for district heating networks
- **Urban decentral** - Suburban areas with individual heating systems
- **Rural** - Sparsely populated areas with decentralized heating


Heat Pump Sources and COP Profiling
====================================

Heat pump source availability is configured per geographic location type:

.. code-block:: yaml

   heat_pump_sources:
     urban central: [air]
     urban decentral: [air]
     rural: [air, ground]


Heat Demand Modeling
====================

Heat Demand Representation
--------------------------

Heat demand in the model is driven by:

1. **Daily heat demand** - Based on heating degree days and building thermal properties
2. **Intraday profiles** - Time-of-use patterns from BDEW (German heating profiles adapted for GB context)
3. **Technology-specific demand** - Split between resistive, ASHP, and GSHP based on technology shares

The total heat demand is divided among:

- **Residential sector**: Space heating and domestic hot water for households
- **I&C sector**: Space heating and hot water for commercial and industrial facilities

Data Processing Pipeline
=========================

The heat system relies on several key processing steps:

Coefficient of Performance (COP) Processing
--------------------------------------------

**Rule**: `process_cop_profiles`

Processes baseline COP profiles from PyPSA-Eur workflow:

- Input: Base COP profiles for the target year, clustered population layout, district heating share
- Output: Hourly COP profiles by location type and heat source
- Method: Incorporates district heating temperature lift adjustments and clustering

Future Energy Scenario Processing
----------------------------------

**Rule**: `process_fes_heat_technologies`

Extracts technology penetration rates from FES workbook:

- Input: FES ED3 data with technology shares by sector
- Output: Year-by-year technology consumption curves for:
  - Residential sector heat technologies
  - Industrial & Commercial sector heat technologies
- Method: Parses FES categories and maps to model technology representations

Resistive Heater Demand Profiling
----------------------------------

**Rule**: `resistive_heater_demand_profile`

Creates technology-specific demand profiles:

- Input: 
  - Energy totals by region
  - Hourly heat demand shapes
  - FES technology consumption splits (residential and I&C)
- Output: Hourly resistive heater demand profiles by region and year
- Method: Applies technology shares to total heat demand to isolate resistive heating demand

Heat Demand Shape Processing
-----------------------------

**Rule**: `process_heat_demand_shape`

Generates demand profiles for different heat sources and sectors:

- Input:
  - Total hourly heat demand
  - COP profiles
  - Technology heating mix (technology-specific consumption)
  - Energy totals
- Output: Hourly heat demand profiles by technology and sector
- Method: 
  - Associates demand with appropriate heat source (resistive, ASHP, GSHP)
  - Incorporates COP variations for heat pump systems
  - Splits demand by geographic location type

Configuration Parameters
=========================

Key Heat System Parameters
---------------------------

The following parameters control heat system behavior in configuration files:

**Heat Pump Sources** (``config.default.yaml``):

.. code-block:: yaml

   sector:
     heat_pump_sources:
       urban central: [air]
       urban decentral: [air]
       rural: [air, ground]

**Heat Pump Sink Temperatures** (PyPSA-Eur derived):

- Target heating temperatures for individual systems
- Affects COP calculations via temperature lift requirements
- Typically 35-55°C for individual heating, 60-80°C for district heating

**Residential Heat DSM Parameters**:

- Enables/disables thermal storage flexibility
- Controls checkpoint hours for consumption enforcement
- Specifies flexibility availability over time (2020-2050)

**Technology Mapping** (``config.gb.2024.yaml``):

Maps FES technology names to model representation, allowing:

- Year-by-year scenario evolution
- Multiple scenario narratives (e.g., Leading the Way, System Transformation, Consumer Transformation)
- Consistent technology classification across GB and EU regions




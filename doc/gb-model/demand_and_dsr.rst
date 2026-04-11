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

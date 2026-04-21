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
---------------

The model includes the following demand sectors with DSR capabilities:

1. **Baseline Electricity (Residential)** - Residential electricity demand excluding heat pumps
2. **Baseline Electricity (I&C)** - Industrial and Commercial electricity demand
3. **Residential Heat** - Electricity demand for residential heating systems
4. **I&C Heat** - Electricity demand for industrial and commercial heating systems
5. **EV Charging** - Electric vehicle charging demand
6. **Hydrogen Electrolyser Demand** - Electricity demand for hydrogen production via electrolysis
7. **Additional Demand** - Other electricity demand not captured in the above categories (e.g., direct transmission demand, T&D losses)

Data Sources
============

Great Britain data
------------------
Demand and flexibility data are sourced from the Future Energy Scenario FES 2024 workbooks. The model processes FES data to extract:

- Annual electricity demand by sector and region
- DSR capacity and flexibility potential
- Technology-specific demand characteristics

The table below summarizes the specific sheets extracted from the FES workbook for each of the demand sectors:

.. _demand_and_dsr_data_sources:
.. list-table:: FES Data Sources for Demand and DSR
   :header-rows: 1

   * - Demand Sector
     - FES Workbook Sheet
   * - Baseline Electricity (Residential)
     - Building Block Data (BB1)
   * - Baseline Electricity (I&C)
     - Building Block Data (BB1)
   * - Residential Heat
     - Building Block Data (BB1), Gas Demand Summary (ED3) and Flexibility (FLX1)
   * - I&C Heat
     - Building Block Data (BB1), Gas Demand Summary (ED3) and Flexibility (FLX1)
   * - EV Charging
     - Building Block Data (BB1), FES ED5(EV Demand) and Flexibility (FLX1)
   * - Hydrogen Electrolyser Demand
     - Building Block Data (BB1), WS1 (Whole System)
   * - Additional demand
     - Electricity Demand Summary (ED1)

European data
---------------

For European neighbour countries included in the model, the demand data is sourced from the ENTSO-E energy balance data (via ``energy_totals.csv``). Specific data requirements are explained in detail in the corresponding sections of the documentation (e.g., :ref:`heat_system` uses ES2 from the FES workbook for European demand assumptions).

Configuration
=============

DSR is configured in the model configuration files under the ``fes.gb.flexibility`` section:

.. code-block:: yaml

   flexibility:
     carrier_mapping:
       battery:
          Data item: "Electricity storage energy storage potential"
          Detail: "Battery"
       ev_v2g:
          Detail: "V2G maximum potential"
       ev_dsr:
          Detail: "smart charging impact at peak"
       residential_dsr:
          Detail: "residential DSR impact at peak"
       residential_heat_dsr:
          Detail:
          - Residential peak reduction - storage heater flex
          - Residential peak reduction - hybrid heat pump flex
          - Residential peak reduction - thermal storage and district heating flex
       iandc_dsr:
          Detail: "i&c dsr impact at peak"
       iandc_heat_dsr:
          Detail:
          - Non-residential peak reduction - storage heater flex
          - Non-residential peak reduction - hybrid heat pump flex
          - Non-residential peak reduction - thermal storage and district heating flex
     dsr_hours:  # hours during which demand-side management can occur (E.g., 5pm-8pm)
       residential_heat_dsr: [0, 22]
       residential_dsr: [17, 20]
       iandc_dsr: [17, 20]
       iandc_heat_dsr: [0, 22]
       ev_dsr: [8, 6]
     carrier_suffix:
       residential: " Baseline Electricity (Residential) DSR"
       iandc: " Baseline Electricity (I&C) DSR"
       ev: " EV DSR"
       residential_heat: " Residential Heat DSR"
       iandc_heat: " I&C Heat DSR"
     v2g_storage_to_capacity_ratio: 7.14 # based on ratio in FES 2021. Same value will be applied to all future years.

System Components
==================

DSR is implemented using PyPSA's Bus, Store, Link and Load components to model energy storage and power flow capabilities. For each demand sector with DSR, the model creates:

- **DSR Storage Bus** - A dedicated bus for storing shifted energy
- **Shift Link** - Allows energy to be moved from the main demand bus to storage during off-peak periods
- **Reverse Link** - Allows energy to be returned from storage to the main demand bus during peak periods
- **Store Component** - Represents the energy storage capacity with time-dependent constraints
- **Load Component** - Represents the total demand that must be met, including both the baseline demand and any additional demand from DSR.

The representation of each demand sector is illustrated in the :ref:`_model_representation` section of the documentation, which provides detailed descriptions of the components and their interactions.

Implementation Notes
====================

Model Representation
--------------------

.. _model_representation:

The DSR implementation for each demand type (residential, I&C, EV and heat) follows this structure:

.. graphviz::

  digraph Flow {
      rankdir=LR;   // Left to Right

      node [shape=box, style=filled];

      AC_bus [label="AC Bus", fillcolor="#B3D9FF", shape=ellipse, width=2, height=1.5, fixedsize=true];
      Sector_bus [label="Sector Bus", fillcolor="#FFD1DC", shape=ellipse, width=1.8, height=1.2, fixedsize=true];
      Sector_DSR_bus [label="Sector DSR Bus", fillcolor="#FFD1DC", shape=ellipse, width=1.8, height=1.2, fixedsize=true];
      heat_load [label="Sector Load", fillcolor="#FFDAB9"];
      sector_dsr_store [label="Sector DSR Store", fillcolor="#FFFACD"];

      AC_bus -> Sector_bus [label="unmanaged load link"];
      Sector_bus -> heat_load;
      Sector_bus -> Sector_DSR_bus [label="charge"];
      Sector_DSR_bus -> Sector_bus [label="discharge"];
      Sector_DSR_bus -> sector_dsr_store [dir=both];
  }

"Sector" refers to the specific demand sector being modeled (e.g., residential, I&C, EV, heat or hydrogen).
Whilst the general structure is consistent across demand types, some extra components are added for specific sectors which can be found in the corresponding documentation sections (e.g., EV charging includes additional V2G storage components).


DSR Parameters
--------------

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
  Time-dependent constraints on when DSR can operate. This is specific for EV DSR and is linked to vehicle availability patterns.


European Integration
--------------------

For European neighbour countries included in the model, demand and DSR data are synthesized by:

1. Using GB data as a reference for technology mixes
2. Scaling based on relative annual demand totals from energy balance data
3. Applying appropriate time zone shifts for DSR operation windows

This ensures consistent representation of cross-border demand flexibility effects.

Data Processing Workflow
========================

The demand and DSR implementation follows a multi-step workflow:

1. **Demand Table Creation** - Extract annual demand data from FES workbook by technology and region
2. **Flexibility Table Creation** - Process DSR capacity data from FES flexibility sheets
3. **Regional Distribution** - Distribute national data to regional PyPSA buses
4. **Demand Profile Scaling** - Apply temporal profiles to annual demands
5. **Network Composition** - Add DSR components to the PyPSA network

Data Processing Pipeline
=========================

The demand and DSR workflow is implemented through Snakemake rules in `rules/gb-model/demand_and_dsr.smk` and Python scripts in the folder `scripts/gb_model/demand_and_dsr`.

The rulegraph for the demand and DSR workflow is illustrated below, showing the key processing steps and their dependencies:

.. image:: img/demand_and_dsr.svg
   :align: center

The demand and DSR workflow was built based on several key processing steps as follows:

* **create_demand_table**: Extracts annual demand data from FES workbook indexed by scenario, GSP and year.
* **cluster_baseline_electricity_demand_timeseries**: Clusters PyPSA-Eur baseline electricity demand to the gb-model bus layout, producing a historic baseline electricity demand profile for each bus in GB and rest of Europe.
* **process_baseline_demand_shape**: Removes future resistive heating demand from historical baseline electricity demand to get the net electrified heat demand and then normalizes the resulting demand profiles for each bus.
* **create_flexibility_table**: Extracts DSR capacity from FES data, converts to MW and outputs flexibility tables for each flexibility type.
* **synthesise_gb_regional_data**: Clusters demand and DSR capacity from GSP to network bus level for GB.
* **distribute_eur_demands**: Allocates European neighbour annual demand totals across the model. Uses energy totals and regional FES demand data to create consistent demand inputs.
* **synthesise_eur_data**: Synthesizes demand and DSR data for European neighbours using GB data as a reference, scaling based on relative annual demand totals and applying time zone shifts for DSR operation windows.
* **scaled_demand_profile**: Scales normalized demand profile shapes to match annual regional demand totals
* **add_extra_demands**: Adds additional demand from FES workbook not accounted for in the BB1 sheet (e.g., direct transmission demand, T&D losses) to the model inputs.

Assumptions:
------------
 - DSR links operate at 100% efficiency (no losses in shifting energy).
 - DSR links and stores do not have any capital or operational costs associated with them, as the model focuses on the technical potential of DSR rather than economic optimization.
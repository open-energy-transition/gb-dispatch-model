..
  SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _heating_overview:

############
Heat System
############

This page provides an overview of the electrified heating system representation in the GB dispatch model, including the data sources, components, configuration, and implementation.

Overview
========

The electrified heating system in the model is represented with three primary technologies:

- **Resistive heating (direct electric heating)**
- **Air source heat pump (ASHP)**
- **Ground source heat pump (GSHP)**

Various heating technologies such as district heating, hybrid systems (ASHP with hydrogen boiler, biofuel boiler or resistive heater), and storage heating are consolidated and mapped to one of these three primary technologies for model representation. This approach allows the model to capture the essential electrification pathways.
The technology splits are sourced from the Future Energy Scenario (FES) workbook and are detailed in the :ref:`electrified_heating_technologies` section.

Heat System Structure
---------------------

The heat system is organized by two main demand sectors as follows:

1. **Residential heat** - Space heating and domestic hot water for households
2. **Industrial & Commercial (I&C) heat** - Space heating and process heat for commercial and industrial buildings (referred to by the services sector in PyPSA-Eur)


Data Sources
=============
Great Britain Data
------------------

The GB data for the model is sourced as follows:

1. **BB1 (Building Block Data)**: Regional GSP-level data for residential and I&C sectors

2. **ED3 (Gas Demand Summary)**: Annual heat demand totals at national level for residential and I&C sectors, broken down by heating technology categories from the FES workbook (in TWh, converted to MWh)

3. **FES FLX1 (Flexibility)**: Annual DSR capacity for residential and I&C sectors by scenario and year (in GW, converted to MW)


European Data 
-------------
For countries outside of Great Britain, the model obtains the data from the FES workbook **ES2 (European Electricity Supply Table)**. The table provides annual electricity installed capacity and annual demand assumptions for each technology category by country.

To consolidate the European data with the GB data, the EU scenarios are mapped to the GB scenarios as follows:

.. list-table:: Scenario Mapping
   :header-rows: 1

   * - EU Scenario
     - GB Scenario
   * - EU Green
     - Holistic Transition , Electric Engagement
   * - EU Blue
     - Hydrogen Evolution, Counterfactual

The heat demands for the European countries are calculated by scaling the annual electricity demands to match the annual heat demands as a share of the total electricity demand in GB. 

PyPSA-Eur Data
----------------
Apart from the data from FES workbook, the model also uses baseline COP profiles and district heating shares from the PyPSA-Eur workflow.

The COP profiles provides population-weighted COP values for ASHP and GSHP across different nodes in the network. The ASHP COP profiles are available for all geographic location types such as urban central, urban decentral and rural areas, while the GSHP COP profiles are only available for rural areas.

The district heating shares are used to determine the share of heat demand that is met through district heating in the model. The district heating shares are applied only for GB nodes in the network and are not applied to the European countries.

.. _heat_system_components:

System Components
=================

The heat system model includes the following key buses:

- **AC Bus**: Represents the connection to the electrical grid for the heat system.

- **Sector heat Bus**: Represents the bus for the heating sector (residential or I&C) and serves as the connection point for heat demand and demand-side response components.

- **Sector heat DSR Bus**: Represents the bus for demand-side response (DSR) capabilities in the heat sector, allowing for the shifting and reversing of heat demand.

The buses are modelled using the PyPSA component type `Bus`.

Links between the buses represent the flow of electricity to meet heat demand and the ability to shift or reverse demand through DSR. The key links connected between these buses include:

- **Unmanaged load link**: Represents the direct electricity demand from the AC bus to meet the heat demand on the Sector heat bus without any demand-side response.

- **Charge link**: Represents the ability to shift heat demand from the Sector heat bus to the Sector heat DSR bus, allowing for demand-side response actions.

- **Discharge link**: Represents the ability to shift heat demand back from the Sector heat DSR bus to the Sector heat bus when demand-side response actions are reversed.

The links are modelled using the PyPSA component type `Link`.

Additionally, the Sector heat DSR bus is connected to a **Sector heat DSR Store**, which represents the storage of shifted heat demand for later use. This allows the model to capture the temporal shifting of heat demand in response to grid conditions.
The store is modelled using the PyPSA component type `Store`.

Finally, the **Sector heat Load** represents the total heat demand for the sector (residential or I&C) that must be met by the electricity supplied through the AC bus and managed through the Sector heat bus and Sector heat DSR bus. This load is modelled using the PyPSA component type `Load`.

Detailed descriptions of these components and their interactions are provided in the :ref:`demand_and_dsr` section.


Configuration
=============

The configuration maps FES heating technology categories to model representations in the config file `config.gb.2024.yaml``:

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

Heat pump source availability is configured per geographic location type in the config file `config.default.yaml`:

.. code-block:: yaml

   heat_pump_sources:
     urban central: [air]
     urban decentral: [air]
     rural: [air, ground]

Implementation Notes
====================

Model representation
--------------------

The heat system is represented in the model with the following components:

.. graphviz::

  digraph Flow {
      rankdir=LR;   // Left to Right

      node [shape=box, style=filled];

      AC_bus [label="AC Bus", fillcolor="#B3D9FF", shape=ellipse, width=2, height=1.5, fixedsize=true];
      Sector_heat_bus [label="Sector heat Bus", fillcolor="#FFD1DC", shape=ellipse, width=1.8, height=1.2, fixedsize=true];
      Sector_heat_DSR_bus [label="Sector heat DSR Bus", fillcolor="#FFD1DC", shape=ellipse, width=1.8, height=1.2, fixedsize=true];
      heat_load [label="Sector heat Load\n(unmanaged)", fillcolor="#FFDAB9"];
      sector_heat_dsr_store [label="Sector heat DSR Store", fillcolor="#FFFACD"];

      AC_bus -> Sector_heat_bus [label="unmanaged load link"];
      Sector_heat_bus -> heat_load;
      Sector_heat_bus -> Sector_heat_DSR_bus [label="charge"];
      Sector_heat_DSR_bus -> Sector_heat_bus [label="discharge"];
      Sector_heat_DSR_bus -> sector_heat_dsr_store [dir=both];
  }


The Sector refers to either `Residential` or `I&C` depending on the demand sector being modeled. The AC bus represents the electrical grid connection for the heat system, while the Sector Heat and Sector Heat DSR represent the heat demand and demand-side response capabilities respectively.


Implementation Steps
=====================

**Data Processing**:

The heat system is built through a pipeline implemented across ``rules/gb-model/heat.smk`` and ``rules/gb-model/demand_and_dsr.smk``:

The rulegraph for the heat system is illustrated below, showing the key processing steps and their dependencies:

.. image:: img/heat_workflow.svg
   :align: center

The heat system pipeline was built based on several key processing steps as follows:

1. **process_cop_profiles**: Processes baseline COP profiles from the PyPSA-Eur workflow to generate hourly COP profiles for each node in the network and heat sources (ASHP, GSHP).
2. **process_fes_heat_technologies**: Extracts share of electrified heating technologies from the FES workbook
3. **resistive_heater_demand_profile**: Creates technology-specific demand profiles for resistive heating by removing future resistive heating demand from historical electrified heat demand.
4. **heat_demand_electricity_load_profile**: Generates hourly heat demand profiles by technology and sector for each node in the network and also accounts for the COP of heat pump technologies to convert heat demand into equivalent electricity demand.

Assumptions
------------

- The model maps all heating technologies to three primary categories (resistive, ASHP, GSHP) for simplicity, which may not capture all nuances of specific technologies provided in the FES workbook.
- The share of district heating is assumed from what is provided in the PyPSA-Eur workflow.


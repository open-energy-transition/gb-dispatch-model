..
  SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _generators:

##########################################
Generation Components
##########################################

This page describes how generation assets are represented in the model, including their data sources, capacity assignment, cost parameterisation, availability modelling, and implementation.

Overview
========

The generation system covers all electricity-producing assets connected to the modelled network, spanning both Great Britain and the surrounding European countries.
Generators are split into two broad groups:

- **Conventional generators**: Thermal and nuclear plant (CCGT, OCGT, gas engine, coal, nuclear, oil, waste, biomass, geothermal) that can be dispatched by the model within physical and contractual limits
- **Renewable generators**: Solar, onshore wind, offshore wind, run-of-river hydro, and pumped hydro storage (PHS) whose output is constrained by weather-derived capacity-factor time series
- **Hydrogen-to-electricity generators**: Fuel cells and hydrogen turbines — documented in :doc:`hydrogen_overview`

Additionally, a subset of conventional generators with heat-driven cogeneration duties are subject to simplified *combined heat and power (CHP)* constraints that enforce minimum loading when heat demand is present.

All generators are modelled as *fixed-capacity*, non-extendable assets — the model dispatches within the capacities supplied and cannot invest in new plant.

The figure below gives a high-level view of the generator pipeline:

.. graphviz::

   digraph {
      rankdir=LR;
      node [shape=box, style=filled];

      fes    [label="FES BB1\n(future caps)", fillcolor="#B3D9FF"];
      dukes  [label="DUKES 5.11\n(existing caps)", fillcolor="#B3D9FF"];
      entsoe [label="ENTSO-E\n(outage data)", fillcolor="#B3D9FF"];
      costs  [label="FES AS.1/AS.7\n+ PyPSA tech-data\n(costs)", fillcolor="#B3D9FF"];
      lcr    [label="Low Carbon\nContracts", fillcolor="#B3D9FF"];

      ppl    [label="Powerplants\ntable", fillcolor="#FFFACD"];
      avail  [label="Monthly availability\nfractions", fillcolor="#FFFACD"];
      enrich [label="Enriched powerplants\n(+ costs)", fillcolor="#FFFACD"];

      network [label="PyPSA Network\n(Generators / Stores)", fillcolor="#90EE90", shape=ellipse];

      fes -> ppl;
      dukes -> ppl;
      lcr -> ppl;
      ppl -> enrich;
      costs -> enrich;
      enrich -> network;
      entsoe -> avail -> network;
   }


.. _generators-data-sources:

Data Sources
============

FES BB1 — Future Capacity Projections
--------------------------------------

The primary source of *future* generator capacities for Great Britain is the NESO **Future Energy Scenarios (FES) 2024** workbook, sheet **BB1 (Building Blocks)**.

BB1 provides annual capacity projections (in GW) disaggregated to Grid Supply Points (GSPs) for each FES scenario.
Carrier and component-set assignment is performed through the ``fes.gb.carrier_mapping`` and ``fes.gb.set_mapping`` dictionaries in the configuration.

DUKES 5.11 — Existing Infrastructure
--------------------------------------

The **Digest of UK Energy Statistics (DUKES) table 5.11** provides a snapshot of *current* (2023) installed capacities for major power producers in Great Britain.
It is used to:

1. Anchor the spatial distribution of generators where FES does not provide GSP-level data, only Transmission Operator (TO) level data.
2. Provide current-year capacity baselines that are blended with FES future projections for the distribution step

Carrier assignment from DUKES uses the ``dukes-5.11.carrier_mapping`` and ``dukes-5.11.set_mapping`` configuration sections.

ENTSO-E Transparency Platform — Outage Data
-------------------------------------------

Historical generation unit unavailability data is retrieved from the **ENTSO-E Transparency Platform** (document type A77, "Unavailability of Generation Units") via its public API.
Data covers Great Britain over the period configured by ``entsoe_unavailability.start_date`` / ``entsoe_unavailability.end_date`` (default 2020–2024).

.. note::
   API access requires an ``ENTSO_E_API_KEY`` set in your ``.env`` file or shell environment.
   Register at https://transparency.entsoe.eu/ to obtain a key.

FES Costing Workbook — Cost Data
----------------------------------

Running costs (fuel and variable O&M) for GB generators are drawn from the **FES costing workbook**:

- **AS.1 (Power Generation costs)**: Variable O&M (VOM) and fuel costs per technology and year
- **AS.7 (Carbon Cost)**: CO₂ price trajectory used to compute marginal carbon costs

These data are combined with capital/fixed-cost parameters from **PyPSA technology-data** (costs at the configured planning horizon year, default 2035).

PyPSA-Eur Powerplant Database
-------------------------------

For **European** countries (outside GB), conventional generator capacities are sourced from the PyPSA-Eur powerplant matching pipeline, supplemented with FES European supply data (``fes.eur.carrier_mapping``).

Low Carbon Contracts Register
------------------------------

The **Low Carbon Contracts Company (LCCC)** register of active Contracts for Difference (CfD) is used to identify additional renewable generator capacities (offshore wind, onshore wind, solar, biomass, waste) that may not yet be reflected in the FES or DUKES data.


.. _generators-components:

System Components
=================

.. _generators-conventional:

Conventional Generators
-----------------------

**PyPSA Component**: ``Generator`` attached to the regional AC bus

Conventional generators include all thermal plant that can be freely dispatched up to their installed capacity.

The carriers modelled are:

+---------------------------+---------------------------------+
| Carrier                   | Notes                           |
+===========================+=================================+
| ``CCGT``                  | Combined-cycle gas turbine      |
+---------------------------+---------------------------------+
| ``OCGT``                  | Open-cycle gas turbine          |
+---------------------------+---------------------------------+
| ``engine``                | Gas reciprocating engine        |
+---------------------------+---------------------------------+
| ``nuclear``               | Nuclear (baseload)              |
+---------------------------+---------------------------------+
| ``coal``                  | Coal plant                      |
+---------------------------+---------------------------------+
| ``oil``                   | Oil-fired plant                 |
+---------------------------+---------------------------------+
| ``waste``                 | Waste incineration              |
+---------------------------+---------------------------------+
| ``biomass``               | Biomass/bioenergy               |
+---------------------------+---------------------------------+
| ``geothermal``            | Geothermal (minor capacity)     |
+---------------------------+---------------------------------+

Capacity is set to the FES BB1 projection for the model year, spatially distributed to model regions.

.. _generators-renewables:

Renewable Generators
---------------------

**PyPSA Component**: ``Generator`` with a time-varying ``p_max_pu`` profile

Renewable generators are capacity-constrained to an hourly capacity-factor time series derived from the atlite cutout and ERA5/SARAH reanalysis data via the standard PyPSA-Eur pipeline.

The renewable carriers are:

+---------------------------+------------------------------------------+
| Carrier                   | Notes                                    |
+===========================+==========================================+
| ``solar``                 | Utility-scale and rooftop PV             |
+---------------------------+------------------------------------------+
| ``onwind``                | Onshore wind                             |
+---------------------------+------------------------------------------+
| ``offwind-dc``            | Offshore wind (DC-connected)             |
+---------------------------+------------------------------------------+
| ``hydro``                 | Reservoir hydro                       |
+---------------------------+------------------------------------------+

.. _generators-chp:

Combined Heat and Power (CHP)
------------------------------

**PyPSA Component**: ``Generator`` with a time-varying ``p_min_pu`` profile

CHPs are conventional generators (CCGT, OCGT, biomass, waste, etc.) that also supply heat.
Rather than implementing full sector-coupling, the model applies a simplified *minimum loading* constraint based on local heat demand.

The ``p_min_pu`` profile for each CHP generator is derived from the normalised hourly heat demand of the model region it serves:

.. math::

   p_\text{min,pu}(t) =
   \begin{cases}
     0 & \text{if } \hat{q}(t) < \delta_\text{shutdown} \\
     \max\!\left(\dfrac{\hat{q}(t)}{c_b},\ p_\text{min}\right) & \text{otherwise}
   \end{cases}

where:

- :math:`\hat{q}(t)` — normalised heat demand (fraction of annual peak) at time *t*
- :math:`c_b` — heat-to-power ratio (``chp.heat_to_power_ratio``, default 1.5)
- :math:`p_\text{min}` — minimum operation level when running (``chp.min_operation_level``, default 0.3)
- :math:`\delta_\text{shutdown}` — heat demand threshold below which the CHP may shut down completely (``chp.shutdown_threshold``, default 0.1)

This ensures CHPs run at least enough to cover heat demand, without modelling a separate heat bus.

.. _generators-availability:

Availability Modelling
======================

Monthly availability fractions are computed for GB thermal generators using ENTSO-E outage data.
The pipeline:

1. **Retrieves** planned and forced outage records from the ENTSO-E API by time period (chunked to stay within API rate limits)
2. **Filters** outages that exceed ``entsoe_unavailability.max_unavailable_days`` (default 366 days), treating very long outages as effectively retired plant rather than temporary unavailability
3. **Accumulates** unavailable MW on a 1-minute resolution time series for each carrier, then takes the monthly mean
4. **Calculates** availability fractions as:

.. math::

   a_m = \frac{C_\text{total} - \overline{U}_m}{C_\text{total}}

where :math:`C_\text{total}` is the total installed capacity for a carrier (from DUKES) and :math:`\overline{U}_m` is the mean unavailable capacity in month *m*.

The resulting monthly profile is applied uniformly across all generators of the same carrier for the simulated year, scaling their effective ``p_nom``.
This is also applied to European neighbouring countries by default unless explicitly de-selected (entsoe_unavailability.extend_to_eur_regions).

.. _generators-costs:

Cost Assignment
===============

Each generator in the powerplants table is enriched with cost and technical parameters through a multi-source lookup:

1. **FES AS.1**: Provides year-specific fuel costs and VOM for GB-relevant technologies; averaged across scenarios because scenario names change between FES editions
2. **FES AS.7**: Provides year-specific carbon price (£/tCO₂); averaged across scenarios
3. **PyPSA technology-data**: Provides capital costs (£/MW), fixed O&M (£/MW/year), efficiency, and lifetime for technologies absent from the FES costing workbook
4. **Default characteristics**: Fallbacks defined in ``fes_costs.default_characteristics`` ensure all columns exist even when data is missing

Marginal costs are assembled as:

.. math::

   c_\text{marginal} = \frac{c_\text{fuel}}{\eta} + c_\text{VOM} + c_\text{CO2} \cdot I_\text{CO2}

where :math:`\eta` is efficiency, :math:`c_\text{fuel}` thermal price, :math:`c_\text{VOM}` variable O&M, :math:`c_\text{CO2}` carbon price, and :math:`I_\text{CO2}` the CO₂ intensity.


.. _generators-configuration:

Configuration
=============

Carrier and Set Definitions
-----------------------------

The carriers included in the model and how they are treated (conventional vs renewable, generator vs store) are controlled by:

From ``config/config.gb.2024.yaml``:

.. literalinclude:: ../../config/config.gb.2024.yaml
   :language: yaml
   :start-after: # [doc:electricity-config-start]
   :end-before: # [doc:electricity-config-end]

CHP Constraints
----------------

From ``config/config.gb.2024.yaml``:

.. literalinclude:: ../../config/config.gb.2024.yaml
   :language: yaml
   :start-after: # [doc:chp-config-start]
   :end-before: # [doc:chp-config-end]

ENTSO-E Unavailability
-----------------------

From ``config/config.gb.2024.yaml``:

.. literalinclude:: ../../config/config.gb.2024.yaml
   :language: yaml
   :start-after: # [doc:entsoe-unavailability-config-start]
   :end-before: # [doc:entsoe-unavailability-config-end]


.. _generators-implementation-notes:

Implementation Notes
====================

**Data Processing Workflow**:

The generator system is built through a pipeline implemented in ``rules/gb-model/generators.smk``:

.. image:: img/generators_workflow.svg
   :align: center

.. note::
   The graph above was generated using::

      pixi run filtered_rulegraph \
      "resources/GB/gb-model/HT/fes_powerplants_inc_tech_data.csv
      resources/GB/gb-model/GB_generator_monthly_availability_fraction.csv
      resources/GB/gb-model/chp_p_min_pu.csv" \
      "doc/gb-model/img/generators_workflow.svg" \
      "-w fes_scenario -w year" \
      "-s 10,8" \
      "-f rules/gb-model/generators.smk"

   The ``filtered_rulegraph`` task allows us to trim the full DAG to generator-related rules only.

1. **Retrieve outage data** (``retrieve_entsoe_unavailability_data``): Fetches ENTSO-E unavailability records for GB over the configured date range, chunked to respect API limits
2. **Monthly availability fractions** (``generator_monthly_availability_fraction``): Combines outage records with DUKES capacity baselines to produce per-carrier, per-month availability fractions
3. **Powerplants table** (``create_powerplants_table``): Assembles GSP-level capacity table from FES BB1 (future), DUKES (existing), and European supply data; handles both direct GSP-level data and distribution from Transmission Owner (TO) regional totals
4. **Cost assignment** (``assign_costs``): Enriches the powerplants table with fuel costs, VOM, carbon costs, capital costs, and technical parameters; produces ``fes_powerplants_inc_tech_data.csv``
5. **CHP minimum operation profile** (``create_chp_p_min_pu_profile``): Derives the hourly ``p_min_pu`` time series for CHP generators from the aggregated heat demand dataset

**Capacity Distribution**:

When FES supplies only TO-level (Transmission Owner region) totals rather than GSP-level data, capacities are distributed to GSPs using a four-step hierarchy:

1. GSP-level FES data for the same carrier and year
2. GSP-level DUKES data for the same carrier
3. GSP-level FES + DUKES data for *all* carriers combined
4. Overall national distribution as a fallback

This hierarchy preserves known regional concentrations (e.g., offshore wind clusters near specific GSPs) while gracefully handling carriers with no fine-grained spatial data.

**Key Assumptions**:

- **No capacity expansion**: All generator capacities are fixed at the FES scenario projections for the modelled year; the optimiser dispatches within these limits
- **Uniform availability**: Monthly availability fractions are applied uniformly to all generators of the same carrier across all regions
- **CHP heat coupling**: CHPs are constrained to run above a heat-demand-derived floor but are not connected to a separate heat bus; heat demand is satisfied exogenously
- **Cost averaging**: FES fuel and VOM costs are averaged across scenarios due to naming changes between FES editions; this has a small effect (≤ ~10% for battery VOM) on marginal costs
- **European generators**: European countries use the PyPSA-Eur conventional generator database; availability and CHP constraints are not applied outside GB


.. seealso::

   **Related Documentation**:

   - :ref:`hydrogen-electrolysis` - Electrolysis as an additional flexible load
   - :doc:`ev` - EV demand and flexibility
   - :doc:`dispatch_redispatch` - How generators are dispatched in the optimisation
   - :doc:`configuration` - Full configuration reference

   **External Resources**:

   - `FES 2024 Data Workbook <https://www.neso.energy/publications/future-energy-scenarios-fes>`_ - Primary capacity data source
   - `DUKES 5.11 <https://www.gov.uk/government/statistics/electricity-chapter-5-digest-of-united-kingdom-energy-statistics-dukes>`_ - Existing installed capacity
   - `ENTSO-E Transparency Platform <https://transparency.entsoe.eu/>`_ - Outage data API
   - `PyPSA technology-data <https://github.com/PyPSA/technology-data>`_ - Technology costs and parameters
   - `Low Carbon Contracts Company <https://www.lowcarboncontracts.uk/>`_ - CfD generation register

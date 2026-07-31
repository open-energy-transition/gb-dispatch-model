..
  SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _system-ev:

##########################################
Electric Vehicle Subsystem
##########################################

This page describes the electric vehicle (EV) subsystem, its data sources, components, configuration, and implementation.

Overview
========

Electric vehicles are modelled as a combination of an inflexible unmanaged charging demand and flexible smart-charging demand-side response (DSR).
Optionally, vehicle-to-grid (V2G) capability allows stored EV energy to be discharged back to the electricity network.

The EV system comprises three main components:

- **Unmanaged EV demand**: Inflexible charging load representing EVs that charge at full capacity whenever plugged in
- **EV DSR store**: Flexible smart-charging buffer on a dedicated DSR bus, connected to the EV bus via charge and discharge links, that can shift charging load within a configurable daily window
- **V2G store and link**: Vehicle-to-grid capability allowing EVs to discharge energy back to the grid

.. graphviz::

   digraph {
      rankdir=LR;
      node [shape=box, style=filled];

      AC_bus [label="AC Bus", fillcolor="#B3D9FF", shape=ellipse, width=2, height=1.5, fixedsize=true];
      EV_bus [label="EV Bus", fillcolor="#FFD1DC", shape=ellipse, width=1.8, height=1.2, fixedsize=true];
      EV_DSR_bus [label="EV DSR Bus", fillcolor="#FFD1DC", shape=ellipse, width=1.8, height=1.2, fixedsize=true];
      EV_V2G_bus [label="EV V2G Bus", fillcolor="#E6D5FF", shape=ellipse, width=1.8, height=1.2, fixedsize=true];

      ev_load [label="EV Load\n(unmanaged)", fillcolor="#FFDAB9"];
      ev_dsr_store [label="EV DSR Store", fillcolor="#FFFACD"];
      v2g_store [label="V2G Store", fillcolor="#FFFACD"];

      AC_bus -> EV_bus [label="unmanaged load link"];
      EV_bus -> ev_load;
      EV_bus -> EV_DSR_bus [label="charge"];
      EV_DSR_bus -> EV_bus [label="discharge"];
      EV_DSR_bus -> ev_dsr_store [dir=both];
      EV_bus -> EV_V2G_bus [label="V2G feed-in\n(availability-constrained)"];
      EV_V2G_bus -> AC_bus [label="V2G discharge\n(availability-constrained)"];
      EV_V2G_bus -> v2g_store [dir=both];
   }

This modelling approach separates the EV system into a mandatory load (what would happen with no smart management) and optional flexibility layers (smart charging and V2G).


.. _ev-data-sources:

Data Sources
============

Great Britain Data
------------------

GB EV system data is derived from three sources:

- **FES ED5 (EV Demand)**: Annual EV unmanaged peak charging demand by scenario and year, used to size the unmanaged load (in GW, converted to MW)
- **FES FLX1 (Flexibility)**: Annual EV smart-charging DSR capacity and V2G capacity by scenario and year (in GW, converted to MW)
- **FES BB1 (Building Blocks)**: Regional GSP-level data used to disaggregate national EV totals to model regions

European Data
-------------

For European countries, EV demand is synthesised by scaling from GB data using annual electricity demand totals from the ENTSO-E energy balance (via ``energy_totals.csv``).
The scaling applies GB ratios between EV demand and total electricity consumption to European country demand figures.

**Demand Shape**
----------------

The temporal profile (demand shape) of EV charging is derived from German road traffic count data (KFZ dataset from PyPSA-Eur):

- Weekly traffic counts (averaged 2010–2015) define when EVs are on the road.
- A ``plug_in_offset`` shifts the traffic peak to represent when EVs arrive home and plug in.
- A rolling sum over ``charging_duration`` hours simulates the charging session duration.
- The resulting profile is normalised so it sums to 1 across the year.


.. _ev-components:

System Components
=================

.. _ev-unmanaged-demand:

Unmanaged EV Demand
--------------------

**PyPSA Component**: ``Load`` attached to the EV bus

The unmanaged demand represents the electricity that would be consumed if all EV owners charged at full capacity immediately upon arriving home, with no smart management.
It is an inflexible load — the model cannot reduce or reschedule it.

**Capacity**:

Peak unmanaged charging demand is extracted from the FES ED5 sheet for each scenario and year (in GW, converted to MW).
This provides the scaling factor applied to the normalised demand shape profile.

**Temporal Profile**:

The hourly profile is constructed from the normalised EV demand shape scaled to simultaneously satisfy two constraints — the annual energy total and the peak demand — using a two-step approach:

1. **Simple scaling** (tried first): scale the normalised shape directly by the annual target:

   .. math::

      P(t) = E_\text{annual} \times \hat{S}(t)

   where :math:`\hat{S}(t)` is the normalised shape (sums to 1). If :math:`\max P(t)` is within ``relative_peak_tolerance`` of :math:`P_\text{peak}`, this result is used.

2. **Power transformation** (used when simple scaling violates the peak constraint): a gamma exponent is fitted by minimising the combined peak and annual error:

   .. math::

      P(t) = E_\text{annual} \times \frac{\hat{S}(t)^\gamma}{\sum_t \hat{S}(t)^\gamma}

   Higher :math:`\gamma > 1` makes the profile peakier; lower :math:`\gamma < 1` flattens it.
   The optimal :math:`\gamma` is found via bounded scalar optimisation (``scipy.optimize.minimize_scalar``).

The shape is derived from traffic count data with a configurable plug-in offset (``ev.plug_in_offset``, default 2 hours) and charging duration (``ev.charging_duration``, default 4 hours).

**Regional Distribution**:

National EV peak demand is distributed to GB regions using the spatial pattern of EV demand from FES BB1 ("ev demand 1" and "ev demand 2" technology categories).
For European countries, EV demand is scaled from totals.


.. _ev-dsr:

EV Demand-Side Response (Smart Charging)
-----------------------------------------

**PyPSA Component**: ``Store`` on a dedicated EV DSR bus, connected to the EV bus via two ``Link`` components (charge and discharge)

The DSR store represents the flexibility available through smart charging — EVs that can defer charging to times when electricity is cheaper or the grid has spare capacity.
It acts as a virtual buffer: energy can be shifted forward or backward in time within the daily flexibility window.

**Capacity**:

DSR capacity (in MW) is derived from the FES FLX1 sheet using the ``"smart charging impact at peak"`` data item.
This represents the maximum reduction in peak demand achievable through smart charging at the system level.
Regional distribution uses the V2G spatial pattern as a proxy for EV adoption density.

**Operating Window**:

The DSR store can only shift load within a configurable daily window defined by ``fes.gb.flexibility.dsr_hours.ev_dsr`` (default: 08:00–06:00, i.e., flexible across most of the day).


.. _ev-v2g:

Vehicle-to-Grid (V2G)
----------------------

**PyPSA Component**: ``Store`` and ``Link`` attached to the EV V2G bus

V2G allows EVs to discharge stored energy back to the electricity network, acting as distributed battery storage.
This is modelled as a separate bus and store connected back to the main AC bus via a link.

**Capacity**:

V2G discharge capacity (in MW) is extracted from FES FLX1 using the ``"V2G maximum potential"`` data item.
Storage energy capacity (in MWh) is synthesised by multiplying the V2G power capacity by a fixed ratio:

.. math::

   E_\text{V2G} = P_\text{V2G} \times r_\text{storage}

where the ratio ``fes.gb.flexibility.v2g_storage_to_capacity_ratio`` (default 7.14) is derived from FES 2021 data and applied uniformly across all years.

**Regional Distribution**:

V2G capacity is distributed to GB regions using the spatial pattern of V2G capacity from FES BB1.


.. _ev-configuration:

Configuration
=============

EV behaviour is configured through the ``ev`` section of the configuration file.

From ``config/config.gb.2024.yaml``:

.. literalinclude:: ../../config/config.gb.2024.yaml
   :language: yaml
   :start-after: # [doc:ev-config-start]
   :end-before: # [doc:ev-config-end]

The demand profile transformation parameters control the two-step scaling used to match both the annual energy total and the peak demand (see :ref:`ev-unmanaged-demand`).
Where simple scaling satisfies the peak constraint (within ``relative_peak_tolerance``, default 5%), no transformation is applied.
Otherwise, ``lower_optimization_bound`` and ``upper_optimization_bound`` define the search range for the gamma exponent fitted by ``scipy.optimize.minimize_scalar``.


.. _ev-implementation-notes:

Implementation Notes
====================

Data Processing Workflow
------------------------

The EV system is built through a pipeline implemented across ``rules/gb-model/ev.smk`` and ``rules/gb-model/demand_and_dsr.smk``:

.. image:: img/ev_workflow.svg
   :align: center

.. note::
   The graph above was generated using::

      pixi run filtered_rulegraph \
      "resources/GB/gb-model/HT/regional_ev_v2g_storage_inc_eur.csv \
      resources/GB/gb-model/HT/ev_demand/2035.csv \
      resources/GB/gb-model/HT/regional_ev_dsr_inc_eur.csv \
      -w fes_scenario -w year \
      -f rules/gb-model/ev.smk \
      -s 10,8" \
      "doc/gb-model/img/ev_workflow.svg"


   The ``filtered_rulegraph`` task allows us to trim the full DAG to EV-related rules only.

1. **Demand shape** (``process_ev_demand_shape``): Builds normalised hourly EV charging profile from KFZ traffic data
2. **Peak demand table** (``create_ev_peak_charging_table``): Extracts annual EV unmanaged peak demand from FES ED5
3. **DSR and V2G capacity** (``create_flexibility_table``): Extracts EV smart charging DSR and V2G capacities from FES FLX1
4. **Regional distribution** (``synthesise_gb_regional_data``): Disaggregates national totals to GB regions using FES BB1 spatial pattern
5. **V2G storage** (``create_ev_v2g_storage_table``): Multiplies V2G capacity by the storage ratio to derive energy capacity
6. **Scaled demand profile** (``scaled_ev_demand_profile``): Combines annual/peak data with the demand shape to produce hourly profiles for each model year

.. _system-ev-assumptions:

Key Assumptions
---------------

- **Demand shape**: Derived from German traffic data (KFZ); the same shape is applied uniformly to all GB regions and European countries
- **Plug-in timing**: ``plug_in_offset`` hours after peak traffic activity (default 2 hours); ``charging_duration`` hours of active charging (default 4 hours)
- **V2G energy**: Storage capacity is a fixed multiple of V2G power capacity (ratio 7.14 from FES 2021); no temporal trend is applied
- **DSR window**: Smart charging flexibility is available across most of the day (default 08:00–06:00)
- **Profile scaling**: Simple annual scaling is applied first; a gamma power transformation is used only when simple scaling cannot satisfy the peak constraint within ``relative_peak_tolerance`` (default 5%)


.. seealso::

   **Related Documentation**:

   - :ref:`system-ev-summary` - EVs in the broader system representation
   - :ref:`data_sources_gb` - FES and other data sources
   - :ref:`model_config_gb` - Full configuration reference

   **External Resources**:

   - `FES 2024 Data Workbook <https://www.neso.energy/publications/future-energy-scenarios-fes>`_ - Primary data source
   - `PyPSA-Eur mobility profiles <https://github.com/PyPSA/pypsa-eur>`_ - KFZ traffic data source

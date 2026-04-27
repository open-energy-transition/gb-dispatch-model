..
  SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _system-dispatch_redispatch:

##########################################
Dispatch and Redispatch Modelling
##########################################

Overview
========

The GB model implements a two-stage optimization approach to capture realistic electricity market operations and network constraints:

1. **Unconstrained Dispatch** - Optimal market dispatch without internal GB transmission constraints
2. **Constrained Redispatch** - Adjustments to respect internal GB transmission constraints with bid/offer costs

This approach reflects real GB electricity market operations where the initial, day-ahead market dispatch is followed by a balancing market to resolve transmission constraints.

.. graphviz::
   :caption: Dispatch and Redispatch Process Flow

   digraph dispatch_flow {
      rankdir=TB;
      node [shape=box, style="rounded,filled"];

      // Input nodes
      fes_data [label="FES outputs\n(asset & flex capacities,\nannual demand, marginal costs)", fillcolor="#E8F4F8", shape=box];
      network_data [label="Network Data\n(transmission lines)", fillcolor="#E8F4F8", shape=box];
      timeseries_data [label="Profiles\n(renewables capacity factors,\ndemand profiles)", fillcolor="#E8F4F8"];
      strike_prices [label="CfD Strike Prices\n(Low carbon contracts)", fillcolor="#E8F4F8"];
      multipliers [label="Bid/Offer multipliers", fillcolor="#E8F4F8"];
      boundary_caps [label="ETYS Boundary\nCapabilities", fillcolor="#E8F4F8"];

      // Process nodes
      compose [label="Compose Network\n(attach generators, loads,\ninterconnectors, CHP)", fillcolor="#D4E6F1", fontsize=11];
      unconstrained [label="Unconstrained Dispatch\n(minimize system cost,\nno boundary constraints)", fillcolor="#AED6F1", fontsize=11];
      calc_bids [label="Calculate Bid/Offer\nProfiles\n(interconnectors,\ngenerators)", fillcolor="#D4E6F1", fontsize=11];
      prepare_constrained [label="Prepare Constrained\nNetwork\n(apply bid/offer costs)", fillcolor="#D4E6F1", fontsize=11];
      constrained [label="Constrained Redispatch\n(minimize redispatch cost,\nrespect ETYS boundaries)", fillcolor="#AED6F1", fontsize=11];
      calc_cost [label="Calculate Constraint\nCosts\n(difference in dispatch)", fillcolor="#D4E6F1", fontsize=11];

      // Output nodes
      unconstrained_result [label="Unconstrained Results\n(optimal dispatch,\nprices)", fillcolor="#C8E6C9"];
      constrained_result [label="Constrained Results\n(feasible dispatch,\nredispatch costs)", fillcolor="#C8E6C9"];
      constraint_cost [label="Total Constraint\nCost", fillcolor="#C8E6C9"];

      // Flow
      fes_data -> compose;
      network_data -> compose;
      timeseries_data -> compose;

      compose -> unconstrained [label="  no boundary\n  constraints", fontsize=10];
      unconstrained -> unconstrained_result;

      unconstrained_result -> calc_bids;
      strike_prices -> calc_bids;
      multipliers -> calc_bids;
      calc_bids -> prepare_constrained;

      compose -> prepare_constrained [label="  base network", fontsize=10];
      unconstrained_result -> prepare_constrained;

      prepare_constrained -> constrained [label="  with bid/offer\n  costs", fontsize=10];
      boundary_caps -> constrained [label="  ETYS\n  constraints", fontsize=10];

      constrained -> constrained_result;
      constrained_result -> calc_cost;
      calc_cost -> constraint_cost;
   }

Process
=======

Stage 1: Unconstrained (day-ahead) dispatch
-------------------------------------------

.. image:: img/dispatch.drawio.svg
    :class: full-width
    :align: center

The unconstrained dispatch represents the economically optimal dispatch without considering internal GB transmission constraints.
By default, we optimise with a perfect foresight at an hourly resolution for individual years.

**Objective**: Minimize total system cost

.. math::

    \min \sum_{t,g} MC_g \cdot p_{g,t} + \sum_{t,s} MC_s \cdot p_{s,t}

Where:

- :math:`MC_g` - Marginal cost of generator g (GBP/MWh)
- :math:`MC_s` - Marginal cost of storage unit s (GBP/MWh)
- :math:`p_{g,t}` - Power output of generator g at time t
- :math:`p_{s,t}` - Power output of storage unit s at time t

**Constraints**:

- Supply-demand balance at each bus
- Generator capacity limits
- Minimum generation levels (e.g., heat-led CHP constraints)
- Interconnector flow limits
- Storage state of charge limits

Two changes to the default PyPSA-Eur constraints are made during the unconstrained dispatch stage:

1. We remove all line and link limits within the GB network and we remove Kirchhoff-Voltage-Law (linearised power flow) constraints throughout the entire network.
   We apply the first to "copperplate" the GB bidding zone (no transmission constraints between intra-GB regions).
   We apply the second to ensure all intra-GB regional market prices align in each time period.
2. We set upper and lower bounds on annual nuclear power capacity factors, to limit their dispatchability.
   The bounds can be found within the configuration file.

**Output**:

- Optimal dispatch schedule
- Nodal electricity prices
- Interconnector flows

Stage 2: Constrained (balancing market) redispatch
--------------------------------------------------

.. image:: /gb-model/img/redispatch.drawio.svg
    :class: full-width
    :align: center

The constrained redispatch modifies the unconstrained dispatch to respect ETYS (Electricity Ten Year Statement) boundary capabilities, as well as individual line limits of the GB high-voltage transmission network.
By default, we optimise with a perfect foresight at an hourly resolution for individual years.

**Key Modifications**:

1. **Fixed dispatch & redispatch generators**

   The initial dispatch profiles for generators, GB->neighbour interconnectors, and storage units are all fixed to their optimal values from stage 1.
   We also re-impose the physical intra-GB transmission line and link limits (as well as applying the boundary constraints defined below).
   Generators, interconnectors, and storage units can deviate from their optimal dispatch via virtual ``up`` and ``down`` generators that we create for each asset.
   ``down`` generators can only remove energy from the system, ``up`` generators can only add energy to it.
   To these virtual generators we then apply redispatch (bid/offer) costs.

   .. note::

        This process requires updates to core PyPSA-Eur storage constraints to (a) fix their optimal dispatch correctly and (b) include the virtual generators in the storage energy balance constraint.

2. **Bid/Offer Costs Applied**

   Generators that deviate from unconstrained dispatch incur bid (decrease) or offer (increase) costs based on technology-specific multipliers.
   The modified marginal cost becomes:

   .. math::

       MC'_g = \begin{cases}
       MC_g \times offer\_multiplier & \text{if increase from unconstrained} \\
       MC_g \times bid\_multiplier & \text{if decrease from unconstrained} \\
       \end{cases}

3. **ETYS Boundary Constraints**

   Transmission boundaries between ETYS regions are constrained to their capabilities:

   .. math::

       \sum_{line \in boundary} flow_{line} \leq capability_{boundary}

   Several transmission lines cross each boundary.
   Some lines cross several boundaries, such as offshore HVDC lines that connect northern Scotland with central England.

4. **Rest of Europe**

   At this stage, the operation of assets in the rest of Europe is fixed, with no scope for deviation.
   In fact, we don't care about distinguishing between European countries, we only care about the bid/offer costs on each interconnector with GB.
   We assume that GB can fully deviate from the use of these interconnectors as defined in the initial dispatch stage.
   That is, if it is exporting at full capacity in the optimal dispatch, it can feasibly redispatch to importing at full capacity.
   We assume that the rest of Europe can absorb this change without a change of redispatch costs along the interconnectors.

   Since we only care about the redispatch costs on the interconnectors, we simplify the rest of Europe at this stage to a single node with an infinite store that can inject/remove any quantity of energy from the system at zero additional cost.

**Objective**: Minimize redispatch cost

.. math::

    \min \sum_{t,g} MC'_g \cdot (p_{g,t} - p^{unconstrained}_{g,t})

Bid/Offer Profile Calculation
------------------------------

.. seealso::
    For details on the different system components defined here, see: :ref:`system_overview_gb`.

**Conventional Generators**

Multipliers are based on historical bids and offers.

For conventional assets (e.g., fossil-fuelled power plants), the bid cost is a premium minus the asset's short-run marginal cost.
This premium reflects a penalty paid by the system operator in lieu of profit lost by the asset operator.
The offer cost is the asset's short-run marginal cost plus a premium, to reflect additional profits gained in the balancing market.

**Low-Carbon Generators**

For low-carbon generators with Contracts for Difference (CfD), the bid/offer costs are based on the strike price (the subsidy received by the operator) minus the asset short-run marginal cost.
This value is applied as a cost to the system whether the asset operator is increasing or decreasing output in redispatch.
It reflects a cost to the system operator to account for lost subsidy (bids) and to pay for additional generation (offers).

**Other Generators & Storage Units**

Some generators have no historical bid/offer data, nor do we have subsidy information (e.g., geothermal plants).
For these plants, we use their short run marginal costs as their bid/offer costs.
Therefore, when bidding, the system receives a revenue equal to the cost of generating that power.

We similarly only apply short-run marginal costs as the bid/offer costs for storage units (pumped hydro and batteries), noting that the opportunity cost of this energy in future time periods is already accounted for in the optimisation.

**Demand-side response**

Demand-side response (DSR) can be redispatched and zero bid/offer costs are applied.
As with storage units, we assume DSR aggregators would be receiving revenue from capacity markets and so would not need additional payments in balancing markets.

**Interconnectors**:

Interconnector bid/offer costs are given as an hourly timeseries profile, derived from the unconstrained dispatch.
There are six options available to interconnectors during redispatch, as outlined in `the National Grid Long-term Market and Constraint Modelling methodology document <https://www.nationalgrid.com/sites/default/files/documents/Long-term%20Market%20and%20Network%20Constraint%20Modelling.pdf>`_.

1. Currently importing into GB and offering to import more.
2. Currently importing into GB and bidding to import less or switch to exporting.
3. Currently not in use ("floating") and offering to import.
4. Currently not in use ("floating") and bidding to export.
5. Currently exporting from GB and offering to export less or switch to importing.
6. Currently exporting from GB and bidding to export more.

These can be collapsed into four profiles as we consider (1) & (3) and (4) & (6) to be equivalent.

- **Offering to increase imports**: Treated as generator offering power

  .. math::

      Cost_{offer}^{import+} = |\lambda_{GB} - \lambda_{neighbor}| + offer_{neighbor} \cdot (1 + \eta_{loss})

- **Bidding to decrease imports**: Treated as generator bidding to reduce

  .. math::

      Cost_{bid}^{import-} = \lambda_{GB} - \lambda_{neighbor} \cdot (1 + \eta_{loss}) - bid_{neighbor} \cdot (1 + \eta_{loss})

- **Offering to decrease exports**: Treated as generator offering power

  .. math::

      Cost_{offer}^{export-} = \lambda_{neighbor} \cdot (1 - \eta_{loss}) - \lambda_{GB} - offer_{neighbor} \cdot (1 - \eta_{loss})

- **Bidding to increase exports**: Treated as generator bidding to reduce

  .. math::

      Cost_{bid}^{export+} = |\lambda_{GB} - \lambda_{neighbor}| - bid_{neighbor} \cdot (1 - \eta_{loss})

Where:

- :math:`\lambda_{GB}` - GB marginal price (GBP/MWh)
- :math:`\lambda_{neighbor}` - Neighbor's marginal price (GBP/MWh)
- :math:`offer_{neighbor}` - Neighbor's marginal plant offer cost (GBP/MWh)
- :math:`bid_{neighbor}` - Neighbor's marginal plant bid cost (GBP/MWh)
- :math:`\eta_{loss}` - Interconnector loss rate (fraction)

The neighbours marginal plant is selected as the plant anywhere in Europe with marginal cost closest to the neighbour's marginal price.
This is because any plant could be setting the marginal price, via intra-European interconnectors.
The bid/offer costs of this plant are calculated as in GB, using the same multipliers.
However, since we do not have information on renewables subsidy mechanisms in each European country, if a low-carbon plant is setting the price, we do not set bids/offers to a strike price but rather to the plant's marginal cost.

Analysing results
-----------------

Result files can be found in the ``results`` directory.
Each year in the horizon of interest is optimised separately and a file for each year is stored in the following directory structure:

.. code::

    results/{run}/networks/
    ├── unconstrained_clustered/              # Stage 1 results
    │   └── {year}.nc
    └── constrained_clustered/                # Stage 2 results
        └── {year}.nc

In addition, once all runs have completed, a ``results/{run}/constraint_costs.csv`` will be available, which gives a single value constraint cost for the entire modelling horizon, plus the final year's cost duplicated N times to simulate costs until the end of current infrastructure asset lifetimes.
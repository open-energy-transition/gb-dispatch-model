.. SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
.. SPDX-FileCopyrightText: Contributors to gb-dispatch-model
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _gb_wildcards:

#########
Wildcards
#########

It is easy to run the GB dispatch model for multiple scenarios using the wildcards feature of ``snakemake``.
Wildcards allow a rule to be generalised, to produce all files that follow a regular expression pattern which e.g. defines one particular scenario.
One can think of a wildcard as a parameter that shows up in the input/output file names of the ``Snakefile``, determining which rules to run, what data to retrieve and what files to produce.

.. note::
    Detailed explanations of how wildcards work in ``snakemake`` can be found in the
    `relevant section of the documentation <https://snakemake.readthedocs.io/en/stable/snakefiles/rules.html#wildcards>`__.

Key wildcards
=============

Some wildcards are more important to users than others.
Here, we describe the wildcards you will most likely utilise in your modelling work.

.. _fes_scenario_wc:

The ``{fes_scenario}`` wildcard
-------------------------------

The ``{fes_scenario}`` wildcard specified for which FES scenario to run the workflow.
By default all FES2024 scenarios are available.
The wildcard accepts shorthands for each scenario, which are defined in the configuration:

.. code:: yaml

    HT: Holistic Transition
    CF: Counterfactual
    EE: Electric Engagement
    HE: Hydrogen Evolution

.. _year_wc:

The ``{year}`` wildcard
-----------------------

The ``{year}`` wildcard specifies for which future year to run the workflow.
Any year in the (inclusive) range defined in the configuration item ``redispatch.year_range_incl`` can be chosen,
with the limit that the first year must be the FES year + 1 (2025 for FES2024) and the final year must be 2050.

Other wildcards
===============

The following wildcards are used within specific ``snakemake`` rules, but it is unlikely that you will need to directly generate files that rely upon them.
They are describe here for completeness.

.. _fes_sheet_wc:

The ``{fes_sheet}`` wildcard
----------------------------

The ``{fes_sheet}`` wildcard specifies a FES workbook table to process, based on the Excel workbook sheet name (case sensitive).
Any sheets with a configuration item in ``fes.sheet_config`` can be processed with the ``{fes_sheet}`` wildcard.

.. _demand_type_wc:

The ``{demand_type}`` wildcard
------------------------------


The ``{demand_type}`` wildcard specifies which demand to process, based on those defined in the configuration as the keys under ``fes.gb.demand.Technology Detail``.

.. _flexibility_type_wc:

The ``{flexibility_type}`` wildcard
-----------------------------------

The ``{flexibility_type}`` wildcard specifies which demand-side response component to process, based on those defined in the configuration as the keys under ``fes.gb.flexibility.carrier_mapping``.

.. _regional_reference_wc:

The ``{regional_reference}`` wildcard
-------------------------------------

The ``{regional_reference}`` wildcard specifies which data to regionalise, based on those defined in the configuration as the keys under ``fes.gb.regional_distribution_reference``.

These datasets do not have a regional equivalent defined in the FES, only aggregated national values.
The :mod:`regionalisation workflow <synthesise_gb_regional_data>` uses the regional FES data referenced in ``fes.gb.regional_distribution_reference`` as a proxy to regionalise national data.

.. _h2_dataset_wc:

The ``{h2_dataset}`` wildcard
-----------------------------

The ``{h2_dataset}`` wildcard specifies which European hydrogen subsystem data to synthesise.
Choices are: ``non_networked_electrolysis_demand_annual``, ``H2_storage_capacity``, ``grid_electrolysis_capacities``.

Since no information is given regarding a European hydrogen subsystem in the FES, the :mod:`synthesis script <synthesise_eur_H2_data>` uses equivalent GB data to estimate capacities / demand based on the relative quantity of networked electrolysis demand between GB and each European country (derived from the TYNDP model results for Europe).

.. _report_year_wc:

The ``{report_year}`` wildcard
------------------------------

The ``{report_year}`` wildcard specified which NESO transmission availability report to process, of those listed in the configuration item ``transmission_availability.years``.

.. _transmission_zone_wc:

transmission_zone
-----------------

The ``{transmission_zone}`` wildcard specified which GB transmission zone to process availability data for from NESO availability reports, of those listed in the configuration item ``transmission_availability.intra_gb.zones`` or ``transmission_availability.inter_gb.zones``.
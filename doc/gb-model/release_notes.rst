
..
  SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: gb-dispatch-model contributors

  SPDX-License-Identifier: CC-BY-4.0

##########################################
Release Notes
##########################################

Unreleased
==========

* Created new OSM pre-built network, available at https://zenodo.org/records/18712831, to include updates made in upstream PyPSA-Eur (incl. ignoring unbuilt lines) (#237).
* Fix reaching Elexon API request limit when running snakemake with multiple cores by forcing all cores to be used for the data fetching rule.
* Adding `async` for API request that fetches the mapping of Elexon BMU units to a fueltype (#225).
* Allow for boundary capabilities to increase in line with outputs from an ETYS report (configurable, defaults to True) (#219).
* Account for time aggregation in the nuclear annual operation custom constraints (#233)

v0.2.0 (2026-02-13)
===================

This release of the GB dispatch model is configured for use with FES2024 data.
Due to the differences in data structure compared to FES2021, it is not possible to use this version for FES years prior to 2023.
Instead, development for <= FES2022 should branch off from v0.1.0.

This version was synchronised with the upstream PyPSA-Eur repository on 2026-02-10.

* Publish documentation on readthedocs.org.
* Calculate bid/offer multipliers using data from Elexon (#161, #209).
* Add BritNed as an existing interconnector in default config (#210).
* Use correct currency (GBP, not EUR) in docs.
* Clean SPDX copyright text where an outdated repository name was being used.
* Add docs page detailing our dispatch/redispatch methodology (#158).
* Update custom busmap definition to have one per configured run name (``data/gb-model/custom-busmap.csv`` -> ``data/gb-model/custom-busmap/<run-name>.csv``) (#207).
* Extend workflow to run all FES scenarios in parallel (#215).
* Fix monthly outage calculations by using the `entsoe-py` package to collect outages and DUKES data to define current capacities (#204).
* Impose nuclear capacity factor range to enforce nuclear power plant usage where it would otherwise have unrealistically low generation / high dispatchability (#201 and #202).
* Fix storage flows in redispatch (both for the original asset and the ramp up/down assets) (#196).
* Use `Generator` component for all ramp up/down assets to simplify optimisation problem.
* Update to FES2024, with the side effect of fixing load shedding issues (#163, #155, #156).

v0.1.0 (2026-01-23)
===================

This is the initial release of the GB dispatch model, configured for use with FES2021 data.
It was synchronised with the upstream PyPSA-Eur repository on 2026-01-12.

* Calculate total constraint costs as a sum of all redispatch runs (#189).
* * Update to PyPSA ``>=v1`` (#160).
* Remove KVL constraints from unconstrained GB-EUR model and set transmission losses to 0 (#186).
* Identify marginal generator in Europe when non-neighbouring countries are part of the network (#179).
* Add system representation graphic and details to the documentation (#157).
* Use heat pump uptake trend to define shape of heating mix technological change (#130).
* Avoid double counting the impact of heat demand on both baseline electricity and heat demand profile shapes.
* Move resistive heating profile impact to baseline electricity (since FES heating demand only covers heat pump electricity consumption).
* Replaced EUR buses with a single EUR bus for faster re-dispatch solve (#174)
* Set bus_id for virtual buses using line_id to ensure stable custom busmap mapping (#166)
* Enable custom busmap to prevent incorrect clustering of offshore buses (#159)
* Add interconnector bids and offers to constrained network (#153)
* Calculate interconnector bids and offers (#151)
* Add config option to set load shedding to the most expensive powerplant plus a small delta.
* Enable custom busmap to prevent incorrect clustering of offshore buses (#159)
* Fix H2 demands in Europe using TYNDP H2 NT scenario demands (#152)
* Add bid/offers for generators (#147)
* Distribute all loads into their own buses with independently linked DSR stores
* Add residential heat demand DSR, including district heating flexibility (as it cannot be separated)
* Process low carbon register CfD strike prices for use in redispatch
* Define independent DSR hours for each demand type (#144)
* Disassociate EV DSR and EV V2G components (#140)
* Add DC links into boundary constraints (#136)
* Added flexibility to the baseline electricity and electrified i&c heat demand through demand-side management (#133).
* Added generator and interconnector availability fraction as `p_max_pu` timeseries parameter in the network.
* Fixed missing European neighbour data in EV datasets (#123).
* Add interconnectors to network.
* Add boundary capability constraints to GB model (#131).
* Merge Shetland (region 30) and Northern Ireland (region 31) to other regions (#117).
* Add demands to pypsa Network (#102, #70, #120).
* Limit GB model to ``clustered`` clusters.
* Add EV to pypsa Network (#114)
* Tablulated regional unmanaged EV charging demand data (#112).
* Add demands to pypsa Network (#102)
* Added ETYS report boundary capabilities extractor & linked PyPSA bus-pair lines to these boundaries (#9).
* Added config version for updating the system boundaries to the subset defined in the ETYS report.
* Prepared unmanaged EV charging demand profile shape based on traffic data (#104).
* Tabulated regional EV storage data (#101).
* Extract transmission unavailability from NESO system performance report PDF (internal and interconnectors) (#40, #38).
* Prepared regional flexibility data for EV and demand-side management (DSM) for base electricity (#97).
* Prepared FES costing worksheets (#62).
* Rule to generalize creation of load profiles for different demand types (#93)
* Tabulated flexibility data for EV and demand-side management (DSM) for base electricity (#91).
* Changed base year to 2012 (#92)
* Enabled overwriting onshore clustering with custom GB shapes (#89).
* Prepared transport demand profile shape which will be used for EV demand profile (#84)
* Merged isolated North-West islands regions (`GB 89` and `GB 90`) into mainland region (#90).
* Tabulated regional baseline electricity demand data (#85).
* Tabulated regional EV demand data (#83).
* Tabulated hydrogen related data including demand, supply, storage, and generation capacities (#73).
* Tabulated interconnector capacities between GB regions and neighbouring countries (#10).
* Tabulated monthly GB powerplant fractional availability profiles (#71).
* Remove unnecessary output in `compose_networks` rule that causes error (#2)
* Tabulated regional powerplant capacities for GB (#4) with direct transmission-level / unconnected capacities proportionally distributed to GSPs (#66, #77)
* Tabulated EU country level aggregated powerplant capacities (#33)
* Add rule 'retrieve_unavailability_data' to Snakemake workflow for fetching unavailability data from ENTSO-E. (#43)
* Increase number of HTTP download retries to mitigate against Zenodo file retrieval timeouts.
* Keep all retrieved data locally by default to reduce time spent re-downloading data on every run.
* Add FES workbook data download and sheet extraction rule (#50).
* Restructured documentation (#27).
* Added modelling methodology documentation (#20).
* Added GB custom geographic boundary rule and script (#13).

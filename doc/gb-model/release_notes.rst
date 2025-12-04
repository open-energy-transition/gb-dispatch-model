
..
  SPDX-FileCopyrightText: Open Energy Transition gGmbH and contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: gb-dispatch-model contributors

  SPDX-License-Identifier: CC-BY-4.0

##########################################
Release Notes
##########################################

Upcoming Release
================

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

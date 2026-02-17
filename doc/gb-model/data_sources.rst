..
  SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: gb-dispatch-model contributors

  SPDX-License-Identifier: CC-BY-4.0

#############
Data Sources
#############

gb-dispatch-model is compiled from a variety of data sources.
The following table provides an overview of the data sources used exclusively in gb-dispatch-model.
For data sources used in PyPSA-Eur, see `this page <../data_sources.html>`_.
Different licenses apply to the data sources.

---------------------------------
The Future Energy Scenarios (FES)
---------------------------------

`The FES <https://www.neso.energy/publications/future-energy-scenarios-fes>`_ is the primary data source for defining the model, both for GB and other European countries.
By default, we use the 2024 FES data workbook.
Tables from the workbook we use are:

- BB1: Building Block Data
- BB2: Building Block Metadata
- FLX1: Flexibility data table
- ES2: European electricity supply data table
- ED3: Gas and heat demand summary
- ED5: Road transport summary
- WS1: Whole system & gas supply

We also use the same cost assumptions given by an earlier FES report, available in a separate dataset linked to `a 2023 report <https://assets.publishing.service.gov.uk/media/6556027d046ed400148b99fe/electricity-generation-costs-2023.pdf>`_.
No cost assumption dataset exists specifically for FES2024.

------------------------------------------
The Digest of UK Energy Statistics (DUKES)
------------------------------------------
From `DUKES <https://www.gov.uk/government/statistics/electricity-chapter-5-digest-of-united-kingdom-energy-statistics-dukes>`_, we access existing capacities (Table 5.11).
This is used to help distribute unallocated future capacities to GB regions, based on the relative capacity of technologies already existing.
It replaces the equivalent existing power plant dataset computed in PyPSA-Eur due to being more comprehensive.

-----------------------------------------
The ELectricity Ten Year Statement (ETYS)
-----------------------------------------
The `ETYS <https://www.neso.energy/publications/electricity-ten-year-statement-etys>`_ is a GB annual report which identifies bottlenecks in the transmission network, defined across network system boundaries.
We use the report to:

- Define our model regions, combining the cuts made by the system boundaries to create regions.
  As boundaries often intersect, these regions are usually a combination of several boundaries.
- Define the "current" boundary capabilities (grid transfer capacities - GTCs) with which we will scale PyPSA line limits.

-----------------
GSP coordinates
-----------------
GB `grid-supply point (GSP) coordinates <https://api.neso.energy/dataset/963525d6-5d83-4448-a99c-663f1c76330a/resource/41fb4ca1-7b59-4fce-b480-b46682f346c9/download/fes2021_regional_breakdown_gsp_info.csv>`_ are obtained from the NESO website.
This is used to assign geographic coordinates to powerplants extracted from the FES workbook and to connect geographically varying data from PyPSA-Eur (heat pump COPs, heat demand profiles) to GSP-level supply/demand data.

Several GSPs referenced in the FES workbook cannot be matched to the coordinate dataset.
Where this is the case, a manual match / coordinate assignment has been configured.


-----------------------
FES European data
-----------------------
The FES workbook ES2 sheet (see above) provides us with powerplant and demand data of other countries in Europe.

Note that the split of demand data into types and all other FES datasets (e.g., load flexibility potential) are not available for these countries.
Accordingly, we create synthetic European datasets using the relative magnitude of total annual demand in a European country compared to GB annual demand.

---------------
Interconnectors
---------------
Electricity transmission interconnectors between GB regions and neighbouring countries are based on distinct projects considered in the FES (table 9, `FES modelling methods <https://www.neso.energy/document/199916/download>`_).
We combine those projects manually to create a total GB interconnector capacity curve from 2021-2041 that matches the curves given in the FES workbook, sheet SV.37.
The GB region to which those projects connect is based on geolocating the connecting transformer as defined in the NESO `interconnector register <https://www.neso.energy/data-portal/interconnector-register>`_.
For projects not in the register (since some outdated projects are no longer considered), we have used the respective `TYNDP <https://tyndp.entsoe.eu/>`_ project data sheet to estimate their GB onshoring coordinates.
Project definitions and our manually defined start dates for them are user-configurable.

.. note::
  No reasonable combination of projects perfectly matches the FES results.
  However, the combination culminating in the FES results is not publicly available, so the projects we choose is an opinionated assumption.

------------------------------
Generator availability profile
------------------------------
We define a monthly availability profile for GB generator types for which we have historical data on outages.
We access historical outage data from the `ENSTO-E transparency platform <https://transparency.entsoe.eu/outage-domain/r2/unavailabilityOfProductionAndGenerationUnits/show>`_, spanning a configurable number of years.
We group these outages into PyPSA-Eur generator types ("carriers") and use this to calculate the daily relative availability of each type, by comparing the lost capacity due to forced/planned outages against the total national capacity of that type.
We derive total capacity from the base PyPSA-Eur powerplant dataset.
We finally collapse this multi-year, daily availability profile into a single monthly profile by calculating a monthly grouped average availability.
For instance, if there is a 80% availability in the first half of June for only one of the five assessed historical years, the final June availability will be 98%.

---------------------------------
Transmission availability profile
---------------------------------
Transmission unavailability, as a percentage of hours in a month, is taken from the NESO `System Performance Reports <https://www.neso.energy/industry-information/industry-data-and-reports/system-performance-reports>`.
This covers unavailability for both internal GB transmission (split by transmission operator) and interconnectors (per interconnector).

----------------
GB Hydrogen data
----------------
All hydrogen subsystem data such as demand, supply, storage, and generation capacities are sourced from the FES workbook (see above).

----------------------
European Hydrogen data
----------------------
No hydrogen subsystem data is provided for European countries except for Hydrogen → Electricity conversion technology capacities.

We use a data dump of the `ENTSO-E TYNDP <https://zenodo.org/records/14230568>`_ to define the future hydrogen demands in non-GB countries.
To reduce intermediate data processing requirements in this workflow, we have separately prepared the corresponding input data using the `Open-TYNDP project <https://github.com/open-energy-transition/open-tyndp>`_ workflow, and stored outputs for the NT scenario here as ``data/gb-model/tyndp_h2_demand.csv``.
To interpolate to 2030, we use historical "clean" hydrogen demand from the `European clean hydrogen observatory <https://observatory.clean-hydrogen.europa.eu/hydrogen-landscape/end-use/hydrogen-demand>`_.

For the remainder (e.g. electrolyser capacity, storage capacity), we synthesise data based on trends seen in GB.

-----------------------
Electric vehicles (EVs)
-----------------------
EV annual data is extracted from the FES workbook (see above).
The EV charging demand shape is quantified by shifting traffic rate data used in PyPSA-Eur with a configured plug-in offset and then applying charging duration.
This is the *unmanaged* EV charging profile.
Flexibility via smart charging and vehicle-to-grid (V2G) technologies is given by the FES as a reduction in peak unmanaged EV demand.
We apply this as the capacity of these flexibility mechanisms.

In Europe, for lack of data from FES, we synthesise all EV subsystem data following the same methodology we apply in other parts of the workflow.

--------------------------------
Baseline electricity demand data
--------------------------------
Annual baseline electricity demand data is extracted from the FES workbook per grid supply point.
This is mapped onto PyPSA-Eur hourly profiles, which are historical national load profiles for the configured weather year.

------------------------------------
DSR flexibility for base electricity
------------------------------------
Demand-side response (DSR) flexibility data for base electricity (residential and I&C) is extracted from the FES workbook.

------------------------------
GB regional data distributions
------------------------------
The FES workbook contains regional distributions of several data points, down to the level of grid supply points.
Where this data is not available, we use the regional distribution of a close match dataset to distribute.
For instance, residential base electricity demand-side response capacity is distributed according to the distribution of annual base electricity demand.

-------------------
Low carbon register
-------------------
The `low carbon register <https://www.lowcarboncontracts.uk/our-schemes/contracts-for-difference/register/>`_ provides historical Contracts for Difference (CfD) strike price data.
We use the average strike price data from all historical years before the first model run year to define the renewable generator offers.
These are calculated relative to the GB wholesale market price as given by the solved unconstrained model output.

--------------------
Elexon BMU Fuel Map
-------------------
The `Elexon BMU Fuel Map <https://www.elexon.co.uk/documents/data/operational-data/bmu-fuel-type/>` provides a mapping of the balancing mechanism units (BMU) to their respective fuel types.
This information is required to calculate bid / offer multiplier carrier-wise for redispatch. This is a static copy in the repo and can be updated by navigating to `<https://bmrs.elexon.co.uk/generation-by-fuel-type?>`
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
Here, we use the 2021 FES data workbook.
Tables from the workbook we use are:

- BB1: Building Block Data
- BB2: Building Block Metadata
- SV.34: Installed BECCS generation capacity (GW)
- CV.10: Annual hydrogen demand for home heating
- CV.33: Annual energy demand for Road Transport Leading the Way
- CV.53: Annual hydrogen demand for the industrial sector
- CV.54: Annual hydrogen demand for the commercial sector
- SV.20: Leading the Way Hydrogen supply (TWh)
- ED1: Electricity demand summary
- FL.6: Hydrogen Storage Capacity Requirements

In addition, we use FES 2023 to detailed annual hydrogen demand for other sectors.
Tables from the workbook we use are:

- WS1: Whole System & Gas Supply

We also use the same cost assumptions as given by the FES, available in a separate dataset linked to `a 2023 report <https://assets.publishing.service.gov.uk/media/6556027d046ed400148b99fe/electricity-generation-costs-2023.pdf>`_.

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
This is used to assign lat, lon to powerplants extracted from the FES workbook

-----------------------
FES European data
-----------------------
The `FES European dataset <https://api.neso.energy/dataset/bd83ce0b-7b1e-4ff2-89e8-12d524c34d99/resource/6563801b-6da4-46e7-b147-3d81c0237779/download/fes2023_es2_v001.csv>`_ is used to retrieve powerplant and demand data of other countries in Europe.

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

-------------
Hydrogen data
-------------
All hydrogen related data such as demand, supply, storage, and generation capacities are sourced from the FES workbooks as detailed above.

--------------
EV demand data
--------------
Electric vehicle (EV) demand data is extracted from the FES-2021 workbook table BB1.
EV demand profile shape is prepared based on transport demand profile shape of PyPSA-Eur.
EV charging demand shape is computed by shifting traffic rate data of PyPSA-Eur with plug-in offset and applying charging duration.
Unmanaged EV charging demand is extracted from FES-2021 workbook table FL.11.

--------------------------------
Baseline electricity demand data
--------------------------------
Baseline electricity demand data is extracted from FES-2021 workbook table BB1.

-------------------
EV flexibility data
-------------------
Electric vehicle (EV) flexibility data is extracted from the FES-2021 workbook table FLX1.
Energy storage capacity of EVs are obtained by interpolating EV storage data from FL.14 sheet of FES-2021 workbook with V2G peak capacity from FLX1 sheet.
Storage data is regionally disaggregated based on EV flexibility data.
Regional distribution of V2G and smart charging flexibility is based on V2G distribution provided in BB1 sheet of FES-2021.

-------------------
DSM flexibility for base electricity
-------------------
Demand-side management (DSM) flexibility data for base electricity (residential and I&C) is extracted from the FES-2021 workbook table FLX1.
Regional distribution of residential demand side response (DSR) flexibility is based on Baseline demand distribution provided in BB1 sheet of FES-2021,
while regional distribution of services DSR flexibility is based on I&C Flexibility (TouT) distribution provided in BB1 sheet of FES-2021.

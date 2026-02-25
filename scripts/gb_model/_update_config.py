# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT


from collections.abc import Hashable
from typing import Annotated, Any, Literal, Self, TypeVar, Union

from annotated_types import Len
from pydantic import Field, field_validator, model_validator

from scripts.lib.validation.config._base import ConfigModel, ConfigUpdater
from scripts.lib.validation.config._schema import ConfigSchema

T = TypeVar("T", bound=Hashable | list)
TwoEntryList = Annotated[list[T], Len(min_length=2, max_length=2)]


class GBBaseConfig(ConfigModel):
    """Base configuration for GB model. This is the starting point for the GB config schema, which is built by applying the GBConfigUpdater to this base config."""

    model_config = {"extra": "forbid"}  # forbid any fields not defined in the schema


class FESSheetConfig(GBBaseConfig):
    """FES sheet configuration."""

    model_config = {"extra": "allow"}  # any pandas read_excel kwargs can be included
    rename: dict[str, str] = Field(
        default_factory=dict, description="Column rename mappings"
    )


class CostsGBConfig(GBBaseConfig):
    """GB-specific cost configuration."""

    sheet_config: dict[str, FESSheetConfig] = Field(
        alias="sheet-config",
        description="FES costing workbook sheet configurations. "
        "Any pandas read_excel kwargs can be included for each sheet.",
        default_factory=dict,
    )
    GBP_to_EUR: float = Field(gt=0, description="GBP to EUR exchange rate", default=1)
    GBP_to_USD: float = Field(gt=0, description="GBP to USD exchange rate", default=1)
    relevant_cost_columns: list[str] = Field(
        min_length=1, description="List of relevant cost columns", default_factory=list
    )
    marginal_cost_columns: list[str] = Field(
        min_length=1, description="List of marginal cost columns", default_factory=list
    )
    add_cols: dict[str, dict[str, Any]] = Field(
        default_factory=dict, description="Additional cost columns configuration"
    )
    default_characteristics: dict[str, dict[str, str | float]] = Field(
        default_factory=dict, description="Default characteristics for cost data"
    )
    carrier_gap_filling: dict[str, dict[str, str]] = Field(
        default_factory=dict, description="Carrier gap filling mappings"
    )
    fes_VOM_carrier_mapping: dict[str, str] = Field(
        default_factory=dict, description="FES variable O&M carrier mapping"
    )
    fes_fuel_carrier_mapping: dict[str, str] = Field(
        default_factory=dict, description="FES fuel carrier mapping"
    )
    voll: float = Field(ge=0, description="Value of lost load in £/MWh", default=0)


class LowCarbonRegisterConfig(GBBaseConfig):
    """Low carbon register configuration."""

    carrier_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from low carbon register carriers to model carriers",
    )


class CHPConfig(GBBaseConfig):
    """Simplified CHP configuration for GB model."""

    enable: bool = Field(description="Enable simplified CHP constraints", default=True)
    heat_to_power_ratio: float = Field(
        gt=0, description="Heat-to-power ratio (c_b coefficient)", default=1
    )
    min_operation_level: float = Field(
        ge=0, le=1, description="Minimum operation level when running", default=0
    )
    shutdown_threshold: float = Field(
        ge=0,
        le=1,
        description="Heat demand threshold below which CHPs can shut down completely",
        default=0,
    )


class URLsConfig(GBBaseConfig):
    """URLs for GB data sources."""

    model_config = {"populate_by_name": True}

    gb_etys_boundaries: str = Field(
        alias="gb-etys-boundaries",
        description="URL for ETYS boundary GIS data",
        default="",
    )
    transmission_availability_2020: str | None = Field(
        default=None, alias="transmission-availability-2020"
    )
    transmission_availability_2021: str | None = Field(
        default=None, alias="transmission-availability-2021"
    )
    transmission_availability_2022: str | None = Field(
        default=None, alias="transmission-availability-2022"
    )
    transmission_availability_2023: str | None = Field(
        default=None, alias="transmission-availability-2023"
    )
    transmission_availability_2024: str | None = Field(
        default=None, alias="transmission-availability-2024"
    )
    transmission_availability_2025: str | None = Field(
        default=None, alias="transmission-availability-2025"
    )
    fes_workbook: str = Field(
        alias="fes-workbook",
        description="URL for FES workbook for the given FES year",
        default="",
    )
    fes_costing_workbook: str = Field(
        alias="fes-costing-workbook",
        description="URL for FES costing workbook",
        default="",
    )
    dukes_5_11: str = Field(
        alias="dukes-5.11", description="URL for DUKES 5.11 data", default=""
    )
    gsp_coordinates: str = Field(
        alias="gsp-coordinates", description="URL for GSP coordinates", default=""
    )
    etys: str = Field(description="URL for ETYS report", default="")
    etys_chart_data: str = Field(
        alias="etys-chart-data",
        description="URL for future boundary capabilities (from ETYS chart data). Only used if 'etys.use_future_capacities' is True",
        default="",
    )
    low_carbon_contracts: str = Field(
        alias="low-carbon-contracts",
        description="URL for low carbon contracts actual CfD Generation data",
        default="",
    )
    eur_H2_demand_today: str = Field(
        description="URL for European hydrogen demand data from the Clean Hydrogen Observatory",
        default="",
    )


class RegionSplitConfig(GBBaseConfig):
    """Configuration for splitting regions."""

    region: int = Field(description="Region ID to split")
    type: Literal["vertical", "horizontal"] = Field(description="Type of split")
    coordinate: float = Field(description="Coordinate for the split")


class RegionMergeGroupConfig(GBBaseConfig):
    """Configuration for merging region groups."""

    id: int | str = Field(description="Merged region ID")
    merge: list[int | str] = Field(min_length=1, description="List of regions to merge")
    TO: Literal["NGET", "SPTL", "SHETL", "N-IRL"] = Field(
        description="Transmission owner"
    )


class RegionOperationsConfig(GBBaseConfig):
    """Manual region merging and splitting configuration."""

    area_loss_tolerance_percent: float = Field(
        default=0.01,
        ge=0,
        le=100,
        description="Percentage of area loss tolerated when splitting",
    )
    min_region_area: float = Field(
        default=100000.0, ge=0, description="Minimum area of a region in square meters"
    )
    filter_boundaries_using_capabilities: bool = Field(
        description="Filter boundaries using capabilities", default=False
    )
    splits: list[RegionSplitConfig] = Field(
        default_factory=list, description="Region split configurations"
    )
    merge_groups: list[RegionMergeGroupConfig] = Field(
        default_factory=list,
        min_length=1,
        description="Region merge group configurations",
    )
    add_group_to_neighbour: dict[str, list[str]] = Field(
        default_factory=dict, description="Map regions to neighbouring countries"
    )


class ETYSBoundaryLineConfig(GBBaseConfig):
    """Configuration for ETYS boundary line."""

    bus0: int | str = Field(description="Starting GB bus ID (without preceding `GB`)")
    bus1: int | str = Field(description="Ending GB bus ID (without preceding `GB`)")


class ETYSConfig(GBBaseConfig):
    """Configuration for ETYS boundaries."""

    future_capacities_sheet_name: str = Field(
        description="Name of the sheet to use in the downloaded ETYS future capacities workbook",
        default="ETYS 2022 Chart Data",
    )
    use_future_capacities: bool = Field(
        description="Whether to use future boundary capacities (a.k.a. capabilities) from ETYS chart data, instead of today's capacities.",
        default=True,
    )
    manual_future_capacities: dict[str, dict[int, float]] = Field(
        description="Manual future boundary capacities, to fill gaps in the ETYS chart data, if necessary. Top-level key is the boundary name, then key-value pairs of year and capacity in MW.",
        default_factory=dict,
    )


class EntsoeUnavailabilityConfig(GBBaseConfig):
    """ENTSO-E unavailability data configuration."""

    start_date: str = Field(
        description="Start date for data retrieval", default="2020-01-01"
    )
    end_date: str = Field(
        description="End date for data retrieval", default="2025-12-31"
    )
    bidding_zones: list[str] = Field(
        min_length=1,
        description="List of bidding zones to consider",
        default_factory=list,
    )
    business_types: list[Literal["planned", "forced"]] = Field(
        min_length=1, description="Business types to retrieve", default_factory=list
    )
    max_request_days: int = Field(
        default=365, gt=0, description="Maximum days per API request"
    )
    max_unavailable_days: int = Field(
        default=365, gt=0, description="Maximum days to consider an outage"
    )
    carrier_mapping: dict[str, str] = Field(
        description="Carrier mapping from ENTSO-E to model carriers",
        default_factory=dict,
    )


class TransmissionAvailabilityZoneConfig(GBBaseConfig):
    """Transmission availability zone configuration."""

    zones: list[str] = Field(
        min_length=1, description="List of zones", default_factory=list
    )
    sample_hourly: bool = Field(
        description="Whether to construct an hourly availability profile using random sampling",
        default=True,
    )


class TransmissionAvailabilityConfig(GBBaseConfig):
    """Transmission availability configuration."""

    years: list[int] = Field(
        min_length=1, description="Years to process", default_factory=list
    )
    intra_gb: TransmissionAvailabilityZoneConfig = Field(
        description="Intra-GB transmission zones",
        default_factory=TransmissionAvailabilityZoneConfig,
    )
    inter_gb: TransmissionAvailabilityZoneConfig = Field(
        description="Inter-GB transmission zones",
        default_factory=TransmissionAvailabilityZoneConfig,
    )
    random_seeds: dict[str, int] = Field(
        description="Random seeds for each zone", default_factory=dict
    )

    @model_validator(mode="after")
    def validate_random_seeds(self) -> Self:
        """Validate that random seeds are provided for all zones."""
        all_zones = set(self.intra_gb.zones + self.inter_gb.zones)
        if missing_seeds := all_zones - self.random_seeds.keys():
            raise ValueError(
                f"Random seeds must be provided for all zones. Missing: {missing_seeds}"
            )
        return self


class DukesSheetConfig(GBBaseConfig):
    """DUKES sheet configuration."""

    sheet_name: str = Field(description="Name of the Excel sheet", default="")
    usecols: str = Field(description="Columns to use", default="")
    skiprows: int = Field(ge=0, description="Number of rows to skip", default=0)
    index_col: list[int] = Field(description="Index columns", default_factory=list)
    header: int = Field(ge=0, description="Header row", default=0)


class DukesConfig(GBBaseConfig):
    """DUKES configuration."""

    model_config = {"populate_by_name": True}

    sheet_config: DukesSheetConfig = Field(
        alias="sheet-config",
        description="Sheet configuration",
        default_factory=DukesSheetConfig,
    )
    carrier_mapping: dict[str, dict[str, str]] = Field(
        description="Carrier mapping configuration. Top level keys are column names, second level maps from column values to carrier names.",
        default_factory=dict,
    )
    set_mapping: dict[str, dict[Any, str]] = Field(
        description="Set mapping configuration. Top level keys are column names, second level maps from column values to set names.",
        default_factory=dict,
    )


class GridSupplyPointsConfig(GBBaseConfig):
    """Grid supply points configuration."""

    model_config = {"populate_by_name": True}

    manual_mapping: dict[str, str] = Field(
        default_factory=dict,
        description="Manual GSP name mappings, from FES workbook names to GSP GIS file names.",
    )
    fill_lat_lons: dict[str, dict[str, float]] = Field(
        alias="fill-lat-lons",
        default_factory=dict,
        description="Manual GSP coordinate assignments, to add latitudes and longitudes for GSPs missing from the GIS data. Keys are FES workbook GSP names.",
    )


class FESDemandConfig(GBBaseConfig):
    """FES demand configuration."""

    model_config = {"populate_by_name": True}

    Technology_Detail: dict[str, list[str]] = Field(
        alias="Technology Detail",
        description="Grouping from FES workbook BB2 Technology Detail column to model carriers",
        default_factory=dict,
    )
    heat: dict[
        str,
        dict[
            str,
            Literal["ASHP", "GSHP", "resistive"]
            | list[Literal["ASHP", "GSHP", "resistive"]],
        ],
    ] = Field(
        description="Mapping from FES workbook heating technologies to intermediate model carriers.",
        default_factory=dict,
    )
    bus_suffix: dict[str, str] = Field(
        description="Bus suffixes to apply in the PyPSA network for different demand types",
        default_factory=dict,
    )


class FESFlexibilityConfig(GBBaseConfig):
    """FES flexibility configuration."""

    carrier_mapping: dict[str, dict[str, str | list[str]]] = Field(
        description="Carrier mapping for flexibility. Top level keys are GB model DSR carriers, second level maps from column names to column value(s).",
        default_factory=dict,
    )
    dsr_hours: dict[str, TwoEntryList[int]] = Field(
        description="Demand side response hours for each carrier",
        default_factory=dict,
    )
    carrier_suffix: dict[str, str] = Field(
        description="Carrier suffix mappings", default_factory=dict
    )
    v2g_storage_to_capacity_ratio: float = Field(
        gt=0, description="V2G storage to capacity ratio", default=1
    )


class FESRegionalDistributionReference(GBBaseConfig):
    """FES regional distribution reference."""

    source: str | list[str] = Field(
        description="FES2024 BB2 'Technology Detail' value(s) to use as reference for regional distribution",
        default="",
    )
    name: str = Field(
        description="PyPSA parameter name to use to define the result (e.g. `p_nom` or `p_set`)",
        default="p_nom",
    )


class FESGBConfig(GBBaseConfig):
    """FES GB-specific configuration."""

    carrier_mapping: dict[str, dict[str, str]] = Field(
        description="Carrier mappings based on values in different columns. Top level keys are column names, second level maps from column values to carrier names. The order of the top level keys determines the priority of the mappings, with earlier keys taking precedence over later ones when there are overlapping mappings.",
        default_factory=dict,
    )
    set_mapping: dict[str, dict[str, Literal["CHP", "PP", "Store"]]] = Field(
        description="Set mappings based on values in different columns. Top level keys are column names, second level maps from column values to set names. The order of the top level keys determines the priority of the mappings, with earlier keys taking precedence over later ones when there are overlapping mappings.",
        default_factory=dict,
    )
    demand: FESDemandConfig = Field(
        description="Demand configuration", default_factory=FESDemandConfig
    )
    flexibility: FESFlexibilityConfig = Field(
        description="Flexibility configuration", default_factory=FESFlexibilityConfig
    )
    regional_distribution_reference: dict[str, FESRegionalDistributionReference] = (
        Field(
            description="Regional distribution reference configuration",
            default_factory=dict,
        )
    )


class FESEURConfig(GBBaseConfig):
    """FES European configuration."""

    carrier_mapping: dict[str, dict[str, str]] = Field(
        description="Carrier mappings based on values in different columns. Top level keys are column names, second level maps from column values to carrier names. The order of the top level keys determines the priority of the mappings, with earlier keys taking precedence over later ones when there are overlapping mappings.",
        default_factory=dict,
    )
    set_mapping: dict[str, dict[str, Literal["CHP", "PP", "Store"]]] = Field(
        description="Set mappings based on values in different columns. Top level keys are column names, second level maps from column values to set names. The order of the top level keys determines the priority of the mappings, with earlier keys taking precedence over later ones when there are overlapping mappings.",
        default_factory=dict,
    )
    totals_to_demand_groups: dict[str, list[str]] = Field(
        description="Keys are gb-model demand carriers, values are energy total column names.",
        default_factory=dict,
    )
    add_data_reference: dict[str, str] = Field(
        description="Reference data to use to synthesise datasets for which we have no European data.",
        default_factory=dict,
    )
    load_shedding_cost_above_marginal: float | None = Field(
        default=None,
        description="Marginal cost to apply to European load shedding above the marginal cost of the most expensive European generation technology. This is used to ensure that load shedding only occurs after all generation technologies have been fully utilised, but to stop the resulting marginal prices from being excessively high.",
    )


class FESHydrogenDataSelection(GBBaseConfig):
    """FES hydrogen data selection filter."""

    all_supply: dict[str, str | list[str]] = Field(
        description="Filter to get all hydrogen supply data from the FES workbook. Keys are column names, values are column value(s) to select.",
        default_factory=dict,
    )
    all_demand: dict[str, str | list[str]] = Field(
        description="Filter to get all hydrogen demand data. Keys are column names, values are column value(s) to select.",
        default_factory=dict,
    )
    non_networked_supply: dict[str, str | list[str]] = Field(
        description="Filter to get all hydrogen non-networked supply data from the FES workbook. Keys are column names, values are column value(s) to select.",
        default_factory=dict,
    )
    storage: dict[str, str | list[str]] = Field(
        description="Filter to get all hydrogen storage data. Keys are column names, values are column value(s) to select.",
        default_factory=dict,
    )


class FESHydrogenConfig(GBBaseConfig):
    """FES hydrogen configuration."""

    data_selection: FESHydrogenDataSelection = Field(
        description="FES workbook data aggregation filters for hydrogen",
        default_factory=FESHydrogenDataSelection,
    )
    electrolysis_efficiency: float = Field(
        gt=0,
        description="Non-networked electrolysis efficiency (MWh[H2]/MWh[electricity])",
        default=0.7,
    )


class FESConfig(GBBaseConfig):
    """Future Energy Scenarios (FES) configuration."""

    fes_year: int = Field(description="FES data year", default=2024)
    scenario_mapping: dict[str, str] = Field(
        description="Mapping from FES scenario shorthand (key) to FES scenario long name (value). The shorthand will be used in the output data directories and may be referred to when shorthand is used in data files (e.g. ETYS chart data).",
        default_factory=dict,
    )
    default_set: Literal["PP", "CHP", "Store"] = Field(
        description="Default PyPSA network set identifier", default="PP"
    )

    sheet_config: dict[str, FESSheetConfig] = Field(
        alias="sheet-config",
        description="FES workbook sheet configurations. "
        "Any pandas read_excel kwargs can be included for each sheet.",
        default_factory=dict,
    )
    gb: FESGBConfig = Field(
        description="GB-specific FES configuration", default_factory=FESGBConfig
    )
    eur: FESEURConfig = Field(
        description="European FES configuration", default_factory=FESEURConfig
    )
    hydrogen: FESHydrogenConfig = Field(
        description="Hydrogen configuration", default_factory=FESHydrogenConfig
    )


class EVDemandProfileTransformation(GBBaseConfig):
    """EV demand profile transformation parameters."""

    lower_optimization_bound: float = Field(
        gt=0, description="Lower bound for gamma optimization", default=0
    )
    upper_optimization_bound: float = Field(
        gt=0, description="Upper bound for gamma optimization", default=float("inf")
    )
    relative_peak_tolerance: float = Field(
        ge=0, le=1, description="Relative peak tolerance (fraction)", default=0.0
    )
    relative_energy_tolerance: float = Field(
        ge=0, le=1, description="Relative energy tolerance (fraction)", default=0.0
    )


class EVConfig(GBBaseConfig):
    """Electric vehicle configuration."""

    plug_in_offset: int = Field(
        ge=0,
        description="Hours after traffic peak when EVs are assumed to plug in",
        default=0,
    )
    charging_duration: int = Field(
        gt=0, description="Charging duration in hours", default=1
    )
    ev_demand_profile_transformation: EVDemandProfileTransformation = Field(
        description="Demand profile transformation parameters",
        default_factory=EVDemandProfileTransformation,
    )


class TYNDPInfo(GBBaseConfig):
    """TYNDP project information."""

    id: int | None = Field(default=None, description="TYNDP ID, if known")
    year: list[int] = Field(
        default_factory=list, description="TYNDP years in which the option appears"
    )


class InterconnectorConfig(GBBaseConfig):
    """Individual interconnector configuration."""

    name: str = Field(description="Interconnector name")
    neighbour: str = Field(
        description="Neighbouring country to which the interconnector connects"
    )
    capacity_mw: float = Field(gt=0, description="Capacity in MW")
    tyndp: TYNDPInfo = Field(description="TYNDP information", default_factory=TYNDPInfo)
    lat: float = Field(
        ge=-90,
        le=90,
        description="Latitude of GB substation where the interconnector onshores",
    )
    lon: float = Field(
        ge=-180,
        le=180,
        description="Longitude of GB substation where the interconnector onshores",
    )


class InterconnectorsConfig(GBBaseConfig):
    """Interconnectors configuration."""

    options: list[InterconnectorConfig] = Field(
        description="List of interconnector configurations", default_factory=list
    )
    plan: dict[int, list[str]] = Field(
        description="Interconnector deployment plan by year", default_factory=dict
    )

    @model_validator(mode="after")
    def validate_plan_names(self) -> "InterconnectorsConfig":
        """Validate that all names in plan exist in options."""
        option_names = {opt.name for opt in self.options}
        planned_options: set = set()
        planned_options.update(*self.plan.values())
        unknown_options = planned_options.difference(option_names)
        if unknown_options:
            raise ValueError(
                f"Unknown interconnector options in plan: {unknown_options}."
            )
        return self


class ElexonConfig(GBBaseConfig):
    """Elexon API configuration."""

    technology_mapping: dict[str, str] = Field(
        description="Mapping from Elexon BMU technology names to model carrier names",
        default_factory=dict,
    )
    years: list[int] = Field(
        default_factory=list, description="Years to retrieve data for"
    )
    api_bmu_fuel_map: bool = Field(
        default=False,
        description="Boolean to decide source of Elexon BMU unit -> fuel mapping. If True, data is fetched using API; otherwise, it uses `data/gb-model/BMUFuelType.xlsx`.",
    )
    max_concurrent_requests: int = Field(
        default=4,
        description="Maximum number of concurrent requests to Elexon API. May require adjustment if API rate limits are hit.",
    )


class RedispatchConfig(GBBaseConfig):
    """Redispatch config."""

    year_range_incl: TwoEntryList[int] = Field(
        description="Inclusive year range [start, end]",
        default_factory=list,
    )
    constraint_cost_extra_years: int = Field(
        default=20,
        gt=0,
        description="number of years to extend the constraint cost calculation beyond the final year (using the data from the final year)",
    )
    unconstrain_lines_and_links: bool = Field(
        default=True,
        description="Whether to unconstrain lines and links in the redispatch model. "
        "This will Set s_nom (p_nom) to infinity for lines (links) between GB regions to ensure only boundary capabilities are bounding, not physical line limits.",
    )

    elexon: ElexonConfig = Field(
        description="Elexon API configuration", default_factory=ElexonConfig
    )

    @field_validator("year_range_incl")
    @classmethod
    def validate_year_range(cls, v: list[int]) -> list[int]:
        """Validate that year range is ordered correctly."""
        if len(v) == 2 and v[0] > v[1]:
            raise ValueError("Start year must be less than or equal to end year")
        return v


class TimeAggregationConfig(GBBaseConfig):
    """Time aggregation configuration."""

    method: Literal["segment", "downsample", "resample"] = Field(
        description="Time aggregation method to use, of those available in PyPSA."
    )
    parameters: dict = Field(
        description="Parameters for given PyPSA time aggregation method.",
        default_factory=dict,
    )


class GBConfigUpdater(ConfigUpdater):
    name: str = "gb"

    def _update_clustering(self) -> type[ConfigSchema]:
        """Updates the clustering configuration to include the 'gb_shapes' mode."""
        clustering_config = self.base_config().clustering.__class__
        mode_config = clustering_config.model_fields["mode"]

        current_description = mode_config.description or ""
        new_description = (
            current_description + " (extra) gb_shapes: GB model region shapes."
        )
        new_list = Literal[mode_config.annotation, "gb_shapes"]

        clustering_schema = self._apply_updates(
            __base__=clustering_config,
            mode=(new_list, Field(mode_config.default, description=new_description)),
        )
        new_schema = self._apply_updates(
            clustering=(clustering_schema, Field(default_factory=clustering_schema))
        )

        return new_schema

    def _update_scenario(
        self, existing_schema: type[ConfigSchema]
    ) -> type[ConfigSchema]:
        """Updates the scenario configuration to include the generic 'clusters' wildcard."""
        scenario_config = existing_schema().scenario.__class__
        cluster_config = scenario_config.model_fields["clusters"]

        current_description = cluster_config.description or ""
        new_description = (
            current_description + " (extra) clustered: User-defined clusters scenario."
        )
        new_list = list[
            Union[*cluster_config.annotation.__args__[0].__args__, Literal["clustered"]]
        ]

        scenario_schema = self._apply_updates(
            __base__=scenario_config,
            clusters=(
                new_list,
                Field(
                    default_factory=cluster_config.default_factory,
                    description=new_description,
                ),
            ),
        )
        new_schema = self._apply_updates(
            __base__=existing_schema,
            scenario=(scenario_schema, Field(default_factory=scenario_schema)),
        )

        return new_schema

    def _add_new_sections(
        self, existing_schema: type[ConfigSchema]
    ) -> type[ConfigSchema]:
        """Adds new sections to the configuration schema."""

        # We'll automatically create fields for new sections that are simple BaseModels.
        # Anything more complex, we define the field manually in this dictionary.
        new_sections = {
            "fes_costs": CostsGBConfig,
            "low_carbon_register": LowCarbonRegisterConfig,
            "chp": CHPConfig,
            "urls": URLsConfig,
            "target_crs": (
                str,
                Field(
                    description="Target coordinate reference system (e.g., EPSG:27700)",
                    default="EPSG:27700",
                ),
            ),
            "region_operations": RegionOperationsConfig,
            "etys": ETYSConfig,
            "entsoe_unavailability": EntsoeUnavailabilityConfig,
            "transmission_availability": TransmissionAvailabilityConfig,
            "dukes_5_11": (
                DukesConfig,
                Field(
                    alias="dukes-5.11",
                    description="DUKES 5.11 data configuration",
                    default_factory=DukesConfig,
                ),
            ),
            "grid_supply_points": GridSupplyPointsConfig,
            "fes": FESConfig,
            "ev": EVConfig,
            "interconnectors": InterconnectorsConfig,
            "redispatch": RedispatchConfig,
            "time_aggregation": (
                list[TimeAggregationConfig],
                Field(
                    description="List of time aggregation configurations to apply sequentially. "
                    "If multiple configurations are provided, they will be applied in the order they appear in the list. "
                    "See `PyPSA documentation <https://docs.pypsa.org/latest/api/networks/cluster/#pypsa.Network.cluster.temporal>`_ for details on the available time aggregation methods and their parameters.",
                    default_factory=list,
                ),
            ),
        }
        new_sections_processed = {
            k: v
            if isinstance(v, tuple)
            else (v, Field(description=v.__doc__, default_factory=v))
            for k, v in new_sections.items()
        }
        extra_doc_str = (
            "\n\nIncludes additional GB-specific configuration sections:\n"
            + "\n".join(
                f"- `{name}`: {field_info.description}"
                for name, (_, field_info) in new_sections_processed.items()
            )
        )
        new_schema = self._apply_updates(
            __base__=existing_schema,
            __doc__=existing_schema.__doc__ + extra_doc_str
            if existing_schema.__doc__
            else None,
            **new_sections_processed,
        )
        return new_schema

    def update(self) -> type[ConfigSchema]:
        # To update and existing config item, we need it's most recent state, as defined in `self.base_config`
        new_schema = self._update_clustering()
        new_schema = self._update_scenario(new_schema)
        new_schema = self._add_new_sections(new_schema)
        return new_schema

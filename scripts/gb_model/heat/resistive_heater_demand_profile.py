# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Historical electrified heat demand profile processor.
"""

import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from scripts._helpers import configure_logging, set_scenario_config

logger = logging.getLogger(__name__)


def get_electrified_heat_demand_profile(
    heat_demand_shape: xr.Dataset,
    heat_demand_annual: pd.DataFrame,
) -> pd.DataFrame:
    """
    Scale the heat demand shape to create an electrified heat demand profile.

    Args:
        heat_demand_shape (xr.Dataset): Profile of hourly heat demand based on heating degree days.
        annual_heat_demand (pd.DataFrame): annual heat demand totals.

    Returns:
        pd.DataFrame: Heat demand shape scaled to match the annual demand totals.
    """
    heat_demand_shape = heat_demand / heat_demand.sum("snapshots")

    heat_demand_profile = (
        (heat_demand_shape * heat_demand_annual.to_xarray())
        .to_array()
        .sum("variable")
        .to_pandas()
    )

    return heat_demand_profile


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(Path(__file__).stem, clusters="clustered")
    configure_logging(snakemake)
    set_scenario_config(snakemake)

    # Load the file path
    heat_demand = xr.open_dataset(snakemake.input.heat_demand_shape).rename(
        {"node": "bus"}
    )

    energy_totals = (
        pd.read_csv(snakemake.input.energy_totals, index_col="name")
        .mul(1e6)  # TWh to MWh
        .rename_axis(index="bus")
    )
    resistive_heater_demand: dict = {}
    for sector in ["residential", "services"]:
        water_to_space_ratio = energy_totals[f"electricity {sector} water"] / (
            energy_totals[f"electricity {sector} space"]
            + energy_totals[f"electricity {sector} water"]
        )
        df = (
            pd.read_csv(
                snakemake.input[f"{sector}_heat_demand_annual"],
                index_col=["bus", "year", "technology"],
            )
            .xs(
                (int(snakemake.wildcards.year), "resistive"),
                level=("year", "technology"),
            )
            .squeeze()
        )
        resistive_heater_demand[f"{sector} water"] = df * water_to_space_ratio
        resistive_heater_demand[f"{sector} space"] = df * (1 - water_to_space_ratio)

    historical_heat_demand_annual = energy_totals.filter(
        regex="electricity (residential|services)"
    ).rename(columns=lambda x: x.removeprefix("electricity").strip())
    historical_heat_demand = get_electrified_heat_demand_profile(
        heat_demand, historical_heat_demand_annual
    )

    historical_heat_demand.rename_axis(index="time").to_csv(snakemake.output.historical)

    future_heat_demand = get_electrified_heat_demand_profile(
        heat_demand, pd.DataFrame(resistive_heater_demand)
    )
    future_heat_demand.rename_axis(index="time").to_csv(snakemake.output.future)

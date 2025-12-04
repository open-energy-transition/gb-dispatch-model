# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Aggregate monthly transmission unavailability across reports and spread to hourly.
"""

import logging

import numpy as np
import pandas as pd

from scripts._helpers import configure_logging, get_snapshots, set_scenario_config

logger = logging.getLogger(__name__)


ZONES = {
    "National Grid Electricity": "NGET",
    "Scottish Power": "SPTL",
    "Scottish Hydro Electric": "SHETL",
}


def _read_inputs(paths: list[str]) -> pd.DataFrame:
    """Load inputs and concatenate into a single DataFrame."""
    df = pd.concat(
        (pd.read_csv(p, usecols=["geography", "month", "Total"]) for p in paths),
        ignore_index=True,
    )
    df["Total"] = pd.to_numeric(df["Total"], errors="coerce")
    return df


def _monthly_means(df: pd.DataFrame) -> pd.DataFrame:
    """Get monthly means per zone and interconnectors over all years."""
    # Zones pivot to monthly means and rename to short codes
    zones = df[df["geography"].isin(ZONES)].pivot_table(
        index="month", columns="geography", values="Total", aggfunc="mean"
    )
    zones = zones.rename(columns=ZONES).reindex(range(1, 13))

    # Interconnectors as a single monthly mean across all connectors
    inter = df.loc[~df["geography"].isin(ZONES)].groupby("month")["Total"].mean()
    inter = inter.reindex(range(1, 13))

    # Compose single monthly dataframe
    monthly = zones.assign(interconnectors=inter)
    return monthly


def _sample_hourly(
    monthly_pct: pd.DataFrame, index: pd.DatetimeIndex, seeds: dict[str, int]
) -> pd.DataFrame:
    """Vectorised per-month sampling using shuffling."""
    base = pd.DataFrame(index=index)
    base["month"] = index.month
    counts = base.groupby("month").size()

    def __sample(series):
        n_unavailable = (
            (fraction_unavailable * counts).loc[series.name].round().astype(int)
        )
        sampled = pd.Series(0, index=series.index, dtype="int8")
        sampled.loc[sampled.sample(n=n_unavailable, random_state=rng).index] = 1
        return sampled

    dfs = {}
    for col in monthly_pct.columns:
        fraction_unavailable = monthly_pct[col].fillna(0) / 100.0
        rng = np.random.default_rng(seeds[col])
        dfs[col] = base.groupby("month", group_keys=False).month.apply(__sample)
    return pd.concat(dfs.values(), keys=dfs.keys(), axis=1).rename_axis(index="time")


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("process_transmission_availability")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    raw = _read_inputs(snakemake.input.unavailability)
    monthly_pct = _monthly_means(raw)[snakemake.params.zones]

    # Log compact monthly means
    msg = monthly_pct.round(2).to_dict(orient="index")
    logger.info("Monthly unavailability (%%): %s", msg)

    time_index = get_snapshots(
        snakemake.config["snapshots"], drop_leap_day=False, freq="h"
    )
    if snakemake.params.sample_hourly:
        df = _sample_hourly(monthly_pct, time_index, snakemake.params["random_seeds"])
    else:
        df = 1.0 - (monthly_pct / 100.0)
    df.to_csv(snakemake.output.csv)

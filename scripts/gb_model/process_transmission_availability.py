# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Aggregate monthly transmission unavailability across reports and sample to hourly.

This script reads multiple CSVs extracted from NESO monthly transmission availability PDFs.
It averages the monthly total unavailability percentages across years for the three GB
transmission operators and for all interconnectors (aggregated), then samples a
deterministic hourly 0/1 unavailability series per category for the configured
snapshots period.
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
    monthly = zones.assign(INTERCONNECTORS=inter)
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
    for col, seed in seeds.items():
        fraction_unavailable = monthly_pct[col].fillna(0) / 100.0
        rng = np.random.default_rng(seed)
        dfs[col] = base.groupby("month", group_keys=False).month.apply(__sample)
    return pd.concat(dfs.values(), keys=dfs.keys(), axis=1).rename_axis(index="time")


def process_transmission_availability(
    inputs: list[str], snapshots_cfg: dict, random_seeds: dict[str, int]
) -> pd.DataFrame:
    """Process transmission availability inputs to produce hourly availability."""
    raw = _read_inputs(inputs)
    monthly_pct = _monthly_means(raw)

    # Log compact monthly means
    msg = monthly_pct.round(2).to_dict(orient="index")
    logger.info("Monthly unavailability (%%): %s", msg)

    time_index = get_snapshots(snapshots_cfg, drop_leap_day=False, freq="h")
    return _sample_hourly(monthly_pct, time_index, random_seeds)


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake("process_transmission_availability")

    configure_logging(snakemake)
    set_scenario_config(snakemake)

    hourly_df = process_transmission_availability(
        snakemake.input.unavailability,
        snakemake.config["snapshots"],
        snakemake.params["random_seeds"],
    )
    hourly_df.to_csv(snakemake.output.csv)

# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: MIT

"""Tests for CHP utilities module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa
import pytest
import xarray as xr

from scripts.gb_model.chp_utils import (
    attach_chp_constraints,
    calculate_chp_minimum_operation,
    identify_chp_powerplants,
)


@pytest.fixture
def powerplants_with_chp():
    """Sample powerplants dataframe with CHP and non-CHP plants."""
    return pd.DataFrame(
        {
            "bus": ["GB0", "GB0", "GB1", "GB1"],
            "carrier": ["CCGT", "CCGT", "nuclear", "CCGT"],
            "set": ["CHP", "PP", "PP", "CHP"],
            "p_nom": [100.0, 150.0, 500.0, 80.0],
        },
        index=["GB0 CCGT-2030-0", "GB0 CCGT-2030-1", "GB1 nuclear-2030-0", "GB1 CCGT-2030-0"],
    )


@pytest.fixture
def powerplants_no_set():
    """Sample powerplants dataframe without 'set' column."""
    return pd.DataFrame(
        {
            "bus": ["GB0", "GB0", "GB1"],
            "carrier": ["CCGT", "CCGT", "nuclear"],
            "p_nom": [100.0, 150.0, 500.0],
        },
        index=["GB0 CCGT-2030-0", "GB0 CCGT-2030-1", "GB1 nuclear-2030-0"],
    )


@pytest.fixture
def heat_demand_netcdf(tmp_path):
    """Create a sample heat demand NetCDF file."""
    # Create time series (24 hours)
    times = pd.date_range("2030-01-01", periods=24, freq="h")
    buses = ["GB0", "GB1"]

    # Create synthetic heat demand profiles
    # GB0: Strong daily pattern with peak at hour 18
    # GB1: Lower demand with peak at hour 20
    hour_of_day = np.arange(24)
    gb0_profile = 100 * (1 + 0.5 * np.sin((hour_of_day - 6) * np.pi / 12))
    gb1_profile = 50 * (1 + 0.3 * np.sin((hour_of_day - 8) * np.pi / 12))

    # Create dataset
    ds = xr.Dataset(
        {
            "residential water": xr.DataArray(
                np.vstack([gb0_profile * 0.2, gb1_profile * 0.2]).T,
                coords={"snapshots": times, "node": buses},
                dims=["snapshots", "node"],
            ),
            "residential space": xr.DataArray(
                np.vstack([gb0_profile * 0.5, gb1_profile * 0.5]).T,
                coords={"snapshots": times, "node": buses},
                dims=["snapshots", "node"],
            ),
            "services water": xr.DataArray(
                np.vstack([gb0_profile * 0.1, gb1_profile * 0.1]).T,
                coords={"snapshots": times, "node": buses},
                dims=["snapshots", "node"],
            ),
            "services space": xr.DataArray(
                np.vstack([gb0_profile * 0.2, gb1_profile * 0.2]).T,
                coords={"snapshots": times, "node": buses},
                dims=["snapshots", "node"],
            ),
        }
    )

    # Save to file
    filepath = tmp_path / "heat_demand.nc"
    ds.to_netcdf(filepath)
    return str(filepath)


@pytest.fixture
def simple_network():
    """Create a simple PyPSA network for testing."""
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=24, freq="h"))

    # Add buses
    n.add("Bus", "GB0", carrier="AC")
    n.add("Bus", "GB1", carrier="AC")

    return n


class TestIdentifyCHPPowerplants:
    """Tests for identify_chp_powerplants function."""

    def test_identify_chp_plants(self, powerplants_with_chp):
        """Test that CHP plants are correctly identified."""
        chp_plants = identify_chp_powerplants(powerplants_with_chp)

        assert len(chp_plants) == 2
        assert all(chp_plants["set"] == "CHP")
        assert list(chp_plants.index) == ["GB0 CCGT-2030-0", "GB1 CCGT-2030-0"]
        assert chp_plants["p_nom"].sum() == 180.0  # 100 + 80

    def test_no_set_column(self, powerplants_no_set):
        """Test handling when 'set' column is missing."""
        chp_plants = identify_chp_powerplants(powerplants_no_set)

        assert chp_plants.empty

    def test_no_chp_plants(self):
        """Test when no CHP plants exist."""
        ppl = pd.DataFrame(
            {
                "bus": ["GB0", "GB1"],
                "carrier": ["CCGT", "nuclear"],
                "set": ["PP", "PP"],
                "p_nom": [100.0, 500.0],
            }
        )
        chp_plants = identify_chp_powerplants(ppl)

        assert chp_plants.empty


class TestCalculateCHPMinimumOperation:
    """Tests for calculate_chp_minimum_operation function."""

    def test_basic_calculation(self, heat_demand_netcdf):
        """Test basic minimum operation calculation."""
        buses = pd.Index(["GB0", "GB1"])
        heat_to_power_ratio = 1.5
        min_operation_level = 0.3

        p_min_pu = calculate_chp_minimum_operation(
            heat_demand_netcdf, buses, heat_to_power_ratio, min_operation_level
        )

        # Check structure
        assert isinstance(p_min_pu, pd.DataFrame)
        assert len(p_min_pu) == 24
        assert list(p_min_pu.columns) == ["GB0", "GB1"]

        # Check values are in valid range
        assert (p_min_pu >= 0.0).all().all()
        assert (p_min_pu <= 1.0).all().all()

        # Check that minimum operation level is applied
        # Where heat exists, p_min_pu should be >= min_operation_level
        non_zero = p_min_pu[p_min_pu > 0]
        assert (non_zero >= min_operation_level).all().all()

    def test_shutdown_threshold(self, heat_demand_netcdf):
        """Test that CHPs can shut down when heat demand is very low."""
        buses = pd.Index(["GB0", "GB1"])

        # Use very high heat-to-power ratio to create low p_min_pu values
        # This should trigger shutdown threshold logic
        p_min_pu = calculate_chp_minimum_operation(
            heat_demand_netcdf,
            buses,
            heat_to_power_ratio=10.0,  # Very high ratio
            min_operation_level=0.3,
        )

        # At some hours, p_min_pu should be exactly 0 (shutdown)
        assert (p_min_pu == 0.0).any().any()

    def test_missing_bus(self, heat_demand_netcdf):
        """Test handling of bus not in heat demand data."""
        buses = pd.Index(["GB0", "GB1", "GB2"])  # GB2 doesn't exist
        heat_to_power_ratio = 1.5
        min_operation_level = 0.3

        p_min_pu = calculate_chp_minimum_operation(
            heat_demand_netcdf, buses, heat_to_power_ratio, min_operation_level
        )

        # Should have all buses, with GB2 filled with zeros
        assert "GB2" in p_min_pu.columns
        assert (p_min_pu["GB2"] == 0.0).all()

    def test_heat_to_power_ratio_effect(self, heat_demand_netcdf):
        """Test that heat-to-power ratio affects p_min_pu correctly."""
        buses = pd.Index(["GB0"])

        # Higher ratio = lower p_min_pu (less electricity per unit heat)
        p_min_low_ratio = calculate_chp_minimum_operation(
            heat_demand_netcdf, buses, heat_to_power_ratio=1.0, min_operation_level=0.0
        )
        p_min_high_ratio = calculate_chp_minimum_operation(
            heat_demand_netcdf, buses, heat_to_power_ratio=2.0, min_operation_level=0.0
        )

        # Higher heat-to-power ratio should give lower p_min_pu
        assert (p_min_high_ratio["GB0"] < p_min_low_ratio["GB0"]).any()


class TestAttachCHPConstraints:
    """Tests for attach_chp_constraints function."""

    def test_attach_with_set_attribute(
        self, simple_network, powerplants_with_chp, heat_demand_netcdf
    ):
        """Test attaching CHP constraints when generators have 'set' attribute."""
        n = simple_network

        # Add generators with 'set' attribute
        n.add(
            "Generator",
            "GB0 CCGT-2030-0",
            bus="GB0",
            carrier="CCGT",
            p_nom=100.0,
            set="CHP",
        )
        n.add(
            "Generator",
            "GB0 CCGT-2030-1",
            bus="GB0",
            carrier="CCGT",
            p_nom=150.0,
            set="PP",
        )
        n.add(
            "Generator",
            "GB1 CCGT-2030-0",
            bus="GB1",
            carrier="CCGT",
            p_nom=80.0,
            set="CHP",
        )

        # Apply constraints
        attach_chp_constraints(
            n,
            powerplants_with_chp,
            heat_demand_netcdf,
            heat_to_power_ratio=1.5,
            min_operation_level=0.3,
        )

        # Check that p_min_pu is set for CHP generators
        assert "GB0 CCGT-2030-0" in n.generators_t.p_min_pu.columns
        assert "GB1 CCGT-2030-0" in n.generators_t.p_min_pu.columns

        # Check that non-CHP generator doesn't have constraint
        # (should be NaN or not in columns)
        if "GB0 CCGT-2030-1" in n.generators_t.p_min_pu.columns:
            assert n.generators_t.p_min_pu["GB0 CCGT-2030-1"].isna().all()

        # Check values are valid
        assert (n.generators_t.p_min_pu["GB0 CCGT-2030-0"] >= 0.0).all()
        assert (n.generators_t.p_min_pu["GB0 CCGT-2030-0"] <= 1.0).all()

    def test_attach_without_set_attribute(
        self, simple_network, powerplants_with_chp, heat_demand_netcdf
    ):
        """Test that missing 'set' attribute raises an error."""
        n = simple_network

        # Add generators WITHOUT 'set' attribute
        n.add("Generator", "gen_0", bus="GB0", carrier="CCGT", p_nom=100.0)
        n.add("Generator", "gen_1", bus="GB1", carrier="CCGT", p_nom=80.0)

        # Should raise ValueError when 'set' attribute is missing
        with pytest.raises(ValueError, match="'set' attribute missing"):
            attach_chp_constraints(
                n,
                powerplants_with_chp,
                heat_demand_netcdf,
                heat_to_power_ratio=1.5,
                min_operation_level=0.3,
            )

    def test_no_chp_powerplants(self, simple_network, heat_demand_netcdf):
        """Test when no CHP powerplants exist."""
        n = simple_network
        n.add("Generator", "gen_0", bus="GB0", carrier="CCGT", p_nom=100.0)

        ppl = pd.DataFrame(
            {"bus": ["GB0"], "carrier": ["CCGT"], "set": ["PP"], "p_nom": [100.0]}
        )

        # Should handle gracefully
        attach_chp_constraints(
            n, ppl, heat_demand_netcdf, heat_to_power_ratio=1.5, min_operation_level=0.3
        )

        # No constraints should be applied
        assert n.generators_t.p_min_pu.empty or n.generators_t.p_min_pu.isna().all().all()

    def test_no_matching_generators(
        self, simple_network, powerplants_with_chp, heat_demand_netcdf
    ):
        """Test when powerplants exist but no matching generators in network."""
        n = simple_network
        # Add generator with different carrier
        n.add("Generator", "gen_0", bus="GB0", carrier="nuclear", p_nom=500.0, set="PP")

        # Should handle gracefully
        attach_chp_constraints(
            n,
            powerplants_with_chp,
            heat_demand_netcdf,
            heat_to_power_ratio=1.5,
            min_operation_level=0.3,
        )

        # No constraints should be applied
        assert n.generators_t.p_min_pu.empty or n.generators_t.p_min_pu.isna().all().all()


class TestIntegration:
    """Integration tests for the full CHP workflow."""

    def test_full_workflow(self, simple_network, powerplants_with_chp, heat_demand_netcdf):
        """Test the complete workflow from powerplants to network constraints."""
        n = simple_network

        # Add all generators with proper attributes
        for idx, row in powerplants_with_chp.iterrows():
            n.add(
                "Generator",
                idx,
                bus=row["bus"],
                carrier=row["carrier"],
                p_nom=row["p_nom"],
                set=row["set"],
            )

        # Identify CHPs
        chp_plants = identify_chp_powerplants(powerplants_with_chp)
        assert len(chp_plants) == 2

        # Apply constraints
        attach_chp_constraints(
            n,
            powerplants_with_chp,
            heat_demand_netcdf,
            heat_to_power_ratio=1.5,
            min_operation_level=0.3,
        )

        # Verify only CHPs have constraints
        chp_indices = chp_plants.index
        for gen in chp_indices:
            assert gen in n.generators_t.p_min_pu.columns
            assert not n.generators_t.p_min_pu[gen].isna().all()

        # Verify constraints follow heat demand pattern
        # Peak heat should correspond to higher minimum operation
        for gen in chp_indices:
            p_min = n.generators_t.p_min_pu[gen]
            # Should have time-varying constraint
            assert p_min.std() > 0

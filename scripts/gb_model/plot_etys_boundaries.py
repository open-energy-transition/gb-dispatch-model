# SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
#
# SPDX-License-Identifier: MIT
"""
Plot ETYS boundary capabilities with interactive map.

Creates an interactive Plotly map showing:
- Regional boundaries with capacity annotations
- OSM transmission infrastructure (lines, substations, DC links)
- Interactive slider to highlight individual boundaries
- Toggleable OSM infrastructure layers
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import shapely


def get_voltage_color(voltage):
    """Return color for voltage level."""
    color_map = {
        400: "#8B0000",
        330: "#FF0000",
        275: "#FF6600",
        220: "#FF9900",
        132: "#0066FF",
        110: "#00CCFF",
    }
    for threshold, color in color_map.items():
        if voltage >= threshold:
            return color
    return "#CCCCCC"


def load_osm_infrastructure(osm_dir, gb_shapes, allowed_voltages):
    """Load and filter OSM infrastructure data for GB."""
    osm_lines = gpd.read_file(osm_dir / "lines.geojson")
    osm_buses = gpd.read_file(osm_dir / "buses.geojson")
    osm_links = gpd.read_file(osm_dir / "links.geojson")

    # Filter for GB
    osm_buses_gb = osm_buses[osm_buses["country"] == "GB"].copy()
    gb_bus_ids = set(osm_buses_gb["bus_id"])
    osm_lines_gb = osm_lines[
        osm_lines["bus0"].isin(gb_bus_ids) | osm_lines["bus1"].isin(gb_bus_ids)
    ].copy()
    osm_links_gb = gpd.sjoin(gb_shapes, osm_links, how="right").dropna(subset=["name"])

    # Filter by voltage
    osm_lines_gb = osm_lines_gb[osm_lines_gb["voltage"].isin(allowed_voltages)]
    osm_buses_gb = osm_buses_gb[osm_buses_gb["voltage"].isin(allowed_voltages)]

    # Convert to lat/lon
    return {
        "lines": osm_lines_gb.to_crs("EPSG:4326"),
        "buses": osm_buses_gb.to_crs("EPSG:4326"),
        "links": osm_links_gb.to_crs("EPSG:4326"),
    }


def extract_geometry_coords(geometry):
    """Extract lon/lat coordinates from LineString or MultiLineString."""
    all_lons, all_lats = [], []
    linestrings = (
        [geometry]
        if isinstance(geometry, shapely.geometry.LineString)
        else geometry.geoms
    )

    for linestring in linestrings:
        x, y = linestring.xy
        all_lons.extend(list(x) + [None])
        all_lats.extend(list(y) + [None])

    return all_lons, all_lats


def compute_annotation_position(lons, lats):
    """Compute annotation position: end of line or center if circular."""
    valid_lons = [x for x in lons if x is not None]
    valid_lats = [y for y in lats if y is not None]

    is_circular = (
        len(valid_lons) > 1
        and abs(valid_lons[0] - valid_lons[-1]) < 1e-6
        and abs(valid_lats[0] - valid_lats[-1]) < 1e-6
    )

    if is_circular:
        return sum(valid_lons) / len(valid_lons), sum(valid_lats) / len(valid_lats)
    return valid_lons[-1], valid_lats[-1]


def prepare_boundary_data(lines_plot):
    """Prepare boundary line data with geometries and annotations."""
    line_data = []
    for boundary_name in lines_plot["Boundary_n"].unique():
        boundary_rows = lines_plot[lines_plot["Boundary_n"] == boundary_name]
        capacity = boundary_rows["capability_mw"].iloc[0]

        # Collect all line segments
        all_lons, all_lats = [], []
        for _, row in boundary_rows.iterrows():
            if row.geometry:
                lons, lats = extract_geometry_coords(row.geometry)
                all_lons.extend(lons)
                all_lats.extend(lats)

        ann_lon, ann_lat = compute_annotation_position(all_lons, all_lats)

        line_data.append(
            {
                "name": boundary_name,
                "capacity": capacity,
                "lon": all_lons,
                "lat": all_lats,
                "centroid_lon": ann_lon,
                "centroid_lat": ann_lat,
            }
        )

    return line_data


def load_boundary_data(shapes_file, etys_caps_file, boundaries_file):
    """Load data for boundary visualization."""
    shapes = (
        gpd.read_file(shapes_file)
        .query("country == 'GB'")
        .query("TO_region != 'N-IRL'")
    )
    etys_caps = pd.read_csv(etys_caps_file)
    lines = gpd.read_file(boundaries_file).to_crs(shapes.crs)
    lines = lines[lines["Boundary_n"].isin(etys_caps["boundary_name"])]

    shapes_plot = (
        shapes.to_crs("EPSG:4326")
        .reset_index(drop=False)
        .rename(columns={"index": "fid"})
    )
    lines_plot = lines.to_crs("EPSG:4326").merge(
        etys_caps[["boundary_name", "capability_mw"]],
        left_on="Boundary_n",
        right_on="boundary_name",
        how="left",
    )

    minx, miny, maxx, maxy = shapes_plot.total_bounds
    n_regions = len(shapes_plot)
    colors = (
        pc.qualitative.Set3[:n_regions]
        if n_regions <= len(pc.qualitative.Set3)
        else pc.sample_colorscale("Rainbow", n_regions)
    )

    line_data = prepare_boundary_data(lines_plot)

    return {
        "shapes_plot": shapes_plot,
        "color_map": {fid: colors[i] for i, fid in enumerate(shapes_plot["fid"])},
        "n_regions": n_regions,
        "line_data": line_data,
        "center": {"lat": (miny + maxy) / 2, "lon": (minx + maxx) / 2},
    }


def add_osm_lines_trace(fig, osm_lines):
    """Add OSM transmission line traces grouped by voltage."""
    for voltage in sorted(osm_lines["voltage"].dropna().unique(), reverse=True):
        group = osm_lines[osm_lines["voltage"] == voltage]
        all_lons, all_lats = [], []

        for _, row in group.iterrows():
            lons, lats = extract_geometry_coords(row.geometry)
            all_lons.extend(lons)
            all_lats.extend(lats)

        fig.add_trace(
            go.Scattermap(
                lon=all_lons,
                lat=all_lats,
                mode="lines",
                line=dict(color=get_voltage_color(voltage), width=2.5),
                opacity=0.6,
                name=f"Lines {int(voltage)}kV",
                legendgroup="osm_lines",
                legendgrouptitle_text="OSM Lines"
                if voltage
                == sorted(osm_lines["voltage"].dropna().unique(), reverse=True)[0]
                else None,
                showlegend=True,
                visible="legendonly",
                hovertext=f"{int(voltage)} kV transmission line",
                hoverinfo="text",
            )
        )


def add_osm_buses_trace(fig, osm_buses):
    """Add OSM substation traces grouped by voltage."""
    for voltage in sorted(osm_buses["voltage"].dropna().unique(), reverse=True):
        group = osm_buses[osm_buses["voltage"] == voltage]
        lons = [p.x for p in group.geometry]
        lats = [p.y for p in group.geometry]

        fig.add_trace(
            go.Scattermap(
                lon=lons,
                lat=lats,
                mode="markers",
                marker=dict(
                    size=8 if voltage >= 275 else 6, color=get_voltage_color(voltage)
                ),
                opacity=0.7,
                name=f"Substations {int(voltage)}kV",
                legendgroup="osm_buses",
                legendgrouptitle_text="OSM Substations"
                if voltage
                == sorted(osm_buses["voltage"].dropna().unique(), reverse=True)[0]
                else None,
                showlegend=True,
                visible="legendonly",
                hovertext=f"{int(voltage)} kV substation",
                hoverinfo="text",
            )
        )


def add_osm_links_trace(fig, osm_links):
    """Add OSM DC links trace (all voltages combined)."""
    if osm_links.empty:
        return

    all_lons, all_lats = [], []
    for _, row in osm_links.iterrows():
        lons, lats = extract_geometry_coords(row.geometry)
        all_lons.extend(lons)
        all_lats.extend(lats)

    fig.add_trace(
        go.Scattermap(
            lon=all_lons,
            lat=all_lats,
            mode="lines",
            line=dict(color="#9400D3", width=3),
            opacity=0.8,
            name="DC Links",
            legendgroup="osm_links",
            showlegend=True,
            visible="legendonly",
            hovertext="DC link",
            hoverinfo="text",
        )
    )


def add_boundary_traces(fig, line_data):
    """Add boundary line traces and annotation placeholder."""
    for line_info in line_data:
        fig.add_trace(
            go.Scattermap(
                lon=line_info["lon"],
                lat=line_info["lat"],
                mode="lines",
                line=dict(color="gray", width=2),
                opacity=0.5,
                name=line_info["name"],
                showlegend=False,
                hovertext=f"{line_info['name']}: {line_info['capacity'] / 1000:.1f} GW",
                hoverinfo="text",
                visible=True,
            )
        )

    # Add annotation placeholder
    fig.add_trace(
        go.Scattermap(
            lon=[],
            lat=[],
            mode="text",
            text=[],
            textfont=dict(size=12, color="black", weight="bold"),
            opacity=1.0,
            showlegend=False,
            hoverinfo="skip",
            visible=True,
        )
    )


def add_choropleth_trace(fig, data):
    """Add choropleth trace for regions."""
    shapes_plot = data["shapes_plot"]
    color_map = data["color_map"]

    fig.add_trace(
        go.Choroplethmap(
            geojson=shapes_plot.__geo_interface__,
            locations=shapes_plot["fid"],
            featureidkey="properties.fid",
            z=shapes_plot["fid"],
            colorscale=[
                [i / (data["n_regions"] - 1), color_map[fid]]
                for i, fid in enumerate(shapes_plot["fid"])
            ],
            showscale=False,
            marker_opacity=0.5,
            marker_line_width=1,
            customdata=shapes_plot[["name", "TO_region"]],
            hovertemplate="%{customdata[0]}<br>TO_region: %{customdata[1]}<extra></extra>",
            visible=True,
            name="Regions",
            showlegend=False,
        )
    )


def create_slider_steps(line_data, num_osm_traces, num_boundary_lines):
    """Create slider steps for boundary highlighting."""
    all_unique_lines = ["None"] + [ld["name"] for ld in line_data]
    boundary_trace_indices = list(
        range(num_osm_traces, num_osm_traces + num_boundary_lines)
    )
    annotation_trace_idx = num_osm_traces + num_boundary_lines
    slider_steps = []

    for line_name in all_unique_lines:
        all_lons, all_lats, line_colors, line_widths, line_opacities = (
            [],
            [],
            [],
            [],
            [],
        )

        for line_info in line_data:
            is_highlighted = line_name != "None" and line_info["name"] == line_name
            all_lons.append(line_info["lon"])
            all_lats.append(line_info["lat"])
            line_colors.append("red" if is_highlighted else "gray")
            line_widths.append(4 if is_highlighted else 2)
            line_opacities.append(1.0 if is_highlighted else 0.5)

        # Build annotation data
        matching = [l for l in line_data if l["name"] == line_name]
        if line_name != "None" and matching:
            h = matching[0]
            ann_lon, ann_lat = [h["centroid_lon"]], [h["centroid_lat"]]
            ann_text = [f"{line_name}<br>{h['capacity'] / 1000:.1f} GW"]
        else:
            ann_lon, ann_lat, ann_text = [], [], []

        slider_steps.append(
            {
                "args": [
                    {
                        "lon": all_lons + [ann_lon],
                        "lat": all_lats + [ann_lat],
                        "line.color": line_colors + [None],
                        "line.width": line_widths + [None],
                        "opacity": line_opacities + [None],
                        "text": [None] * num_boundary_lines + [ann_text],
                    },
                    boundary_trace_indices + [annotation_trace_idx],
                ],
                "label": line_name,
                "method": "restyle",
            }
        )

    return slider_steps


if __name__ == "__main__":
    if "snakemake" not in globals():
        from scripts._helpers import mock_snakemake

        snakemake = mock_snakemake(
            "plot_etys_boundaries",
            configfiles="config/config.gb.default.yaml",
        )

    # Extract parameters
    allowed_voltages = set(snakemake.params.voltages)

    # Load data
    osm_dir = Path(snakemake.input.osm_dir)
    gb_shapes = gpd.read_file(snakemake.input.shapes)
    osm_data = load_osm_infrastructure(osm_dir, gb_shapes, allowed_voltages)

    boundary_data = load_boundary_data(
        snakemake.input.shapes,
        snakemake.input.etys_caps,
        snakemake.input.boundaries,
    )

    # Count traces
    num_osm_traces = (
        len(osm_data["lines"]["voltage"].dropna().unique())
        + len(osm_data["buses"]["voltage"].dropna().unique())
        + (1 if not osm_data["links"].empty else 0)
    )

    # Create figure and add traces
    fig = go.Figure()
    add_osm_lines_trace(fig, osm_data["lines"])
    add_osm_buses_trace(fig, osm_data["buses"])
    add_osm_links_trace(fig, osm_data["links"])

    line_data = boundary_data["line_data"]
    add_boundary_traces(fig, line_data)
    add_choropleth_trace(fig, boundary_data)

    num_boundary_lines = len(line_data)

    # Create slider
    slider_steps = create_slider_steps(line_data, num_osm_traces, num_boundary_lines)

    # Configure layout
    fig.update_layout(
        width=800,
        height=1000,
        margin={"r": 0, "t": 50, "l": 0, "b": 100},
        transition={"duration": 0},
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "y": 0,
                "xanchor": "left",
                "x": 0.1,
                "currentvalue": {
                    "prefix": "Boundary: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "pad": {"b": 10, "t": 50},
                "len": 0.8,
                "steps": slider_steps,
            }
        ],
        map=dict(
            style="carto-positron",
            center=boundary_data["center"],
            zoom=5,
        ),
        dragmode="zoom",
    )

    # Save output
    fig.write_html(snakemake.output.html)

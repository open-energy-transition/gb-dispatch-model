# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT
"""
Plot regions, ETYS boundary capabilities, and underlying network with interactive map.

Creates an interactive Plotly map showing:
- Regional boundaries with capacity annotations
- OSM transmission infrastructure (lines, substations, DC links)
- Current and planned interconnectors with build years
- Interactive slider to highlight individual boundaries
- Toggleable OSM infrastructure layers
"""

import geopandas as gpd
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import pypsa
import shapely

from scripts.gb_model.transmission.create_interconnectors_table import (
    projects_to_pypsa_links,
)


def get_voltage_color(voltage: float) -> str:
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


def load_network_data(
    network: pypsa.Network, gb_shapes: gpd.GeoDataFrame, allowed_voltages: set
) -> dict[str, gpd.GeoDataFrame]:
    """Load and filter OSM infrastructure data for GB, which has already been loaded into a PyPSA network."""

    # Filter for GB
    buses_gb = _prep_static_table(network.buses[network.buses["country"] == "GB"])
    buses_gb = gpd.sjoin(buses_gb, gb_shapes[["geometry"]], how="left").dropna(
        subset=["index_right"]
    )
    lines_gb = _prep_static_table(
        network.lines.query("bus0 in @buses_gb.index and bus1 in @buses_gb.index")
    )
    links_gb = _prep_static_table(
        network.links.query("bus0 in @buses_gb.index and bus1 in @buses_gb.index")
    )

    # Filter by voltage
    lines_gb = lines_gb.query("v_nom in @v", local_dict={"v": allowed_voltages})
    buses_gb = buses_gb.query("v_nom in @v", local_dict={"v": allowed_voltages})

    # Convert to lat/lon
    return {"lines": lines_gb, "buses": buses_gb, "links": links_gb}


def _prep_static_table(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """Prepare static table for plotting."""
    df = df.copy()
    df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry"])
    return gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")


def load_interconnector_data(
    regions: gpd.GeoDataFrame,
    interconnector_options: list[dict],
    interconnector_plan: dict[str, dict[int, list[str]]],
    year_range: tuple[int, int],
) -> gpd.GeoDataFrame:
    """Load and filter interconnector data for GB."""
    # Map interconnector projects to PyPSA links
    all_interconnector_plan = {
        year: plan
        for year, plan in interconnector_plan["HT"].items()
        if year < year_range[0]
    }
    all_interconnector_plan[int(year_range[0])] = sorted(
        set()
        .union([i["name"] for i in interconnector_options])
        .difference(*all_interconnector_plan.values())
    )
    interconnector_links = projects_to_pypsa_links(
        interconnector_options,
        all_interconnector_plan,
        regions,
        [year_range[0], year_range[0]],
        regions.crs.to_string(),
    )
    for scenario, plan in interconnector_plan.items():
        interconnector_links[f"{scenario}_year"] = interconnector_links.project.map(
            lambda x: next(
                (year for year, projects in plan.items() if x in projects), None
            )
        )

    return interconnector_links


def extract_geometry_coords(
    geometry: shapely.geometry.base.BaseGeometry,
) -> tuple[list[float], list[float]]:
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


def compute_annotation_position(
    lons: list[float], lats: list[float]
) -> tuple[float, float]:
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


def prepare_boundary_data(lines_plot: gpd.GeoDataFrame) -> list[dict]:
    """
    Prepare boundary line data with geometries and annotations.

    Parameters
    ----------
    lines_plot: gpd.GeoDataFrame
        GeoDataFrame containing boundary lines with 'Boundary_n' and 'capability_mw' columns.

    Returns
    -------
    list of dict
        Each dict contains:
        - 'name': Boundary name
        - 'capacity': Boundary capacity in MW
        - 'lon': List of longitudes for the line geometry
        - 'lat': List of latitudes for the line geometry
        - 'centroid_lon': Longitude for annotation placement
        - 'centroid_lat': Latitude for annotation placement
    """
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


def load_boundary_data(
    gb_shapes: gpd.GeoDataFrame, etys_caps: pd.DataFrame, boundaries: gpd.GeoDataFrame
) -> dict:
    """
    Load data for boundary visualization.

    Parameters
    ----------
    gb_shapes: gpd.GeoDataFrame

        GeoDataFrame containing GB shapes.
    etys_caps: pd.DataFrame
        DataFrame containing ETYS boundary capabilities.
    boundaries: gpd.GeoDataFrame
        GeoDataFrame containing boundary geometries.

    Returns
    -------
    dict
        Dictionary containing shapes for plotting, color mapping, number of regions, line data, and map center coordinates.
    """
    lines = boundaries[boundaries["Boundary_n"].isin(etys_caps["boundary_name"])]

    shapes_plot = (
        gb_shapes.to_crs("EPSG:4326")
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


def add_lines_trace(fig: go.Figure, lines: pd.DataFrame):
    """
    Add OSM transmission line traces grouped by voltage.

    Parameters
    ----------
    fig: plotly.graph_objects.Figure
        The Plotly figure to which the traces will be added.
    lines: pd.DataFrame
        DataFrame containing line geometries and voltage levels.
    """
    for voltage in sorted(lines["v_nom"].dropna().unique(), reverse=True):
        group = lines[lines["v_nom"] == voltage]
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
                name=f"AC {int(voltage)}kV",
                legendgroup="lines",
                legendgrouptitle_text="Transmission Lines"
                if voltage == sorted(lines["v_nom"].dropna().unique(), reverse=True)[0]
                else None,
                showlegend=True,
                visible="legendonly",
                hovertext=f"{int(voltage)} kV transmission line",
                hoverinfo="text",
            )
        )


def add_buses_trace(fig: go.Figure, buses: gpd.GeoDataFrame):
    """
    Add OSM substation traces grouped by voltage.

    Parameters
    ----------
    fig: plotly.graph_objects.Figure
        The Plotly figure to which the traces will be added.
    buses: geopandas.GeoDataFrame
        GeoDataFrame containing bus geometries and voltage levels.
    """
    for voltage in sorted(buses["v_nom"].dropna().unique(), reverse=True):
        group = buses[buses["v_nom"] == voltage]
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
                name=f"{int(voltage)}kV",
                legendgroup="buses",
                legendgrouptitle_text="Buses (substations, offshore wind farms, etc.)"
                if voltage == sorted(buses["v_nom"].dropna().unique(), reverse=True)[0]
                else None,
                showlegend=True,
                visible="legendonly",
                hovertext=f"{int(voltage)} kV substation",
                hoverinfo="text",
            )
        )


def add_links_trace(
    fig: go.Figure,
    internal_links: pd.DataFrame,
    interconnector_data: pd.DataFrame,
    start_year: int,
    interconnector_plan: dict,
):
    """
    Add DC links trace (all voltages combined).

    Parameters
    ----------
    fig: plotly.graph_objects.Figure
        The Plotly figure to which the traces will be added.
    internal_links: pd.DataFrame
        DataFrame containing internal GB DC links.
    interconnector_data: pd.DataFrame
        DataFrame containing interconnector data.
    start_year: int
        The starting year for the dispatch runs (one year after the FES year).
    interconnector_plan: dict
        Dictionary containing the interconnector plan for different scenarios.
    """
    if internal_links.empty and interconnector_data.empty:
        return

    all_lons, all_lats = [], []
    for _, row in internal_links.iterrows():
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
            name="Internal GB HVDC",
            legendgroup="lines",
            showlegend=True,
            visible="legendonly",
            hovertext="Internal GB HVDC line",
            hoverinfo="text",
        )
    )
    legend = True
    for _, row in interconnector_data.iterrows():
        lons, lats = extract_geometry_coords(row.geometry)
        hovertext = f"{row['project']} ({row['p_nom'] / 1000:.0f} GW)"
        if row["build_year"] < start_year:
            hovertext += f"<br>Already constructed prior to {start_year}"
        else:
            for scenario in interconnector_plan.keys():
                if pd.notna(row[f"{scenario}_year"]):
                    hovertext += (
                        f"<br>{scenario} pathway build year: {row[f'{scenario}_year']}"
                    )
                else:
                    hovertext += f"<br>{scenario} pathway: Not built"

        fig.add_trace(
            go.Scattermap(
                lon=lons,
                lat=lats,
                mode="lines",
                line=dict(color="#D3007F", width=3),
                opacity=0.8,
                name="HVDC interconnectors",
                legendgroup="lines",
                showlegend=legend,
                visible="legendonly",
                hovertext=hovertext,
                hoverinfo="text",
            )
        )
        legend = False  # Only show legend for the first interconnector trace


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
            customdata=shapes_plot[["name", "TO_region", "raw_region_ids"]],
            hovertemplate="%{customdata[0]}<br>TO_region: %{customdata[1]}<br>Raw Region IDs: %{customdata[2]}<extra></extra>",
            visible=True,
            name="Regions",
            showlegend=False,
        )
    )


def create_slider_steps(line_data, num_traces, num_boundary_lines):
    """Create slider steps for boundary highlighting."""
    all_unique_lines = ["None"] + [ld["name"] for ld in line_data]
    boundary_trace_indices = list(range(num_traces, num_traces + num_boundary_lines))
    annotation_trace_idx = num_traces + num_boundary_lines
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
    regions = gpd.read_file(snakemake.input.shapes)

    gb_shapes = regions.query("country == 'GB' and TO_region != 'N-IRL'")
    network = pypsa.Network(snakemake.input.network)
    network_data = load_network_data(network, gb_shapes, allowed_voltages)
    interconnector_data = load_interconnector_data(
        regions,
        snakemake.params.interconnector_options,
        snakemake.params.interconnector_plan,
        snakemake.params.year_range,
    )
    etys_caps = pd.read_csv(snakemake.input.etys_caps)
    boundaries = gpd.read_file(snakemake.input.boundaries).to_crs(gb_shapes.crs)
    boundary_data = load_boundary_data(gb_shapes, etys_caps, boundaries)

    # Count traces
    num_traces = (
        len(network_data["lines"]["v_nom"].dropna().unique())
        + len(network_data["buses"]["v_nom"].dropna().unique())
        + (1 if not network_data["links"].empty else 0)
        + len(interconnector_data)
    )

    # Create figure and add traces
    fig = go.Figure()
    add_choropleth_trace(fig, boundary_data)
    add_lines_trace(fig, network_data["lines"])
    add_buses_trace(fig, network_data["buses"])
    add_links_trace(
        fig,
        network_data["links"],
        interconnector_data,
        snakemake.params.year_range[0],
        snakemake.params.interconnector_plan,
    )

    line_data = boundary_data["line_data"]
    add_boundary_traces(fig, line_data)

    num_boundary_lines = len(line_data)

    # Create slider
    slider_steps = create_slider_steps(line_data, num_traces, num_boundary_lines)

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

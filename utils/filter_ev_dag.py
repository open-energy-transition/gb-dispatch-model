# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Filter a Snakemake rulegraph DOT to EV-related rules only.

Usage::

    snakemake --rulegraph -F <targets> | python utils/filter_ev_dag.py | dot -Tsvg -o doc/gb-model/img/ev_workflow.svg
"""

import re
import sys

# Rule name substrings to retain in the filtered graph
KEEP = {
    "process_ev",  # process_ev_demand_shape
    "create_ev",  # create_ev_peak_charging_table, create_ev_v2g_storage_table
    "scaled_ev",  # scaled_ev_demand_profile
    "flexibility_table",  # create_flexibility_table (DSR and V2G capacities)
    "synthesise_gb_regional",  # regional disaggregation
    "process_fes_gsp",  # FES BB1 regional source data
    "extract_fes",  # FES workbook sheet extraction
    "synthesise_eur",  # synthesise_eur_data (adds European neighbours to regional data)
}

dot_input = sys.stdin.read()

# Parse node definitions:  <id>[label = "<name>", ...]
node_re = re.compile(r'(\d+)\[label\s*=\s*"([^"]+)"')
nodes: dict[str, str] = {}
for m in node_re.finditer(dot_input):
    nid, label = m.group(1), m.group(2)
    if any(k in label.lower() for k in KEEP):
        nodes[nid] = label

# Parse edges: <src> -> <dst>;
edge_re = re.compile(r"(\d+)\s*->\s*(\d+)")
edges: list[tuple[str, str]] = [
    (m.group(1), m.group(2))
    for m in edge_re.finditer(dot_input)
    if m.group(1) in nodes and m.group(2) in nodes
]

# Emit filtered DOT
print("digraph snakemake_dag {")
print("    graph[bgcolor=white, margin=0];")
print("    node[shape=box, style=rounded, fontname=sans, fontsize=10, penwidth=2];")
print("    edge[penwidth=2, color=grey];")
for nid in sorted(nodes, key=int):
    print(
        f'    {nid}[label = "{nodes[nid]}", '
        'style="rounded", shape=box, peripheries=1, color=black];'
    )
for src, dst in edges:
    print(f"    {src} -> {dst};")
print("}")

# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

r"""
Filter a Snakemake rulegraph DOT to gb-model related rules only.

Keeps nodes whose labels exactly match a rule defined in any
``rules/gb-model/*.smk`` file, plus their direct parent nodes.

File nodes showing the output path for each leaf rule are appended, with
concrete wildcard values filled in from the supplied target paths.
Wildcard names passed via ``-w`` / ``--wildcard`` are kept as ``{name}``
instead of being filled in.

Usage::

    python utils/filter_gb_model_rulegraph.py \\
        [-w wildcard_name ...] [-f rule_file ...] [-c cutoff_rule ...] [targets...] | \\
        dot -Tsvg -o doc/gb-model/img/gb_model_workflow.svg

Example — keep ``fes_scenario`` and ``year`` as wildcards, highlight ev rules::

    python utils/filter_gb_model_rulegraph.py \\
        -w fes_scenario -w year \\
        -f rules/gb-model/ev.smk \\
        resources/GB/gb-model/HT/ev_demand/2035.csv

Example — prune all rules upstream of ``compose_network`` (useful for redispatch docs
where the upstream pipeline is already documented elsewhere)::

    python utils/filter_gb_model_rulegraph.py \\
        -c compose_network \\
        -w fes_scenario \\
        -f rules/gb-model/redispatch.smk \\
        results/GB/networks/HT/constrained_clustered/2040.nc
"""

import re
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import click

SMK_DIR = Path(__file__).parent.parent / "rules" / "gb-model"

# ---------------------------------------------------------------------------
# Compiled regexes
# ---------------------------------------------------------------------------
_RULE_RE = re.compile(r"^rule\s+(\w+)\s*:", re.MULTILINE)
_USE_RULE_RE = re.compile(r"^use rule\s+(\w+)\s+as\s+(\w+)", re.MULTILINE)
_TOP_LEVEL_RE = re.compile(r"^(?:rule|use rule)\s+", re.MULTILINE)
_RESOURCES_RE = re.compile(r'resources\s*\(\s*f?["\']([^"\']+)["\']')
_RESULTS_RE = re.compile(r'RESULTS\s*\+\s*f?["\']([^"\']+)["\']')
_OUTPUT_SEC_RE = re.compile(r"\n    output\s*:(.*?)(?=\n    \w|\Z)", re.DOTALL)
_NODE_RE = re.compile(r'(\d+)\[label\s*=\s*"([^"]+)"')
_EDGE_RE = re.compile(r"(\d+)\s*->\s*(\d+)")

# ---------------------------------------------------------------------------
# Colour palette for node categories
# ---------------------------------------------------------------------------
_COLOUR_HIGHLIGHT = "forestgreen"
_COLOUR_HIGHLIGHT_FILL = "#e6f4eac0"
_COLOUR_GB_MODEL = "black"
_COLOUR_GB_MODEL_FILL = "#fbfbfbc0"
_COLOUR_UPSTREAM = "steelblue"
_COLOUR_UPSTREAM_FILL = "#ddeeffc0"
_COLOUR_FILE_FILL = "lightyellow"
_COLOUR_FILE_BORDER = "darkorange"


# ---------------------------------------------------------------------------
# Rule name extraction
# ---------------------------------------------------------------------------


def _rule_names_in_files(paths) -> set[str]:
    """Return all rule/alias names declared in *paths*."""
    names: set[str] = set()
    for text in (p.read_text() for p in paths):
        names |= set(_RULE_RE.findall(text)) | {
            m.group(2) for m in _USE_RULE_RE.finditer(text)
        }
    return names


def get_gb_model_rules(smk_dir: Path) -> set[str]:
    """Return all rule names defined in ``*.smk`` files under *smk_dir*."""
    return _rule_names_in_files(smk_dir.glob("*.smk"))


# ---------------------------------------------------------------------------
# SMK output pattern parsing
# ---------------------------------------------------------------------------


def _rule_body(text: str, start: int) -> str:
    """Slice from *start* to the next top-level rule declaration."""
    m = _TOP_LEVEL_RE.search(text, start)
    return text[start : m.start()] if m else text[start:]


def _output_patterns(body: str) -> list[tuple[str, str]]:
    """
    Return ``(match_pattern, display_pattern)`` pairs from an output: section.

    *match_pattern* is the inner path used for wildcard matching; *display_pattern*
    prepends the top-level directory (``resources/`` or ``results/``).
    """
    m = _OUTPUT_SEC_RE.search("\n" + body)
    if not m:
        return []
    section = m.group(1)
    return [
        (inner, f"{prefix}/{inner}")
        for regex, prefix in ((_RESOURCES_RE, "resources"), (_RESULTS_RE, "results"))
        for inner in regex.findall(section)
    ]


def parse_rule_outputs(smk_dir: Path) -> dict[str, list[tuple[str, str]]]:
    """
    Return ``{rule_name: [(match_pattern, display_pattern), ...]}`` for every
    rule in *smk_dir*.
    """
    bodies: dict[str, str] = {}
    parents: dict[str, str] = {}  # alias -> original rule

    for smk_file in smk_dir.glob("*.smk"):
        text = smk_file.read_text()
        for m in _RULE_RE.finditer(text):
            bodies[m.group(1)] = _rule_body(text, m.end())
        for m in _USE_RULE_RE.finditer(text):
            original, alias = m.group(1), m.group(2)
            parents[alias] = original
            bodies[alias] = _rule_body(text, m.end())

    rule_outputs = {
        name: pats for name, body in bodies.items() if (pats := _output_patterns(body))
    }
    # aliases inherit the parent output when they don't override it
    for alias, parent in parents.items():
        if alias not in rule_outputs and parent in rule_outputs:
            rule_outputs[alias] = rule_outputs[parent]

    return rule_outputs


# ---------------------------------------------------------------------------
# Wildcard pattern helpers
# ---------------------------------------------------------------------------


def _top_level(display_pattern: str) -> str:
    """Return the top-level directory prefix from a display pattern, e.g. ``results/``."""
    return display_pattern.split("/")[0] + "/"


def _wildcard_regex(pattern: str) -> tuple[re.Pattern[str], list[str]]:
    """
    Convert a Snakemake pattern to a named-group capturing regex.

    Returns ``(regex, [wildcard_name, ...])``.
    """
    names = re.findall(r"\{(\w+)\}", pattern)
    capturing = re.sub(
        r"\\{(\w+)\\}",
        lambda m: f"(?P<{m.group(1)}>[^/]+)",
        re.escape(pattern),
    )
    return re.compile(capturing + "$"), names


def patterns_matching_targets(
    patterns: list[tuple[str, str]], targets: list[str]
) -> list[tuple[str, str]]:
    """Return ``(match_pattern, display_pattern)`` pairs matched by at least one target path."""

    def _matches_any(mp: str, dp: str) -> bool:
        top, cap_re = _top_level(dp), _wildcard_regex(mp)[0]
        return any(t.startswith(top) and cap_re.search(t) for t in targets)

    return [(mp, dp) for mp, dp in patterns if _matches_any(mp, dp)]


def _match_score(
    match_pattern: str, target: str, display_pattern: str = ""
) -> int | None:
    """
    Total wildcard match length for *match_pattern* against *target* (smaller = more specific).

    Returns ``None`` on no match.  When *display_pattern* is supplied the target must
    share its top-level directory (``results/`` vs ``resources/``) to avoid false positives.
    """
    if display_pattern and not target.startswith(_top_level(display_pattern)):
        return None
    cap_re, names = _wildcard_regex(match_pattern)
    m = cap_re.search(target)
    return sum(len(m.group(n)) for n in names) if m else None


def make_partial_labels(
    display_pattern: str,
    match_pattern: str,
    targets: list[str],
    keep_wildcards: set[str],
) -> list[str]:
    """
    Fill wildcard values from matching targets into *display_pattern*.

    Uses *match_pattern* for matching and *display_pattern* as the label template.
    Wildcards in *keep_wildcards* are left as ``{name}``.  Returns a deduplicated
    list, or ``[display_pattern]`` when nothing matches.
    """
    cap_re, names = _wildcard_regex(match_pattern)
    top = _top_level(display_pattern)

    def _fill(target: str) -> str | None:
        if not target.startswith(top):
            return None
        m = cap_re.search(target)
        if m is None:
            return None
        label = display_pattern
        for name in names:
            if name not in keep_wildcards:
                label = label.replace(f"{{{name}}}", m.group(name))
        return label

    labels = list(dict.fromkeys(filter(None, (_fill(t) for t in targets))))
    return labels or [display_pattern]


# ---------------------------------------------------------------------------
# Graph operations
# ---------------------------------------------------------------------------


def get_rulegraph_dot(targets: list[str]) -> str:
    """Run ``snakemake --rulegraph -F <targets>`` and return the DOT output."""
    return subprocess.run(
        ["snakemake", "--rulegraph", "-F", *targets],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    ).stdout


def _build_rev_adj(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Return a reverse adjacency map ``{dst: {src, ...}}`` for *edges*."""
    rev_adj: dict[str, set[str]] = defaultdict(set)
    for src, dst in edges:
        rev_adj[dst].add(src)
    return rev_adj


def _ancestors(start: set[str], rev_adj: dict[str, set[str]]) -> set[str]:
    """Return all nodes reachable backward from *start* via *rev_adj*."""
    reachable = set(start)
    queue = deque(start)
    while queue:
        node = queue.popleft()
        for parent in rev_adj[node]:
            if parent not in reachable:
                reachable.add(parent)
                queue.append(parent)
    return reachable


def filter_graph(
    dot_input: str, gb_model_rules: set[str]
) -> tuple[set[str], set[str], dict[str, str], list[tuple[str, str]]]:
    """
    Filter the rulegraph DOT to gb-model rules plus their ancestors.

    Returns ``(keep_nodes, gb_model_nodes, all_nodes, edges)``.
    """
    all_nodes = {m.group(1): m.group(2) for m in _NODE_RE.finditer(dot_input)}
    all_edges = [(m.group(1), m.group(2)) for m in _EDGE_RE.finditer(dot_input)]

    gb_model_nodes = {
        nid for nid, label in all_nodes.items() if label in gb_model_rules
    }
    keep_nodes = gb_model_nodes | {
        src for src, dst in all_edges if dst in gb_model_nodes
    }
    edges = [(s, d) for s, d in all_edges if s in keep_nodes and d in keep_nodes]

    # Drop dead-ends: keep only nodes that can reach a dag target
    # (a node with no outgoing edges in the full graph).
    all_srcs = {src for src, _ in all_edges}
    dag_targets = {nid for nid in keep_nodes if nid not in all_srcs}
    keep_nodes &= _ancestors(dag_targets, _build_rev_adj(edges))
    edges = [(s, d) for s, d in edges if s in keep_nodes and d in keep_nodes]

    return keep_nodes, gb_model_nodes, all_nodes, edges


def build_file_nodes(
    keep_nodes: set[str],
    gb_model_nodes: set[str],
    all_nodes: dict[str, str],
    edges: list[tuple[str, str]],
    targets: list[str],
    keep_wildcards: set[str],
    rule_output_patterns: dict[str, list[tuple[str, str]]],
) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """
    Build output-file nodes for gb-model rules, using specificity to disambiguate.

    For each target, only the rule(s) whose output pattern matches with the
    *smallest* total wildcard match length (most specific) receive a file node for
    that target.  This prevents broad patterns like ``regional_{data}.csv`` from
    claiming targets that are better matched by more specific rules such as
    ``regional_H2_demand_annual_inc_eur.csv``.

    Returns ``(file_node_labels, file_edges)`` where *file_node_labels* maps
    node ids to display labels and *file_edges* is ``(rule_nid, file_nid)`` pairs.
    """
    file_node_labels: dict[str, str] = {}
    file_edges: list[tuple[str, str]] = []
    next_id = max(int(nid) for nid in keep_nodes) + 1 if keep_nodes else 0

    if not targets:
        for nid in sorted(keep_nodes & gb_model_nodes, key=int):
            pats = rule_output_patterns.get(all_nodes[nid], [])
            all_labels = list(
                dict.fromkeys(
                    lbl
                    for match_pat, display_pat in pats
                    for lbl in make_partial_labels(
                        display_pat, match_pat, [], keep_wildcards
                    )
                )
            )
            if all_labels:
                file_node_labels[str(next_id)] = "\\n".join(all_labels)
                file_edges.append((nid, str(next_id)))
                next_id += 1
        return file_node_labels, file_edges

    # For each target, assign it only to the rule(s) with the minimum wildcard score.
    # nid -> ordered label set
    node_labels: dict[str, dict[str, None]] = defaultdict(dict)

    for target in targets:
        best_score: int | None = None
        candidates: list[tuple[str, str]] = []  # (nid, label)

        for nid in sorted(keep_nodes & gb_model_nodes, key=int):
            for match_pat, display_pat in rule_output_patterns.get(all_nodes[nid], []):
                score = _match_score(match_pat, target, display_pat)
                if score is None:
                    continue
                labels = make_partial_labels(
                    display_pat, match_pat, [target], keep_wildcards
                )
                label = labels[0] if labels else display_pat
                if best_score is None or score < best_score:
                    best_score = score
                    candidates = [(nid, label)]
                elif score == best_score:
                    candidates.append((nid, label))

        for nid, label in candidates:
            node_labels[nid][label] = None

    for nid in sorted(node_labels.keys(), key=int):
        all_labels = list(node_labels[nid].keys())
        file_node_labels[str(next_id)] = "\\n".join(all_labels)
        file_edges.append((nid, str(next_id)))
        next_id += 1

    return file_node_labels, file_edges


# ---------------------------------------------------------------------------
# DOT emission
# ---------------------------------------------------------------------------


def _legend_lines(has_highlight: bool) -> list[str]:
    """Return DOT lines for the legend subgraph."""
    items = []
    if has_highlight:
        items.append(
            f'        _l0[label="gb-model subsystem rule", style="rounded,filled", shape=box, color={_COLOUR_HIGHLIGHT}, fillcolor="{_COLOUR_HIGHLIGHT_FILL}"];'
        )
    items += [
        f'        _l1[label="gb-model general rule", style="rounded,filled", shape=box, color={_COLOUR_GB_MODEL}, fillcolor="{_COLOUR_GB_MODEL_FILL}"];',
        f'        _l2[label="PyPSA-Eur rule", style="rounded,filled", shape=box, color={_COLOUR_UPSTREAM}, fillcolor="{_COLOUR_UPSTREAM_FILL}"];',
        f'        _l3[label="output file", shape=note, style=filled, fillcolor={_COLOUR_FILE_FILL}, color={_COLOUR_FILE_BORDER}, fontsize=9];',
    ]
    node_ids = (["_l0"] if has_highlight else []) + ["_l1", "_l2", "_l3"]
    items.append(f"        {' -> '.join(node_ids)} [style=invis];")
    return [
        "    subgraph cluster_legend {",
        '        label="Legend";',
        "        fontname=sans; fontsize=10; style=filled;",
        '        fillcolor="#f5f5f5"; color="#999999";',
        "        node[penwidth=2];",
        *items,
        "    }",
    ]


def emit_dot(
    keep_nodes: set[str],
    gb_model_nodes: set[str],
    highlight_nodes: set[str],
    all_nodes: dict[str, str],
    edges: list[tuple[str, str]],
    file_node_labels: dict[str, str],
    file_edges: list[tuple[str, str]],
    size: str | None = None,
) -> str:
    """Render the filtered graph with file nodes and a legend as DOT."""
    graph_attrs = 'bgcolor=white, margin=0, pack=true, packmode="node"'
    if size:
        graph_attrs += f', size="{size}", ratio=fill'
    lines = [
        "digraph snakemake_dag {",
        f"    graph[{graph_attrs}];",
        "    node[shape=box, style=rounded, fontname=sans, fontsize=10, penwidth=2];",
        "    edge[penwidth=2, color=grey];",
    ]
    for nid in sorted(keep_nodes, key=int):
        if nid in highlight_nodes:
            color, fillcolor = _COLOUR_HIGHLIGHT, _COLOUR_HIGHLIGHT_FILL
        elif nid in gb_model_nodes:
            color, fillcolor = _COLOUR_GB_MODEL, _COLOUR_GB_MODEL_FILL
        else:
            color, fillcolor = _COLOUR_UPSTREAM, _COLOUR_UPSTREAM_FILL
        lines.append(
            f'    {nid}[label = "{all_nodes[nid]}", style="rounded,filled", shape=box, peripheries=1, color={color}, fillcolor="{fillcolor}"];'
        )
    for src, dst in edges:
        lines.append(f"    {src} -> {dst};")
    for file_nid, label in sorted(file_node_labels.items(), key=lambda x: int(x[0])):
        lines.append(
            f'    {file_nid}[label = "{label}", shape=note, style=filled, '
            f"fillcolor={_COLOUR_FILE_FILL}, color={_COLOUR_FILE_BORDER}, fontsize=9];"
        )
    for rule_nid, file_nid in file_edges:
        lines.append(
            f"    {rule_nid} -> {file_nid} [style=dashed, color={_COLOUR_FILE_BORDER}];"
        )
    lines += _legend_lines(bool(highlight_nodes))
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.argument("targets", nargs=-1, required=True)
@click.option(
    "-w",
    "--wildcard",
    "keep_wildcards",
    multiple=True,
    metavar="NAME",
    help="Wildcard name to keep as {name} in file-node labels (repeatable).",
)
@click.option(
    "-f",
    "--rule-file",
    "rule_files",
    multiple=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    metavar="FILE",
    help="Highlight rules defined in this .smk file (repeatable).",
)
@click.option(
    "-c",
    "--cutoff-rule",
    "cutoff_rules",
    multiple=True,
    metavar="RULE",
    help="Rule name above which all ancestor rules are pruned from the graph (repeatable).",
)
@click.option(
    "-s",
    "--size",
    default="10,8",
    show_default=True,
    metavar="W,H",
    help="Graph size in inches (width,height); sets ratio=fill for consistent aspect ratio.",
)
def main(
    targets: tuple[str, ...],
    keep_wildcards: tuple[str, ...],
    rule_files: tuple[Path, ...],
    cutoff_rules: tuple[str, ...],
    size: str,
) -> None:
    """
    Filter a Snakemake rulegraph to gb-model rules and emit DOT to stdout.

    TARGETS are the snakemake target file paths; they are passed to
    ``snakemake --rulegraph -F`` and also used to fill concrete wildcard
    values into the output file-node labels.
    """
    gb_model_rules = get_gb_model_rules(SMK_DIR)
    rule_output_patterns = parse_rule_outputs(SMK_DIR)
    dot_input = get_rulegraph_dot(list(targets))

    keep_nodes, gb_model_nodes, all_nodes, edges = filter_graph(
        dot_input, gb_model_rules
    )

    if cutoff_rules:
        cutoff_nids = {
            nid
            for nid, label in all_nodes.items()
            if label in set(cutoff_rules) and nid in keep_nodes
        }
        if cutoff_nids:
            ancestors = _ancestors(cutoff_nids, _build_rev_adj(edges)) - cutoff_nids
            keep_nodes -= ancestors
            gb_model_nodes &= keep_nodes
            edges = [(s, d) for s, d in edges if s in keep_nodes and d in keep_nodes]

    highlight_nodes = {
        nid
        for nid, label in all_nodes.items()
        if label in _rule_names_in_files(rule_files)
    } & keep_nodes
    file_node_labels, file_edges = build_file_nodes(
        keep_nodes,
        gb_model_nodes,
        all_nodes,
        edges,
        list(targets),
        set(keep_wildcards),
        rule_output_patterns,
    )
    click.echo(
        emit_dot(
            keep_nodes,
            gb_model_nodes,
            highlight_nodes,
            all_nodes,
            edges,
            file_node_labels,
            file_edges,
            size=size or None,
        )
    )


if __name__ == "__main__":
    main()

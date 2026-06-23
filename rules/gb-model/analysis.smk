# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Rules for generating output tables and output/intermediate plots for analysis
"""


rule plot_regions_and_network:
    message:
        "Plot ETYS boundary capabilities with interactive map"
    input:
        shapes=resources("gb-model/merged_shapes.geojson"),
        etys_caps=resources("gb-model/etys_boundary_capabilities.csv"),
        network=resources("networks/base_extended.nc"),
        boundaries="data/gb-model/downloaded/gb-etys-boundaries.zip",
    params:
        voltages=config["electricity"]["voltages"],
        interconnector_options=config["interconnectors"]["options"],
        interconnector_plan=config["interconnectors"]["plan"],
        year_range=config["redispatch"]["year_range_incl"],
    output:
        html=resources("gb-model/plots/regions_and_network.html"),
    log:
        logs("plot_regions_and_network.log"),
    resources:
        mem_mb=2000,
    script:
        scripts("gb_model/analysis/plot_regions_and_network.py")

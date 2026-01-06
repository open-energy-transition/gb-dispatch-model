# SPDX-FileCopyrightText: gb-open-market-model contributors
#
# SPDX-License-Identifier: CC-BY-4.0


rule osm_name_mapper:
    message:
        "Create name to ID mapper for OSM data"
    input:
        raw_cables_way="data/osm-raw/GB/cables_way.json",
        raw_lines_way="data/osm-raw/GB/lines_way.json",
        raw_routes_relation="data/osm-raw/GB/routes_relation.json",
        raw_substations_way="data/osm-raw/GB/substations_way.json",
        raw_substations_relation="data/osm-raw/GB/substations_relation.json",
        network=resources("networks/base.nc"),
    output:
        osm_mapping=resources("gb-model/osm_name_mapping.csv"),
    log:
        logs("osm_name_mapper.log"),
    script:
        "../scripts/gb_model/osm_name_mapper.py"


rule add_noa_options:
    message:
        "Adds NOA options to the model"
    params:
        noa_options=config["noa_options"],
        noa_sets=config["noa_sets"],
        noa_sets_selected=config["noa_sets_selected"],
    input:
        network=resources("networks/base_extended.nc"),
        osm_mapping_csv=resources("gb-model/osm_name_mapping.csv"),
    output:
        network=resources("networks/base_extended_noa.nc"),
    log:
        logs("add_noa_options.log"),
    script:
        "../scripts/gb_model/add_noa_options.py"


use rule simplify_network as simplify_network_noa with:
    message:
        "Simplify NOA extended network"
    input:
        network=resources("networks/base_extended_noa.nc"),
        regions_onshore=resources("regions_onshore.geojson"),
        regions_offshore=resources("regions_offshore.geojson"),
        admin_shapes=resources("admin_shapes.geojson"),
    log:
        logs("simplify_network_noa.log"),
    benchmark:
        benchmarks("simplify_network_noa")


ruleorder: simplify_network_noa > simplify_network

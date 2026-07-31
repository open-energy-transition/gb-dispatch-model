# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Links/Lines (incl. interconnectors, AC & HVDC) component rules.
"""


rule extract_transmission_availability:
    input:
        pdf_report="data/gb-model/downloaded/transmission-availability-{report_year}.pdf",
    output:
        csv=resources("gb-model/transmission-availability-monthly/{report_year}.csv"),
    log:
        logs("extract_transmission_availability_{report_year}.log"),
    script:
        scripts("gb_model/transmission/extract_transmission_availability.py")


rule process_transmission_availability:
    input:
        unavailability=expand(
            resources("gb-model/transmission-availability-monthly/{year}.csv"),
            year=config["transmission_availability"]["years"],
        ),
    output:
        csv=resources("gb-model/{transmission_zone}_transmission_availability.csv"),
    log:
        logs("process_transmission_availability_{transmission_zone}.log"),
    wildcard_constraints:
        transmission_zone="inter_gb|intra_gb",
    params:
        zones=lambda wildcards: config["transmission_availability"][
            wildcards.transmission_zone
        ]["zones"],
        sample_hourly=lambda wildcards: config["transmission_availability"][
            wildcards.transmission_zone
        ]["sample_hourly"],
        random_seeds=config["transmission_availability"]["random_seeds"],
    message:
        "Process {wildcards.transmission_zone} transmission availability stats into timeseries availability fractions."
    script:
        scripts("gb_model/transmission/process_transmission_availability.py")


rule create_interconnectors_table:
    input:
        regions=resources("gb-model/merged_shapes.geojson"),
    output:
        gsp_data=resources("gb-model/{fes_scenario}/interconnectors_p_nom.csv"),
    log:
        logs("create_interconnectors_table_{fes_scenario}.log"),
    params:
        interconnector_options=config["interconnectors"]["options"],
        interconnector_plan=config["interconnectors"]["plan"],
        year_range=config["redispatch"]["year_range_incl"],
        target_crs=config["target_crs"],
    message:
        "Create interconnector table for {wildcards.fes_scenario} scenario using configured commissioning plan."
    script:
        scripts("gb_model/transmission/create_interconnectors_table.py")


rule identify_regions_for_offshore_buses:
    input:
        regions=resources("gb-model/merged_shapes.geojson"),
        base_network=resources("networks/base.nc"),
    output:
        csv=resources("gb-model/custom_busmap.csv"),
    log:
        logs("identify_regions_for_offshore_buses"),
    message:
        "Identify the regions that offshore stub buses should be linked to"
    script:
        scripts("gb_model/transmission/identify_regions_for_offshore_buses.py")

# SPDX-FileCopyrightText: gb-dispatch-model contributors
#
# SPDX-License-Identifier: MIT

"""
Storage unit component rules.
"""


rule fetch_storage_max_hours:
    message:
        "Process FES workbook sheet ES1 to get max_hours for storage plants"
    params:
        year_range=config["redispatch"]["year_range_incl"],
        tech_mapping=config["fes"]["gb"]["generators_and_storage"]["carrier_mapping"],
    input:
        es1_sheet=resources(f"gb-model/fes/ES1.csv"),
    output:
        max_hours=resources("gb-model/{fes_scenario}/max_hours.csv"),
    log:
        logs("fetch_storage_max_hours_{fes_scenario}.log"),
    script:
        scripts("gb_model/storage/fetch_storage_max_hours.py")

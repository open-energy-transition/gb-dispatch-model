..
  SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: gb-dispatch-model contributors

  SPDX-License-Identifier: CC-BY-4.0

.. _gb_model_config:

##########################################
Configuration
##########################################

gb-dispatch-model has several additional configuration options above and beyond those set by PyPSA-Eur, which are documented in this section.
For base PyPSA-Eur configuration, see :ref:`config`.

Configuration Files
===================

Any gb-dispatch-model configuration can be set in a ``.yaml`` file.
The default configuration ``config/config.default.gb.yaml`` is maintained in the repository and covers all the options that are used / can be set.
The configuration ``config/config.gb.2024.yaml`` is an opinionated configuration we maintain that uses FES2024 data and is always applied on top of ``config/config.default.gb.yaml``.

To pass your own configuration, you can create a new file, e.g. ``my_config.yaml``, and specify the options you want to change.
They will override the default settings and options which are not set, will be inherited from the defaults above.

Another way is to use the ``config/config.yaml`` file, which does not exist in the repository and is also not tracked by git.
But snakemake will always use this file if it exists.
This way you can run snakemake with a custom config without having to specify the config file each time.

Configuration order of precedence is as follows:
1. Command line options specified with ``--config`` (optional)
2. Custom configuration file specified with ``--configfile`` (optional)
3. The ``config/config.yaml`` file (optional)
4. The default configuration files ``config/config.default.yaml`` and ``config/plotting.default.yaml``

To use your custom configuration file, you need to pass it to the ``snakemake`` command using the ``--configfile`` option:

.. code:: console

    $ snakemake -call --configfile my_config.yaml

.. _fes_costs_cf:

``fes_costs``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/fes_costs
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#fes_costs
   :end-before: # docs

.. _low_carbon_register_cf:

``low_carbon_register``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/low_carbon_register
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#low_carbon_register
   :end-before: # docs

.. _chp_cf:

``chp``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/chp
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#chp
   :end-before: # docs

.. _urls_cf:

``urls``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/urls
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#urls
   :end-before: # docs

.. _target_crs_cf:

``target_crs``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/target_crs
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#target_crs
   :end-before: # docs

.. _region_operations_cf:

``region_operations``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/region_operations
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#region_operations
   :end-before: # docs

.. _etys_cf:

``etys``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/etys
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#etys
   :end-before: # docs

.. _entsoe_unavailability_cf:

``entsoe_unavailability``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/entsoe_unavailability
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#entsoe_unavailability
   :end-before: # docs

.. _transmission_availability_cf:

``transmission_availability``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/transmission_availability
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#transmission_availability
   :end-before: # docs

.. _dukes_5_11_cf:

``dukes-5.11``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/dukes_5_11
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#dukes-5-11
   :end-before: # docs

.. _grid_supply_points_cf:

``grid_supply_points``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/grid_supply_points
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#grid_supply_points
   :end-before: # docs

.. _fes_cf:

``fes``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/fes
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#fes
   :end-before: # docs

.. _ev_cf:

``ev``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/ev
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#ev
   :end-before: # docs

.. _interconnectors_cf:

``interconnectors``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/interconnectors
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#interconnectors
   :end-before: # docs

.. _redispatch_cf:

``redispatch``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/redispatch
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#redispatch
   :end-before: # docs


``time_aggregation``
===============================

.. jsonschema:: ../../config/schema.default.gb.json#/properties/time_aggregation
   :lift_description:
   :hide_key: /**/additionalProperties

**YAML Syntax**

.. literalinclude:: ../../config/config.default.gb.yaml
   :language: yaml
   :start-after: # docs in https://pypsa-eur.readthedocs.io/en/latest/configuration.html#time_aggregation
   :end-before: # docs


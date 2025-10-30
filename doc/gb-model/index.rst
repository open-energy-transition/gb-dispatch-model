..
  SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: gb-dispatch-model contributors

  SPDX-License-Identifier: CC-BY-4.0

##################################################################################
gb-dispatch-model: Great Britain dispatch model built on the PyPSA-Eur workflow
##################################################################################

About
=====

gb-dispatch-model is an extension of `PyPSA-Eur <../index.html>`_., used to quantify dispatch decisions in Great Britain under the conditions set out by the UK Future Energy Scenarios.

Workflow
========

.. image:: img/workflow.png
    :class: full-width
    :align: center

.. note::
    The graph above was generated using
    ``snakemake --rulegraph -F | sed -n "/digraph/,/}/p" | dot -Tpng -o workflow.png``



Operating Systems
=================

The gb-dispatch-model workflow is continuously tested for Linux, macOS and Windows (WSL only).


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Configuration

   configuration
   costs


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Development

   implementation

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: References

   release_notes
   data_sources
   ../index
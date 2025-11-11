.. SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
.. SPDX-FileCopyrightText: Contributors to gb-dispatch-model
..
.. SPDX-License-Identifier: CC-BY-4.0

.. _installation:

##########################################
Installation
##########################################

The subsequently described installation steps are demonstrated as shell commands, where the path before the ``%`` sign denotes the
directory in which the commands following the ``%`` should be entered.

Clone the Repository
====================

First of all, clone the `gb-dispatch-model repository <https://github.com/open-energy-transition/gb-dispatch-model>`__ using the version control system ``git`` in the command line.

.. code:: console

    $ git clone https://github.com/open-energy-transition/gb-dispatch-model.git


.. _deps:

Install Python Dependencies
===============================

gb-dispatch-model relies on a set of other Python packages to function. We recommend
using the package manager `conda <https://docs.anaconda.com/miniconda/>` or
`mamba <https://mamba.readthedocs.io/en/latest/>`__ to install them and manage
your environments.

The package requirements are curated in the ``envs/environment.yaml`` file.
Since you cannot install these together directly, you should use one of the regularly updated locked environment files for each platform.
Choose the correct file for your platform:

* For Intel/AMD processors:

  - Linux: ``envs/linux-64.lock.yaml``

  - macOS: ``envs/osx-64.lock.yaml``

  - Windows: ``envs/win-64.lock.yaml``

* For ARM processors:

  - macOS (Apple Silicon): ``envs/gb-model/osx-arm64.lock.yaml``

  - Linux (ARM): Currently not supported via lock files; requires building certain packages, such as ``PySCIPOpt``, from source

.. code:: console

    $ conda update conda

    $ conda env create -n gb-dispatch-model -f envs/linux-64.lock.yaml # select the appropriate file for your platform

    $ conda activate gb-dispatch-model

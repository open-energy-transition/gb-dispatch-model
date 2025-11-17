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

Create working environment
==========================

gb-dispatch-model relies on a set of other Python packages to function.
We manage these using `pixi <https://pixi.sh/latest/>`_.
Once pixi is installed, you can activate the project environment (``gb-model``) for your operating system and have access to all the PyPSA-Eur dependencies from the command line:

.. code:: console

    $ pixi -e gb-model shell

.. tip::
    You can also set up automatic shell activation in several popular editors (e.g. in `VSCode <https://pixi.sh/dev/integration/editor/vscode/>`_ or `Zed <https://pixi.sh/dev/integration/editor/zed/>`_).
    Refer to the ``pixi`` documentation for the most up-to-date options.

.. note::
    We don't currently support linux operating systems using ARM processors since certain packages, such as ``PySCIPOpt``, require being built from source.

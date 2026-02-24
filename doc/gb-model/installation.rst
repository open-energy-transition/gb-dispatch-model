..
  SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _gb_installation:

##########################################
Installation
##########################################

The subsequently described installation steps are demonstrated as shell commands.
To use them in your shell, copy all but the initial ``$`` symbol.

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

Access API keys
===============

For successful execution, you will need the ENTSO-E Transparency platform API key, stored in a `.env` file in your working directory.
To get this key, first create an `ENTSO-E transparency platform <https://transparency.entsoe.eu/>`_ account.
Then, contact the ENTSO-E helpdesk by emailing ``transparency@entsoe.eu`` with the email subject ``Restful API access`` and email body being just the email address associated with your account.

Your `.env` file should then look like:

.. code::

    ENTSO_E_API_KEY=<your-api-key-here>


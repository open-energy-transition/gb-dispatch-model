..
  SPDX-FileCopyrightText: Contributors to PyPSA-Eur <https://github.com/pypsa/pypsa-eur>
  SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model>

  SPDX-License-Identifier: CC-BY-4.0

.. _installation_gb:

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

Create working environment
==========================

gb-dispatch-model relies on a set of other Python packages to function.
We manage these using `pixi <https://pixi.sh/latest/>`_.
To install ``pixi``, follow their `installation instructions <https://pixi.prefix.dev/latest/installation/>`_.
You can also install ``pixi`` into a ``conda`` environment using the command line:

.. code:: console

    conda install pixi

Once pixi is installed, you can run various tasks in the command line.
See our ``Running`` page for more information.

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


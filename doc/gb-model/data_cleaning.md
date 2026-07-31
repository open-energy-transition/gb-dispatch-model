..
  SPDX-FileCopyrightText: Contributors to gb-dispatch-model <https://github.com/open-energy-transition/gb-dispatch-model>

  SPDX-License-Identifier: CC-BY-4.0

.. data_cleaning:

#############
Data cleaning
#############

We undertake various data cleaning tasks to prepare input data for use in the model.
Here, we describe some of the more involved tasks to help users and maintainers understand our reasoning.

Offshore wind bus "stubs"
=========================

GB has many transmission lines in the downloaded OpenStreetMap network that originate offshore.
These come from offshore wind farms and are network "stubs", i.e., they represent end points in the network, rather than intermediate points.
They can become problematic further down the line as network clustering can lead to these stubs being assigned to onshore regions that differ from the region they're actually connected to.
For instance, if a wind farm is geographically close to region ``A`` but its power line connects to region ``B`` then on clustering the network, in which substations outside the explicitly defined model regions are attached to their geographically closest region, it will move the wind farm substation to region ``A``.
This then inadvertently creates a transmission line connection between regions ``A`` and ``B``.

PyPSA-Eur does offer some functionality to handle cleaning of stubs before network clustering.
However, we found that it didn't always work as expected.
It would delete legitimate lines that were ending on islands (these *are* indeed stubs).
It would also sometimes *miss* lines from offshore wind farms.

So, we have introduced our own stub cleaning mechanism.
What we do is:

1. Identify all offshore buses linked to the GB network.
   These are buses that are connected to an AC line that originates somewhere onshore in GB but isn't itself onshore in GB.
2. Identify the onshore bus at the other end of the connecting line.
3. Map the offshore bus to the onshore bus region and supply this busmap to the PyPSA-Eur clustering rule.

Here, we can see our identified offshore buses across GB (marked in red):

.. image:: img/identified_offshore_stubs.png

If we zoom into the north Wales coast, we can see one of these buses is very close to region ``GB 15`` but is actually connected to the mainland at a substation in ``GB 22``:

.. image:: img/identified_offshore_stubs_GB22.png

Using our cleaning approach, we are able to correctly map it to ``GB 22``.
Without this cleaning, we would have introduced a line that connects ``GB 15`` and ``GB 22`` directly, leapfrogging intermediate regions ``GB 17`` and ``GB 18``.
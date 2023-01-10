.. Scheduling Sandbox documentation master file, created by
   sphinx-quickstart on Wed Jul 20 09:04:25 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. image:: images/Logo.svg
   :width: 100%
   :alt: logo

Schlably
--------
**A Python-Based Scheduling Laboratory for Reinforcement Learning Solutions**

Schlably is a Python-Based framework for experiments on scheduling problems with Deep Reinforcement Learning (DRL).
It features an extendable gym environment and DRL-Agents along with code for data generation, training and testing.

Schlably was developed such that modules may be used as they are, but also may be customized to fit the needs of the user.
While the framework works out of the box and can be adjusted through config files, some changes are intentionally only possible through changes to the code.
We believe that this makes it easier to apply small changes without having to deal with complex multi-level inheritances.

Github: https://github.com/tmdt-buw/schlably

Paper: arxiv link

Main Features
-------------
- Python-based
- Includes JSSP, FJSSP and the option for resource constrainted scheduling problems
- Includes DRL-Agents
- Easy to use
- Easy to extend and adjust

User Guide
----------
.. toctree::
   :maxdepth: 2

   installation
   quickstart
   tutorials
   api



Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

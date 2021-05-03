
**************************************************************
Evoalg, a scalable and flexible evolutionary algorithm library
**************************************************************

Warning: This library is not finished yet, it is not in a usable state.

.. toctree::
   :maxdepth: 2

Overview
========

Evoalg is a library for various evolutionary algorithms implemented with ask-tell interface.

The library has 2 clear goals:
 * Being scalable (millions of parameters with hundreds of CPUs across multiple nodes). 
   Scalability is achieved by passing random table indices or random seeds instead of large parameter vectors.
 * Being flexible. Flexibility is achieved by defining a simple interface between the three abstract components 
   (Task, Parralelizaton, Algorithm) which allows minimal effort switching of any of the three components without changing the others.




Difference to other libraries
=============================

Evolag is aiming to fill the market gap in evolutionary algorithm libraries of simultaneously being fast and scalable, while also being simple and flexible.
There are plenty of simple, feature-rich and flexible libraries, but they are not designed to be running on a CPU clusters.
While the implementations which are designed for running on a cluster (eg open ai ES, uber ai GA) are not flexible, they complect different conepts.

   
Planned algorithms
==================

 * ES, E-ES, QE-ES, Novelty-ES
 * GA,Random-Search
 * NSGA-II
 * ES Map Elites, Evolvability Map elites
 * ES-MAML
 * CMA-ES (using py-cma)
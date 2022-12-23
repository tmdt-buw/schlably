Benchmark Datasets
==================

Schlably comes with pre-set benchmark datasets. These datasets include 50 instances each and are located
in the *data/benchmark_instances/jssp* and *data/benchmark_instances/jssp* folder along with solutions for each instance,
respectively for JSSP and FJSSP instances.

How the instances were generated
--------------------------------
**Job**

As common for the JSSPs and FJSSPS, all instances consist of jobs consisting of tasks.

**Task**

Each task has a **processing time** that is sampled uniformly from a list of processing times.
In these benchmark datasets, these processing times were [2, 4, 6, 8, 10].
In the JSSP, the number of tasks is the same as the number of machines and each task is assigned
to exactly one machines.

In the FJSSP, the number of tasks is the same as the number of machines and each task can be assigned to
one of multiple machines. The number of applicable machines per task was sampled uniformly from the interval
[1, number_of_machines] and the machines were sampled uniformly from the set of all machines.

Whats more, we provide datasets where an additional resource (tools), much like the machines, is needed
per task. The logic is exactly the same as for the machines, but the number of tools is either half
that of the number of jobs

Overview of the benchmark datasets
----------------------------------
The following table gives a brief overview of the benchmark datasets:

.. list-table:: Overview benchmark datasets
   :widths: 20 5 5 5 5 5
   :header-rows: 1

   * - Foldername/File
     - # jobs
     - # tasks p. job
     - # machines
     - (F)JSSP
     - # tools
   * - jssp/benchmark_job3_task4_tools0.pkl
     - 3
     - 4
     - 4
     - jssp
     - 0
   * - jssp/benchmark_job3_task4_tools3.pkl
     - 3
     - 4
     - 4
     - jssp
     - 3
   * - jssp/benchmark_job6_task6_tools0.pkl
     - 6
     - 6
     - 6
     - jssp
     - 0
   * - jssp/benchmark_job6_task6_tools3.pkl
     - 6
     - 6
     - 6
     - jssp
     - 3
   * - jssp/benchmark_job10_task10_tools0.pkl
     - 10
     - 10
     - 10
     - jssp
     - 0
   * - jssp/benchmark_job10_task10_tools5.pkl
     - 6
     - 6
     - 4
     - jssp
     - 5
   * - fjssp/benchmark_job3_task4_tools0.pkl
     - 3
     - 4
     - 4
     - fjssp
     - 0
   * - fjssp/benchmark_job3_task4_tools3.pkl
     - 3
     - 4
     - 4
     - fjssp
     - 3
   * - fjssp/benchmark_job6_task6_tools0.pkl
     - 6
     - 6
     - 6
     - fjssp
     - 0
   * - fjssp/benchmark_job6_task6_tools3.pkl
     - 6
     - 6
     - 6
     - fjssp
     - 3
   * - fjssp/benchmark_job10_task10_tools0.pkl
     - 10
     - 10
     - 10
     - fjssp
     - 0
   * - fjssp/benchmark_job10_task10_tools5.pkl
     - 6
     - 6
     - 4
     - fjssp
     - 5

Quickstart
=================

Data Generation
---------------

To create your own data, or more precisely, instances of a scheduling problem, proceed as follows:

#. Create a custom data generation configuration from one of the configs listed in *config/data_generation/fjssp* or *config/data_generation/jssp* (e.g. change number of machines, tasks, tools, runtimes etc.) to specify the generated instances.
#. Run

   .. code-block:: bash

      python -m src.data_generator.instance_factory -fp data_generation/jssp/<your_data_generation_config>.yaml


Training
--------

To train your own model, proceed as follows:

#. To train a model, you need to specify a training config, such as the ones already included in *config/training*. Note that different agents have different configs because the come with different algorithm parameters. You can customize the config to your needs, e.g. change the number of episodes, the learning rate, the batch size etc.
   We are using weights & biases (wandb) to track our results.
   If you want to track your results online, create your project at wandb.ai and set config parameter wandb_mode to 1 (offline tracking) or 2 (online tracking)
   and specify *wandb_project* parameter in config file and *wandb_entity* constant in the *logger.py* file.
#. Run

   .. code-block:: bash

      python -m src.agents.train -fp training/ppo/<your_training_config>.yaml


Immediately after training the model will be tested and benchmarked against all heuristics included in the TEST_HEURISTICS constant located in *src/agents/test.py*.
The trained model can be accessed via the experiment_save_path and saved_model_name you specified in the training config.


Testing
-------

As aforementioned, calling ``train.py`` automatically tests the model once training is complete. If you want to test a certain model again, or benchmark it against other heuristics, proceed as follows:

#. As in training, you need to point to a testing config file like the ones provided in *config/testing*.  You may change entries according to your needs.
   We provide a pre-trained masked PPO model. Thus, creating a config in *config/testing/ppo_masked* and assigning *example_ppo_masked_agent* to *saved_model_name* allows you to test without training first.

#. Run

   .. code-block:: bash

      python -m src.agents.test -fp testing/ppo/<your_testing_config>.yaml


#. Optionally, you may use the parameter --plot-ganttchart to plot the test results.

We have pre-implemented many common priority dispatching rules, such as Shortest Processing Time first, and a flexible optimal solver.


Advanced config handling
------------------------

On the one hand running data_generation, training or testing by specifying a config file path offers an easy and comfortable way to use this framework, but on the other hand it might seem a bit restrictive.
Therefore, there is the possibility to start all three files by passing a config dictionary to their main functions.
This comes in handy, if you need to loop across multiple configs or if you want to change single parameters without saving new config files.

.. note::

   Please check out the tutorials for further information on how to use the framework.
Weights & Biases sweeps
=======================

Weights & Biases (wandb) is a tool for tracking and visualizing machine learning experiments.
It's free for open source projects and has a free tier for commercial projects.
It's a great tool for tracking experiments, visualizing results, and sharing results with collaborators.

If you want to use wandb for hyperparameter sweeps, simply follow these instructions:

#. Edit *config/sweep/config_sweep_ppo.yaml* or create an own sweep config.

   * `Here <https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb>`_ you can find instructions for creating a configuration.

   * Make sure that you point to the right training config in your sweep config file.

   * In the training config which you are using as the basis for the sweep, make sure to track the right success metric. You find it under "benchmarking" along with instructions.

#. Run

   .. code-block:: bash

      wandb sweep -e your_entity -p your_project config/sweep/config_sweep_ppo.yaml to create a wandb sweep.

#. Run

   .. code-block:: bash

      wandb agent your_entity/your_project/sweep_name to start the sweep
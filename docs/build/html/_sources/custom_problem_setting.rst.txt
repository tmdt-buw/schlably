Custom Problem Settings
========================

To illustrate a typical use case consider that you want to compare the learning behaviors of two different PPO agents.
For simplicities sake, we will assume that the two agents are trained with the same general hyperparameters,
but differ in the problem setting and size and reward function:

**Agent One:**

* trained on 3x4 tool constrained JSSP instances. There are three unique tools in the problem setting, and each job
  needs exactly one of them.

* reward function:

   .. math::

      reward(t) = makespan(t-1) - makespan(t)


**Agent Two:**

* trained on 6x6 JSSP instances

* reward function:

   .. math::

      reward = \left\{ \begin{array}{rcl}
      0 & \mbox{if} & \mbox{intermediate step} \\
      makespan & \mbox{if} & \mbox{last step (schedule finished)}
      \end{array}\right.

Creating the problem instances
------------------------------
To create the instances for **Agent One**, we set up a new config file (it is easiest to copy an existing one)
in *config/data_generation/jssp* which we call *config_job3_task4_tools0.yaml* which contains the following:

.. code-block:: yaml

   ##############################################################################
   ###                         Data generation                                ###
   ##############################################################################

   # (R) [str] Type of your scheduling problem - this template is for jssp
   sp_type: jssp
   # (O)   [string]  Filename under the generated data will be saved (subdirectory is chosen by sp_type)
   instances_file: config_job3_task4_tools3.pkl
   # (O)   [int]     Seed for all pseudo random generators (random, numpy, torch)
   seed: 0
   # (R) [int] Number of jobs to be scheduled
   num_jobs: 3
   # (R) [int] Number of tasks per job to be scheduled (has to be equal to num_machines for jssp)
   num_tasks: 4
   # (R) [int] Number of available machines (has to be equal to num_tasks for jssp)
   num_machines: 4
   # (O) [int] Number of available tools to be scheduled
   num_tools: 3
   # (O) [list[int]] Duration of tasks are samples uniformly from this list
   runtimes: [2, 4, 6, 8, 10]
   # (R) [int] Number of instances (instances of the scheduling problem) to be generated
   num_instances: 1250
   # (O) [int] Number of parallel processes used to calculate the instances
   num_processes: 1
   # (O) [bool] Save the generated data in a file
   write_to_file: True

After saving the file you can run the following command to generate the instances:

.. code-block::

   python -m src.data_generation.generate_data --fp config/data_generation/jssp/config_job3_task4_tools3.yaml

You should find the file in *data/jssp/config_job3_task4_tools3.pkl*.

To create the instances for **Agent Two**, we set up a new config file (it is easiest to copy an existing one)
in *config/data_generation/jssp* which we call *config_job6_task6_tools0.yaml* which contains the following:

.. code-block:: yaml

   ##############################################################################
   ###                         Data generation                                ###
   ##############################################################################

   # (R) [str] Type of your scheduling problem - this template is for jssp
   sp_type: jssp
   # (O)   [string]  Filename under the generated data will be saved (subdirectory is chosen by sp_type)
   instances_file: config_job6_task6_tools0.pkl
   # (O)   [int]     Seed for all pseudo random generators (random, numpy, torch)
   seed: 0
   # (R) [int] Number of jobs to be scheduled
   num_jobs: 6
   # (R) [int] Number of tasks per job to be scheduled (has to be equal to num_machines for jssp)
   num_tasks: 6
   # (R) [int] Number of available machines (has to be equal to num_tasks for jssp)
   num_machines: 6
   # (O) [int] Number of available tools to be scheduled
   num_tools: 0
   # (O) [list[int]] Duration of tasks are samples uniformly from this list
   runtimes: [2, 4, 6, 8, 10]
   # (R) [int] Number of instances (instances of the scheduling problem) to be generated
   num_instances: 1250
   # (O) [int] Number of parallel processes used to calculate the instances
   num_processes: 1
   # (O) [bool] Save the generated data in a file
   write_to_file: True

After saving the file you can run the following command to generate the instances:

.. code-block::

   python -m src.data_generation.generate_data --fp config/data_generation/jssp/config_job6_task6_tools0.yaml

You should find the file in *data/jssp/config_job6_task6_tools0.pkl*.

Training the agents
-------------------
To train the agents, we set up two new config files (again, it is easiest to copy an existing one)
in *config/training/jssp* which we call *config_job3_task4_tools3.yaml* and *config_job6_task6_tools0.yaml*.

*config_job3_task4_tools3.yaml* should look like this:

.. code-block:: yaml

   ##############################################################################
   ###                         Training                                       ###
   ##############################################################################

   # (R)   [String]  RL algorithm you want to use - This template is for PPO
   algorithm: ppo_masked
   # (R)   [string]  Path to the file with generated data that to be used for training
   instances_file: jssp/config_job3_task4_tools0.pkl
   # (O)   [string]  The finished model is saved under this name. Alternatively set to <automatic>, then it will be
                     # replaced with the current DayMonthYearHourMinute
   saved_model_name: example_ppo_masked_agent
   # (R)   [int]     Seed for all pseudo random generators (random, numpy, torch)
   seed: 2
   # (O)   [bool]    Bool, if the train-test-split of instances should remain the same (1111) and be independent of the
                     # random seed. This is useful for hyperparameter-sweeps with multiple random seeds to keep the same
                     # test instances for comparability. Irrelevant, if the random seed across runs remains the same.
   overwrite_split_seed: False
   # (R)   [string]  Set an individual description that you can identify this training run  more easily later on.
                     # This will be used in "weights and biases" as well
   config_description: tasks_3x4
   # (O)   [string]  Set a directory where you want to save the agent model
   experiment_save_path: models
   # (O)   [int]:    Wandb mode choose: choose from [0: no wandb, 1: wandb_offline, 2: wandb_online]
   wandb_mode: 2
   # (O)   [string]  Set a wandb project where you want to upload all wandb logs
   wandb_project: ppo_tutorial_test

   # --- PPO parameter ---
   # (O)   [int]     Number of steps collected before PPO updates the policy again
   rollout_steps: 2048
   # (O)   [float]   Factor to discount future rewards
   gamma: 0.99
   # (O)   [int]     Number of epochs the network gets fed with the whole rollout data, when training is triggered
   n_epochs: 5
   # (O)   [int]     Batch size into which the rollout data gets split
   batch_size: 256
   # (O)   [float]   Range of the acceptable deviation between the policy before and after training
   clip_range: 0.2
   # (O)   [float]   Entropy loss coefficient for the total loss calculation
   ent_coef: 0.0
   # (O)   [float]   Learning rate for the network updates
   learning_rate: 0.002
   # (O) List[int] List with dimension for the hidden layers (length of list = number of hidden layers) used in the policy net
   policy_layer:  [ 256, 256 ]
   # (O) [str] String for the activation function of the policy net
               # Note, the activation function has to be from the torch.nn module (e.g. ReLU)
   policy_activation: 'ReLU'
   # (O) List[int] List with dimension for the hidden layers (length of list = number of hidden layers) used in the value net
   value_layer:  [ 256, 256 ]
   # (O) [str] String for the activation function of the value net
               # Note, the activation function has to be from the torch.nn module (e.g. ReLU)
   value_activation: 'ReLU'
   # (R)   [int]     Maximum number of instances shown to the agent. Limits the training process. Note that instances may be
                     # multiple times, if total_instances is larger than the number of generated instances
   # (R) [int] Maximum number of instances shown to the agent. Limits the training process
   total_instances: 10_000
   # (R)   [int]     Maximum number of steps that the agent can interact with the env. Limits the training process
   total_timesteps: 1_000_000
   # (R)   [float]   Range between 0 and 1. How much (percentually) of the generated data will be used for training.
   train_test_split: 0.8
   # (R)   [float]   Range between 0 and 1. How much (percentually) of the remaining data (1-train_test_split) will be
                     # used for training.
   test_validation_split: 0.8
   # (R)   [int]     Number of environment step calls between intermediate (validation) tests
   intermediate_test_interval: 30_000

   # --- env (Environment) parameter ---
   # (R)   [str]     Environment you want to use. The vanilla case is env_tetris_scheduling.
   environment: env_tetris_scheduling
   # (O)   [int]     Maximum number of steps the agent can take before the env interrupts the episode.
                     # Should be greater than the minimum number of agent actions required to solve the problem.
                     # Can be larger that the minimum number of agent actions, if e.g. invalid actions or skip actions are
                     # implemented
   num_steps_max: 90
   # (O)   [int]     After this number of episodes, the env prints the last episode result in the console
   log_interval: 10
   # (O)   [bool]    All initial task instances are shuffled before being returned to the agent as observation
   shuffle: False
   # (O)   [str]     The reward strategy determines, how the reward is computed. Default is 'dense_makespan_reward'
   reward_strategy: dense_makespan_reward
   # (O)   [int]     The reward scale is a float by which the reward is multiplied to increase/decrease the reward signal
                     # strength
   reward_scale: 1

   # --- benchmarking
   # (R)   List[str] List of all heuristics and algorithms against which to benchmark
   test_heuristics: ['rand', 'EDD', 'SPT', 'MTR', 'LTR']
   # (O)   [str]     Metric name in the final evaluation table which summarizes the training success best. See
                     # EvaluationHandler.evaluate_test() in utils.evaluations for suitable metrics or add one.
                     # In a wandb hyperparameter sweep this will be usable as objective metric.
   success_metric:   makespan_mean

*config_job6_task6_tools0.yaml* should look like this:

.. code-block:: yaml

   ##############################################################################
   ###                         Training                                       ###
   ##############################################################################

   # (R)   [String]  RL algorithm you want to use - This template is for PPO
   algorithm: ppo_masked
   # (R)   [string]  Path to the file with generated data that to be used for training
   instances_file: jssp/config_job6_task6_tools0.pkl
   # (O)   [string]  The finished model is saved under this name. Alternatively set to <automatic>, then it will be
                     # replaced with the current DayMonthYearHourMinute
   saved_model_name: example_ppo_masked_agent
   # (R)   [int]     Seed for all pseudo random generators (random, numpy, torch)
   seed: 2
   # (O)   [bool]    Bool, if the train-test-split of instances should remain the same (1111) and be independent of the
                     # random seed. This is useful for hyperparameter-sweeps with multiple random seeds to keep the same
                     # test instances for comparability. Irrelevant, if the random seed across runs remains the same.
   overwrite_split_seed: False
   # (R)   [string]  Set an individual description that you can identify this training run  more easily later on.
                     # This will be used in "weights and biases" as well
   config_description: tasks_6x6
   # (O)   [string]  Set a directory where you want to save the agent model
   experiment_save_path: models
   # (O)   [int]:    Wandb mode choose: choose from [0: no wandb, 1: wandb_offline, 2: wandb_online]
   wandb_mode: 2
   # (O)   [string]  Set a wandb project where you want to upload all wandb logs
   wandb_project: ppo_tutorial_test

   # --- PPO parameter ---
   # (O)   [int]     Number of steps collected before PPO updates the policy again
   rollout_steps: 2048
   # (O)   [float]   Factor to discount future rewards
   gamma: 0.99
   # (O)   [int]     Number of epochs the network gets fed with the whole rollout data, when training is triggered
   n_epochs: 5
   # (O)   [int]     Batch size into which the rollout data gets split
   batch_size: 256
   # (O)   [float]   Range of the acceptable deviation between the policy before and after training
   clip_range: 0.2
   # (O)   [float]   Entropy loss coefficient for the total loss calculation
   ent_coef: 0.0
   # (O)   [float]   Learning rate for the network updates
   learning_rate: 0.002
   # (O) List[int] List with dimension for the hidden layers (length of list = number of hidden layers) used in the policy net
   policy_layer:  [ 256, 256 ]
   # (O) [str] String for the activation function of the policy net
               # Note, the activation function has to be from the torch.nn module (e.g. ReLU)
   policy_activation: 'ReLU'
   # (O) List[int] List with dimension for the hidden layers (length of list = number of hidden layers) used in the value net
   value_layer:  [ 256, 256 ]
   # (O) [str] String for the activation function of the value net
               # Note, the activation function has to be from the torch.nn module (e.g. ReLU)
   value_activation: 'ReLU'
   # (R)   [int]     Maximum number of instances shown to the agent. Limits the training process. Note that instances may be
                     # multiple times, if total_instances is larger than the number of generated instances
   # (R) [int] Maximum number of instances shown to the agent. Limits the training process
   total_instances: 10_000
   # (R)   [int]     Maximum number of steps that the agent can interact with the env. Limits the training process
   total_timesteps: 1_000_000
   # (R)   [float]   Range between 0 and 1. How much (percentually) of the generated data will be used for training.
   train_test_split: 0.8
   # (R)   [float]   Range between 0 and 1. How much (percentually) of the remaining data (1-train_test_split) will be
                     # used for training.
   test_validation_split: 0.8
   # (R)   [int]     Number of environment step calls between intermediate (validation) tests
   intermediate_test_interval: 30_000

   # --- env (Environment) parameter ---
   # (R)   [str]     Environment you want to use. The vanilla case is env_tetris_scheduling.
   environment: env_tetris_scheduling
   # (O)   [int]     Maximum number of steps the agent can take before the env interrupts the episode.
                     # Should be greater than the minimum number of agent actions required to solve the problem.
                     # Can be larger that the minimum number of agent actions, if e.g. invalid actions or skip actions are
                     # implemented
   num_steps_max: 90
   # (O)   [int]     After this number of episodes, the env prints the last episode result in the console
   log_interval: 10
   # (O)   [bool]    All initial task instances are shuffled before being returned to the agent as observation
   shuffle: False
   # (O)   [str]     The reward strategy determines, how the reward is computed. Default is 'dense_makespan_reward'
   reward_strategy: sparse_makespan_reward
   # (O)   [int]     The reward scale is a float by which the reward is multiplied to increase/decrease the reward signal
                     # strength
   reward_scale: 1

   # --- benchmarking
   # (R)   List[str] List of all heuristics and algorithms against which to benchmark
   test_heuristics: ['rand', 'EDD', 'SPT', 'MTR', 'LTR']
   # (O)   [str]     Metric name in the final evaluation table which summarizes the training success best. See
                     # EvaluationHandler.evaluate_test() in utils.evaluations for suitable metrics or add one.
                     # In a wandb hyperparameter sweep this will be usable as objective metric.
   success_metric:   makespan_mean

Note that the only changes had to be made to the file names and reward_strategy parameters.

Now all that is left is to log into Weights and Biases and start training. This will take some time, but you can follow
the progress in the console log and in the Weights and Biases dashboard.

.. code-block:: bash

   wandb login
   python -m src.agents.test -fp training/jssp/config_job3_task4_tools3.yaml
   python -m src.agents.test -fp training/jssp/config_job6_task6_tools0.yaml
# Sandbox Tutorial
​
As it is important to us that each user can suit this framework his individual requirements, 
we implemented a lot of files and functions with the intention that they can be adapted.
To make these adaptions as easy as possible, we ensured to give the framework a well-organized structure and good code documentation. 
Furthermore, we provide instructions for how to change things in addition to the instruction for the usage of the framework.

​
## Install
- To install all necessary packages run 
   ````bash
   pip install -r requirements.txt
   ````
- If you want to use [Weights&Biases](https://wandb.ai/site) (wandb) for logging you need to (create and) login with your account. Open a terminal and run:
   ````bash
   wandb login
   ````

​
## Basic agent training and testing
Data generation and training and testing of own agents can be performed without any coding just by changing the provided configuration files.
In the following, we describe what you can adapt and how to do so.

### Data Generation
- To create your own data, or more precisely, instances of a scheduling problem, proceed as follows:
1. Create a custom data generation configuration from one of the configs listed in  [config/data_generation/fjssp](config/data_generation/fjssp) or [config/data_generation/jssp](config/data_generation/jssp) (e.g. change number of machines, tasks, tools, runtimes etc.) to specify the generated instances  
2. Run 
   ``
   python -m src.data_generator.instance_factory -f data_generation/jssp/your_data_generation_config.yaml
   ``

To edit data_generation parameters for a jssp problem open [jssp/config_job3_task4_tools0.yaml](config/data_generation/jssp/config_job3_task4_tools0.yaml).
1. Problem dimension: Select the number of jobs, tasks, machines and tools for your instances
   ```` yaml
   num_jobs: 3
   num_tasks: 4
   num_machines: 4
   num_tools: 0
   ````
2. Runtimes: Set a list with runtimes. Task runtimes will be sampled random from this list during data generation
   ```` yaml
    runtimes: [2, 4, 6, 8, 10]
   ````

​
### Training
- To train your own model, proceed as follows:
1. Create a custom training configuration from one of the configs listed in [config/training](config/training). Note that different Agents have other configs due to different algorithm parameters.
    - We are using weights & biases (wandb) to track our results
   If you want to track your results online, create your project at wandb.ai and set config parameter wandb_mode to 1 (offline tracking) or 2 (online tracking)
   and specify *wandb_project* parameter in config file and *wandb_entity* constant in the [logger.py](src/utils/logger.py)
2. Run 
   ``
   python -m src.agents.train -f training/dqn/your_training_config.yaml
   ``

- Immediately after training the model will be tested and benchmarked against all heuristics included in the TEST_HEURISTICS constant located in [src/agents/test.py](src/agents/test.py)
- The trained model can be accessed via the experiment_save_path and saved_model_name you specified in the training config

To edit training parameters for PPO open [ppo/config_job3_task4_tools0.yaml](config/training/ppo/config_job3_task4_tools0.yaml).
1. File names: You can select you own generated instance file and set a name for teh save agent file

   ```` yaml
    instances_file: jssp/your_instance_file.pkl
    saved_model_name: your_agent
   ````
2. Algorithm parameters: Important PPO settings like learning rate or batch size can be controlled using these parameters 
   ```` yaml
   rollout_steps: 2048
   gamma: 0.99
   n_epochs: 5
   batch_size: 256
   clip_range: 0.2
   ent_coef: 0.0
   learning_rate: 0.002
   ````
3. Training length: The length of a training can be limited through both a total number of instances or steps of the environment that will be run. Depending on your training limit, you may change the frequency of the intermediate tests.
   ```` yaml
   total_instances: 100_000
   total_timesteps: 3_000_000
   ...
   intermediate_test_interval: 1_000
   ````

​
### Testing
- As aforementioned [train.py](src/agents/train.py) automatically tests the trained model. Anyway, if you want to test a certain model again, or benchmark it against other heuristics, proceed as follows:
1. Create a custom testing configuration from one of the configs listed in [config/testing](config/testing).  
We provide a pre-trained PPO model. Thus, creating a config in [config/testing/ppo](config/testing/ppo) and assigning *example_ppo_agent* to *saved_model_name* allows you to test without training first
2. Run 
   ``
   python -m src.agents.test -f testing/dqn/your_testing_config.yaml
   `` 
   and select your generated test config in the filedialog window
3. Use the parameter --plot-ganttchart to plot the test results


To edit testing parameters for DQN open [dqn/config_job3_task4_tools0.yaml](config/testing/dqn/config_job3_task4_tools0.yaml).
1. File names: Select the instance file you want to use as test data as well as the model you want to test
   ```` yaml
   instances_file: jssp/config_job3_task4_tools0.pkl
   ...
   saved_model_name: example_dqn_agent
   ````
2. Testing heuristics: You can choose the heuristics to be tested on the test instances
   ```` yaml
    test_heuristics: ['rand', 'EDD', 'MCM']
   ````

​
## Advanced agent training and testing
On the one hand running data_generation, training or testing by specifying a config file path offers an easy and comfortable way to use this framework, but on the other hand it is also a bit restricting. Therefore, there is the possibility to start all three files by passing an external config dictionary to their main functions.  
This enables you to change parameters in configs and use them without saving a new config file.
Especially if you want to loop over a certain parameters (e.g. seed) passing a dict can be a helpful option.

### Training via external config
1. Import the main function from agents.train to your script 
2. Load a config or specify an entire config
3. Loop over the parameter you want to change and update your config before starting a training 
   ```` python
   # import training main function
   from agents.train import main as training_main

   # load a default config
   train_default_config = ConfigHandler.get_config(DEFAULT_TRAINING_FILE)
   # specify your seeds
   seeds = [1455, 2327, 7776]
   
   # loop
   for seed in seeds:
      train_default_config.update({'seed': seed})
      
      # start training
      training_main(external_config=train_default_config)
   ````   


​
## Advanced data generation
According to your individual scheduling problem you may adapt the generation of instances beyond what is possible by customizing our *data_generation_config*.  
Therefore, all files  accountable for the data generation process and how they can be extended are explained below.  
We also recommend you to read the __task representation__ section of this tutorial, to have a better understanding of task structure used in the generation.

1. The [instance_factory.py](src/data_generator/instance_factory.py) can be executed to  generate new instances for a scheduling problem as explained in the __training in testing__ section.  
It automatically computes deadlines for tasks according to random heuristic. 
- You can change the heuristic used for computation by adapting DEADLINE_HEURISTIC constant at the beginn of the script
   ```` python
  # constants
  DEADLINE_HEURISTIC = 'spt'
  ````


​
2. There is a _generate_instances_ in the [sp_factory.py](src/data_generator/sp_factory.py) which will be called when generating instances.
- If you want to modify the task generation beyond what is possible by customizing the data_generation config, you can do it here (e.g. different random functions)


​
## Using the solver
- We are using a solver based on the Google OR-Tools [CP-SAT solver](https://developers.google.com/optimization/cp/cp_solver)
- Just like the agents and the heuristics, the solver uses instances in the form of lists of our task objects as input
- After computing a solution the solver outputs the same task list, but with an assigned starting time for each task

### Generate solutions with the solver
- First, make sure that you have generated an instance file
- To solve the instances, run ``python -m src.agents.solver.solver -fp jssp/config_job3_task4_tools0.pkl ``
- If you want to save the results as a solver solution add the ``-write`` parameter to the command
- To show the solutions as gantt charts, add ``-plot`` to the terminal


​
## Evaluation handling

### Default evaluation Metrics
- rew(ard): Reward that was returned from the environment
- tardiness: Sum of the tardiness of all jobs, measured in steps. If a job gets finished two steps after his deadline, he has a tardiness of 2 steps
- tardiness_max: Highest tardiness which occurs in this scheduling. E.g. job1 tardiness=1, job2 tardiness=3, job3 tardiness=0 -> tardiness_max=3 steps
- makespan: Total duration of the scheduling. Equal to step in that the last job gets finished
- When a parameter has an additional *_mean* or *_std* (standard deviation), these metrics were calculated over n test runs and for the certain parameter

### Adapt evaluation metrics
- All evaluation metrics are defined in [evaluations.py](src/utils/evaluations.py), more precisely in the ``evaluate_test()`` method of the `EvaluationHandler`class
- You can add or remove metrics here

### Gap to optimal solver solution as metric
- In addition to the aforementioned metrics, it's also possible to monitor the gap of a heuristic or agent result to an optimal solver solution
- Therefore, the solutions must be calculated during the test. Once computed, they are stored under their hash. If the same instance occurs again in a test run, the existing solution is loaded and processed  

​
## Heuristics
We provide several heuristics in [heuristic_agent.py](src/agents/heuristic/heuristic_agent.py) which can be used to solve scheduling problem instances and benchmark the results with other heuristics or agents
The following heuristics are implemented:
- Random task
- EDD: earliest due date
- SPT: shortest processing time first
- MTR: most tasks remaining
- LTR: least tasks remaining

### Add custom heuristics
The heuristics are located in [heuristic_agent.py](src/agents/heuristic/heuristic_agent.py). To add custom heuristics proceed as follows:
1. Define your function e.g. `def my_custom_heuristic()`
2. Consider yourself a shortcut for your heuristic e.g. __MCH__ and add assignment fo shortcut and function name to the `self.task_selection` attribute of the `HeuristicSelectionAgent` class.
    ```` python
    self.task_selections = {
        'rand': random_task,
        ...,
        'XYZ': my_custom_heuristic
    }
    ````
3. Now you can make use of your heuristic via the shortcut e.g. by add it to *test_heuristcs* parameter in a trainings config
   ```` yaml
    test_heuristics: ['rand', 'MCH']
   ````

​
## wandb Sweep
- If you want to use wandb for a parameter sweep simply follow these instructions
1. Edit [config/sweep/config_sweep_ppo.yaml](config/sweep/config_sweep_ppo.yaml) or create an own sweep config. [Here](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb) you can find instructions for creating a configuration.
2. Run ``wandb sweep -e your_entity -p your_project config/sweep/config_sweep_ppo.yaml`` to create a wandb sweep.
3. Run ``wandb agent your_entity/your_project/sweep_name`` to start the sweep

Your sweep configuration could look like this:
- Here, we are sweeping over five discrete seeds and the learning rate between two continuous values 
   ```` yaml
   method: random
   metric:
   goal: minimize
   name: success_metric
   parameters:
   seed:
     values: [1, 2, 3, 4, 5]
   learning_rate:
     max: 0.005
     min: 0.001

   program: src/agents/train.py

   # needed to run agents.train as module and avoid ModuleNotFoundError
   command:
   - python
   - -m
   - src.agents.train
   - -fp
   - training/ppo_masked/config_job3_task4_tools0.yaml

   ````

​
## Task representation
To represent tasks and jobs of a scheduling problem we implemented a __Task__ class which is located in [task.py](src/data_generator/task.py).  
A wide range of useful attributes is offered by the class which are explained here. 
 - Task objects are used to represent tasks and their dependencies in a real world production process
 - Tasks with same *job_num* can be viewed as one job
 - A list of task objects builds an instance, which can be put into the environment
 - Each task can be performed on *machines* and with *tool* and has a *runtime*
 - To follow the scheduling of an instance each task has a *started* and *finished* attribute 


​
## Environments
- The environments track the current scheduling state of all tasks, machines and tools and adjust the state according to valid incoming actions 
- Action and state space are initialized, depending on the number of tasks, machines and tools 
- As the [indirect_action_env](src/environments/env_tetris_scheduling_indirect_action.py) inherits  from the  [tetris_env](src/environments/env_tetris_scheduling.py), changes to the [tetris_env](src/environments/env_tetris_scheduling.py) also apply to the [indirect_action_env](src/environments/env_tetris_scheduling_indirect_action.py)

### Observation
If you want to change the observation shown to an agent just edit the `state_obs()` function which is always called when returning an observation. 
- For example, you can use different normalization. Currently, we apply min/max scaling to the observation
- By default, the observation maps the current runtime, id and deadline of each job next task. Feel free to use other parameters from the env or use them to calculate new ones. (e.g. number of free machines, current episode step, ...)

### Reward strategy
We use the `compute_reward()` strategy to call an implemented reward strategy according to the *reward_strategy* parameter in the training or testing config.  
By default, we implemented two reward strategies:
- Dense rewards according to [https://arxiv.org/pdf/2010.12367.pdf](https://arxiv.org/pdf/2010.12367.pdf)
- Sparse rewards
As aforementioned you can choose between them by adapting the *reward_strategy* parameter as follows:
   ```` yaml
    reward_strategy: dense_makespan_reward
   ````
  ```` yaml
    reward_strategy: sparse_makespan_reward
   ````
- If you want to implement your own reward strategy you can do it in [tetris_env](src/environments/env_tetris_scheduling.py) just as follows:
   ```` python
      def your_custom_reward_reward_strategy(self):
        ...
   ````
   and use by adapting the *reward_strategy* parameter
   ```` yaml
    reward_strategy: your_custom_reward_reward_strategy
   ````

​
## Agents
All agents we are using, currently DQN and PPO are implemented in PyTorch. Hence, network structures can be adapted by using different PyTorch objects.

### DQN
Our [DQN](src/agents/reinforcement_learning/dqn.py) implementation uses one policy for both the Q and the Target network.  
So feel free to replace the structure defined in the `__init__()` function of the `Policy` class.  
   ```` python
   class Policy(nn.Module):
      def __init__():
         # Your custom policy net
   ````

### PPO
To adapt the [PPO](src/agents/reinforcement_learning/ppo.py) note that there are separate nets for policy and value.  
Therefore, you can change the `__init__()` function of both the `PolicyNetwork` class and the `ValueNetwork` class.
   ```` python
    class PolicyNetwork(nn.Module):
      def __init__():
         # Your custom policy net
    
    class ValueNetwork(nn.Module):
      def __init__():
         # Your custom value net
   ````


​
## Logging
We log our results with wandb. But we created the [Logger class](src/utils/logger.py) to make the whole framework is independent of wandb.  
That gives you the possibility to either simply turn off the logging by setting the wandb_mode config parameter to 0 or proceed as follows and implement your logging method into the logger class.  

1. The `record()` function of the logger is used to save results as a key value pair into the logger buffer.
- For example, we are recording the ppo parameters after each policy update
   ```` python
   self.logger.record(
            {
                'agent_training/n_updates': self.n_updates,
                'agent_training/loss': np.mean(total_losses),
                ...
            }
            )
   ````
- You can record your own parameters. Note, make sure to dump the logger before recording the same key again, otherwise it will be overwritten
2. By default, the `dump()` method of the logger calls the `dump_wandb()` method (if wandb_mode is not 0). Feel free to implement and call your own dump method here (e.g. print to console, write to JSON file, ...)

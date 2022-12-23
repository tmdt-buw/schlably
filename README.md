# Schlably
​
Schlably is a Python-Based framework for experiments on scheduling problems with Deep Reinforcement Learning (DRL). 
It features an extendable gym environment and DRL-Agents along with code for data generation, training and testing.  


Schlably was developed such that modules may be used as they are, but also may be customized to fit the needs of the user.
While the framework works out of the box and can be adjusted through config files, some changes are intentionally only possible through changes to the code.
We believe that this makes it easier to apply small changes without having to deal with complex multi-level inheritances.

Please see the documentation for more detailed information and tutorials.
​
## Install
To install all necessary packages run 
   ````bash
   pip install -r requirements.txt
   ````
If you want to use [Weights&Biases](https://wandb.ai/site) (wandb) for logging, which we highly recommend,
you need to (create and) login with your account. Open a terminal and run:
   ````bash
   wandb login
   ````
  
​
## Quickstart

### Data Generation
To create your own data, or more precisely, instances of a scheduling problem, proceed as follows:
1. Create a custom data generation configuration from one of the configs listed in [config/data_generation/fjssp](config/data_generation/fjssp) or [config/data_generation/jssp](config/data_generation/jssp) (e.g. change number of machines, tasks, tools, runtimes etc.) to specify the generated instances.  
2. Run 
   ``
   python -m src.data_generator.instance_factory -fp data_generation/jssp/<your_data_generation_config>.yaml
   ``


​
### Training
To train your own model, proceed as follows:
1. To train a model, you need to specify a training config, such as the ones already included in [config/training](config/training). Note that different agents have different configs because the come with different algorithm parameters. You can customize the config to your needs, e.g. change the number of episodes, the learning rate, the batch size etc.
    - We are using weights & biases (wandb) to track our results.
   If you want to track your results online, create your project at wandb.ai and set config parameter wandb_mode to 1 (offline tracking) or 2 (online tracking)
   and specify *wandb_project* parameter in config file and *wandb_entity* constant in the [logger.py](src/utils/logger.py)
2. Run 
   ``
   python -m src.agents.train -fp training/ppo/<your_training_config>.yaml
   ``

Immediately after training the model will be tested and benchmarked against all heuristics included in the TEST_HEURISTICS constant located in [src/agents/test.py](src/agents/test.py)
The trained model can be accessed via the experiment_save_path and saved_model_name you specified in the training config.


​
### Testing
As aforementioned, [train.py](src/agents/train.py) automatically tests the model once training is complete. If you want to test a certain model again, or benchmark it against other heuristics, proceed as follows:
1. As in training, you need to point to a testing config file like the ones provided in [config/testing](config/testing).  You may change entries according to your needs.
We provide a pre-trained masked PPO model. Thus, creating a config in [config/testing/ppo_masked](config/testing/ppo_masked) and assigning *example_ppo_masked_agent* to *saved_model_name* allows you to test without training first.
2. Run 
   ``
   python -m src.agents.test -fp testing/ppo_masked/<your_testing_config>.yaml
   ``
3. Optionally, you may use the parameter --plot-ganttchart to plot the test results.

We have pre-implemented many common priority dispatching rules, such as Shortest Processing Time first, and a flexible optimal solver.

​
## Advanced config handling
On the one hand running data_generation, training or testing by specifying a config file path offers an easy and comfortable way to use this framework, but on the other hand it might seem a bit restrictive. 
Therefore, there is the possibility to start all three files by passing a config dictionary to their main functions. 
This comes in handy, if you need to loop across multiple configs or if you want to change single parameters without saving new config files.


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

  
## wandb Sweep
If you want to use wandb for hyperparameter sweeps, simply follow these instructions
1. Edit [config/sweep/config_sweep_ppo.yaml](config/sweep/config_sweep_ppo.yaml) or create an own sweep config. 
   1. [Here](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W%26B.ipynb) you can find instructions for creating a configuration.
   2. Make sure that you point to the right training config in your sweep config file. 
   3. In the training config which you are using as the basis for the sweep, make sure to track the right success metric. You find it under "benchmarking" along with instructions.
2. Run ``wandb sweep -e your_entity -p your_project config/sweep/config_sweep_ppo.yaml`` to create a wandb sweep.
3. Run ``wandb agent your_entity/your_project/sweep_name`` to start the sweep

## Structure
For everyone who wants to work with the project beyond training and testing within the current possibilities, important files and functions are explained in the following section. 
​
### [data_generator/task.py](src/data_generator/task.py)
1. Class Task
    - Task objects are used to represent tasks and their dependencies in a real world production process
    - Tasks with same job_num can be viewed as belonging to the same job
    - A list of task objects builds an instance, which can be put into the environment
​
### [data_generator/instance_factory.py](src/data_generator/instance_factory.py)
- Can be executed to generate new instances for a scheduling problem as data for training or testing
- Automatically computes deadlines for tasks according to random heuristic. You can change the heuristic used for computation by adapting *DEADLINE_HEURISTIC* constant 

### [data_generator/sp_factory.py](src/data_generator/sp_factory.py)
1. Function *generate_instances*
   - This function will be called when generating instances
   - If you want to modify the task generation beyond what is possible by customizing the data_generation config, you can do it here (e.g. different random functions)

​
### [env_tetris_scheduling.py](src/environments/env_tetris_scheduling.py)
- This environment tracks the current scheduling state of all tasks, machines and tools and adjusts the state according to valid incoming actions 
- Action and state space are initialized, depending on the number of tasks, machines and tools 
​
1. Function *Step*
    - Selected actions (select a job) are processed here
    - Without action masking, actions are checked on validity
    - Returning reward and the current production state as observation
    
2. Function *compute_reward*
    - Returns dense reward for makespan optimization according to [https://arxiv.org/pdf/2010.12367.pdf](https://arxiv.org/pdf/2010.12367.pdf)
    - You can set custom reward strategies (e.g. depending on tardiness or other production scheduling attributes provided by the env)
​
3. Function *state_obs* 
    - For each task, scales runtime, task_index, deadline between 0-1
    - This observation will be returned from the environment to the agent
    - You can set other production scheduling attributes as return or different normalization 
    
​
### [agents/heuristic/heuristic_agent.py](src/agents/heuristic/heuristic_agent.py)
- This file contains following heuristics, which can be used to choose actions based on the current state
  - Random task
  - EDD: earliest due date
  - SPT: shortest processing time first
  - MTR: most tasks remaining
  - LTR: least tasks remaining
- You can add own heuristics as function to this file
1. Class *HeuristicSelectionAgent*
   - *__call__* function selects the next task action according to heuristic passed as string
   - If you added own heuristics as functions to this file, you have to add your_heuristic_function_name: "your_heuristic_string" item to the *task_selections* dict attribute of this class

​
### [agents/intermediate_test.py](src/agents/intermediate_test.py)
- During training, every *n_test_steps* (adjust in train.py *IntermediateTest*)  *_on_step* tests and saves the latest model (save only if better results than current saved model)

​
### [utils/evaluation.py](src/utils/evaluations.py)
- Function *evaluate_test* gets all test results, then calculate and returns evaluated parameters
- You can add or remove metrics here

​
### [utils/logger.py](src/utils/logger.py)
1. Class *Logger*
   - *record* can be called to store a dict with parameters in a logger object
   - calling *dump* tracks the stored parameters and clears the logger object memory buffer. 
   - At the moment only wandb logging is supported. Own logging tools (e.g. Tensorboard) can be implemented und then used by calling them in the *dump* function
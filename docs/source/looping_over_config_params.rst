Looping over Configuration Parameters
=====================================

This tutorial comes in handy, if you need to loop across multiple configs or if you want to change single parameters
without saving new config files.


#. Import the main function from agents.train to your script

#. Load a config or specify an entire config

#. Loop over the parameter you want to change and update your config before starting a training

.. code-block:: python

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

.. note::
   If you want to use the functionality of sweeps in weights and biases, check out the designated tutorial!
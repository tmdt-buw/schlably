"""
PPO implementation with action mask according to the StableBaselines3 implementation.
To reuse trained models, you can make use of the save and load function
"""
import numpy as np
import random
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import pickle
from typing import Tuple, Any, List

from src.utils.logger import Logger

# constants
POLICY_LAYER: List[int] = [256, 256]
POLICY_ACTIVATION: str = 'ReLU'
VALUE_LAYER: List[int] = [256, 256]
VALUE_ACTIVATION: str = 'ReLU'


class RolloutBuffer:
    """
    Handles episode data collection and batch generation

    :param buffer_size: Buffer size
    :param batch_size: Size for batches to be generated

    """
    def __init__(self, buffer_size: int, batch_size: int):

        self.observations = []
        self.probs = []
        self.values = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.advantages = None
        self.returns = None

        if buffer_size % batch_size != 0:
            raise TypeError("rollout_steps has to be a multiple of batch_size")
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        self.reset()

    def generate_batches(self) -> Tuple:
        """
        Generates batches from the stored data

        :return:  batches: Lists with all indices from the rollout_data, shuffled and sampled in lists with batch_size
            e.g. [[0,34,1,768,...(len: batch size)], [], ...(len: len(rollout_data) / batch size)]

        """
        # create random index list and split into arrays with batch size
        indices = np.random.permutation(self.buffer_size)
        num_batches = int(self.buffer_size / self.batch_size)
        batches = indices.reshape((num_batches, self.batch_size))

        return np.array(self.observations), np.array(self.actions), np.array(self.probs), np.array(self.action_masks),\
               batches

    def compute_advantages_and_returns(self, last_value, gamma, gae_lambda) -> None:
        """
        Computes advantage values and returns for all stored episodes. Required to

        :param last_value: Value from the next step to calculate the advantage for the last episode in the buffer
        :param gamma: Discount factor for the advantage calculation
        :param gae_lambda: Smoothing parameter for the advantage calculation

        :return: None

        """
        # advantage: advantage from the actual returned rewards over the baseline value from step t onwards
        last_advantage = 0
        for step in reversed(range(self.buffer_size)):
            # use the predicted reward for the advantage computation of the last step of the buffer
            if step == self.buffer_size - 1:
                # if a step is the last one of the episode (done = 1) -> not_done = 0 => the advantage
                # doesn't contain values outside the own episode
                not_done = 1.0 - self.dones[step]
                next_values = last_value
            else:
                not_done = 1.0 - self.dones[step]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + gamma * next_values * not_done - self.values[step]
            last_advantage = delta + gamma * gae_lambda * not_done * last_advantage
            self.advantages[step] = last_advantage

        # compute returns = discounted rewards, advantages = discounted rewards - values
        # Necessary to update the value network
        self.returns = self.values + self.advantages

    def store_memory(self, observation: np.ndarray, action: int, prob: float, value: float,
                     reward: Any, done: bool, action_mask: np.ndarray) -> None:
        """
        Appends all data from the recent step

        :param observation: Observation at the beginning of the step
        :param action: Index of the selected action
        :param prob: Probability of the selected action (output from the policy_net)
        :param value: Baseline value that the value_net estimated from this step onwards according to the
        :param observation: Output from the value_net
        :param reward: Reward the env returned in this step
        :param done: True if the episode ended in this step
        :param action_mask: One hot vector with ones for all possible actions

        :return: None

        """
        self.observations.append(observation)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def reset(self) -> None:
        """
        Resets all buffer lists
        :return: None
        """
        self.observations = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.action_masks = []
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)


class PolicyNetwork(nn.Module):
    """
    Policy Network for the agent

    :param input_dims: Observation size to determine input dimension
    :param n_actions: Number of action to determine output size
    :param learning_rate: Learning rate for the network
    :param fc1_dims: Size hidden layer 1
    :param fc2_dims: Size hidden layer 2

    """
    def __init__(self, input_dim: int, n_actions: int, learning_rate: float, hidden_layers: List[int], activation: str):

        super(PolicyNetwork, self).__init__()

        net_structure = []
        # get activation class according to string
        activation = getattr(nn, activation)()

        # create first hidden layer in accordance with the input dim and the first hidden dim
        net_structure.extend([nn.Linear(input_dim, hidden_layers[0]), activation])

        # create the other hidden layers
        for i, layer_dim in enumerate(hidden_layers):
            if not i + 1 == len(hidden_layers):
                net_structure.extend([nn.Linear(layer_dim, hidden_layers[i + 1]), activation])
            else:
                # create output layer
                net_structure.extend([nn.Linear(layer_dim, n_actions), nn.Softmax(dim=-1)])

        self.policy_net = nn.Sequential(*net_structure)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation, action_mask):
        """forward through the actor network"""
        observation.to(self.device)
        logits = self.policy_net(observation)

        # mask probabilities if action_mask is not None (for env.reset)
        if action_mask is not None:
            action_mask.to(self.device)
            logits = T.where(action_mask, logits, T.tensor(-1e+8).to(self.device))

        dist = Categorical(logits=logits)
        
        return dist


class ValueNetwork(nn.Module):
    """
    Value Network for the agent

    :param input_dims: Observation size to determine input dimension
    :param learning_rate: Learning rate for the network
    :param fc1_dims: Size hidden layer 1
    :param fc2_dims: Size hidden layer 2

    """
    def __init__(self, input_dim: int, learning_rate: float, hidden_layers: List[int], activation: str):
        super(ValueNetwork, self).__init__()

        net_structure = []
        # get activation class according to string
        activation = getattr(nn, activation)()

        # create first hidden layer in accordance with the input dim and the first hidden dim
        net_structure.extend([nn.Linear(*input_dim, hidden_layers[0]), activation])

        # create the other hidden layers
        for i, layer_dim in enumerate(hidden_layers):
            if not i + 1 == len(hidden_layers):
                net_structure.extend([nn.Linear(layer_dim, hidden_layers[i + 1]), activation])
            else:
                # create output layer
                net_structure.append(nn.Linear(layer_dim, 1))

        self.value_net = nn.Sequential(*net_structure)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        """forward through the value network"""
        value = self.value_net(observation)

        return value


class MaskedPPO:
    def __init__(self, env, config: dict, logger: Logger = None):
        """
        | gamma: Discount factor for the advantage calculation
        | learning_rate: Learning rate for both, policy_net and value_net
        | gae_lambda: Smoothing parameter for the advantage calculation
        | clip_range: Limitation for the ratio between old and new policy
        | batch_size: Size of batches which were sampled from the buffer and fed into the nets during training
        | n_epochs: Number of repetitions for each training iteration
        | rollout_steps: Step interval within the update is performed. Has to be a multiple of batch_size
        """

        self.env = env
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_range = config.get('clip_range', 0.2)
        self.n_epochs = config.get('n_epochs', 0.5)
        self.rollout_steps = config.get('rollout_steps', 2048)
        self.ent_coef = config.get('ent_coef', 0.0)
        self.num_timesteps = 0
        self.n_updates = 0
        self.learning_rate = config.get('learning_rate', 0.002)
        self.batch_size = config.get('batch_size', 256)

        self.logger = logger if logger else Logger(config=config)
        self.seed = config.get('seed', None)

        # torch seed setting
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            T.manual_seed(self.seed)
            # self.env.action_space.seed(seed)
            self.env.seed(self.seed)

        # create networks and buffer
        self.policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, self.learning_rate,
                                        config.get('policy_layer', POLICY_LAYER),
                                        config.get('policy_activation', POLICY_ACTIVATION))
        self.value_net = ValueNetwork(env.observation_space.shape, self.learning_rate,
                                      config.get('value_layer', VALUE_LAYER),
                                      config.get('value_activation', VALUE_ACTIVATION))
        self.rollout_buffer = RolloutBuffer(self.rollout_steps, self.batch_size)

    @classmethod
    def load(cls, file: str, config: dict, logger: Logger = None):
        """
        Creates a PPO object according to the parameters saved in file.pkl

        :param file: Path and filname (without .pkl) of your saved model pickle file
        :param config: Dictionary with parameters to specify PPO attributes
        :param logger: Logger

        :return: MaskedPPO object

        """
        with open(f"{file}.pkl", "rb") as handle:
            data = pickle.load(handle)

        env = data["params"]["env"]

        # create PPO object, commit necessary parameters. Update remaining parameters
        model = cls(env=env, config=config, logger=logger)
        model.__dict__.update(data["params"])

        # set weights from policy and value
        model.policy_net.load_state_dict(data["policy_params"])
        model.value_net.load_state_dict(data["value_params"])

        return model

    def save(self, file: str) -> None:
        """
        Save model as pickle file

        :param file: Path under which the file will be saved

        :return: None

        """
        params_dict = self.__dict__.copy()
        del params_dict['logger']
        data = {
            "params": params_dict,
            "policy_params": self.policy_net.state_dict(),
            "value_params": self.value_net.state_dict()
        }

        with open(f"{file}.pkl", "wb") as handle:
            pickle.dump(data, handle)

    def forward(self, observation: np.ndarray, action_mask: np.ndarray) -> Tuple:
        """
        Predicts an action according to the current policy and based on the action_mask and observation
        and the value for the next state

        :param observation: Current observation of teh environment
        :param action_mask: One hot vector with ones for all possible actions

        :return: Predicted action, probability for this action, and predicted value for the next state

        """

        observation = T.tensor(observation, dtype=T.float).to(self.policy_net.device)
        if action_mask is not None:
            action_mask = T.tensor(action_mask, dtype=T.bool).to(self.policy_net.device)

        dist = self.policy_net(observation, action_mask)
        value = self.value_net(observation)
        action = dist.sample()

        prob = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, prob, value

    def predict(self, observation: np.ndarray, action_mask: np.ndarray,
                deterministic: bool = True, state=None) -> Tuple:
        """
        Action prediction for testing

        :param observation: Current observation of teh environment
        :param action_mask: One hot vector with ones for all possible actions
        :param deterministic: Set True, to force a deterministic prediction
        :param state: The last states (used in rnn policies)

        :return: Predicted action and next state (used in rnn policies)

        """
        observation = T.tensor(np.array(observation), dtype=T.float).to(self.policy_net.device)
        action_mask = T.tensor(action_mask, dtype=T.bool).to(self.policy_net.device)

        with T.no_grad():
            dist = self.policy_net(observation, action_mask)
            if deterministic:
                action = T.argmax(dist.probs)
            else:
                # choose random action according to the predicted probs
                action = dist.sample()
            action = T.squeeze(action).item()

        return action, state

    def train(self) -> None:
        """
        Trains policy and value

        :return: None

        """
        # switch to train mode
        self.policy_net.train(True)
        self.value_net.train(True)

        policy_losses, value_losses, entropy_losses, total_losses = [], [], [], []

        for _ in range(self.n_epochs):

            # get data from buffer and random batches(index lists) to iterate over
            # e.g. obs[batch] returns the observations for all indices in batch
            obs_arr, action_arr, old_prob_arr, action_mask_arr, batches = self.rollout_buffer.generate_batches()

            # get advantage and return values from buffer
            advantages = T.tensor(self.rollout_buffer.advantages).to(self.policy_net.device)
            returns = T.tensor(self.rollout_buffer.returns).to(self.value_net.device)

            # normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            for batch in batches:
                observations = T.tensor(obs_arr[batch], dtype=T.float).to(self.policy_net.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.policy_net.device)
                actions = T.tensor(action_arr[batch]).to(self.policy_net.device)
                action_masks = T.tensor(action_mask_arr[batch], dtype=T.bool).to(self.policy_net.device)

                dist = self.policy_net(observations, action_masks)
                values = self.value_net(observations)
                values = T.squeeze(values)

                # ratio between old and new policy (probs of selected actions)
                # Should be one at the first batch of every train iteration
                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()

                # policy clip
                policy_loss_1 = prob_ratio * advantages[batch]
                policy_loss_2 = T.clamp(prob_ratio, 1-self.clip_range, 1+self.clip_range) * advantages[batch]
                # we want to maximize the reward, but running gradient descent -> negate the loss here
                policy_loss = -T.min(policy_loss_1, policy_loss_2).mean()

                value_loss = (returns[batch]-values)**2
                value_loss = value_loss.mean()

                # entropy loss
                entropy_loss = -T.mean(dist.entropy())
                entropy_losses.append(entropy_loss.item())

                total_loss = policy_loss + 0.5*value_loss + self.ent_coef*entropy_loss
                self.policy_net.optimizer.zero_grad()
                self.value_net.optimizer.zero_grad()
                total_loss.backward()
                self.policy_net.optimizer.step()
                self.value_net.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                total_losses.append(total_loss.item())

        self.n_updates += self.n_epochs

        # logs
        # compute explained variance
        explained_var = explained_variance(np.asarray(self.rollout_buffer.values), self.rollout_buffer.returns)

        self.logger.record(
            {
                'agent_training/n_updates': self.n_updates,
                'agent_training/loss': np.mean(total_losses),
                'agent_training/policy_gradient_loss': np.mean(policy_losses),
                'agent_training/value_loss': np.mean(value_losses),
                'agent_training/entropy_loss': np.mean(entropy_losses),
                'agent_training/explained_variance': explained_var
            }
        )
        self.logger.dump()

    def learn(self, total_instances: int, total_timesteps: int, intermediate_test=None) -> None:
        """
        Learn over n environment instances or n timesteps. Break depending on which condition is met first
        One learning iteration consists of collecting rollouts and training the networks

        :param total_instances: Instance limit
        :param total_timesteps: Timestep limit
        :param intermediate_test: (IntermediateTest) intermediate test object. Must be created before.

        """
        instances = 0

        # iterate over n episodes = the agents has n episodes to interact with the environment
        for _ in range(total_instances):
            obs = self.env.reset()
            info = {'mask': None}
            done = False
            instances += 1

            # run agent on env until done
            while not done:
                action, prob, val = self.forward(obs, action_mask=info['mask'])
                new_obs, reward, done, info = self.env.step(action)
                self.num_timesteps += 1
                self.rollout_buffer.store_memory(obs, action, prob, val, reward, done, info['mask'])

                # call intermediate_test on_step
                if intermediate_test:
                    intermediate_test.on_step(self.num_timesteps, instances, self)

                # break learn if total_timesteps are reached
                if self.num_timesteps >= total_timesteps:
                    print("total_timesteps reached")
                    self.logger.record(
                        {
                            'results_on_train_dataset/instances': instances,
                            'results_on_train_dataset/num_timesteps': self.num_timesteps
                        }
                    )
                    self.logger.dump()

                    return None

                # update every n rollout_steps
                if self.num_timesteps % self.rollout_steps == 0:
                    # predict the next reward, needed for the advantage computation of the last collected step
                    with T.no_grad():
                        _, _, val = self.forward(new_obs, info['mask'])
                    self.rollout_buffer.compute_advantages_and_returns(val, self.gamma, self.gae_lambda)

                    # train networks
                    self.train()
                    # switch back to normal mode
                    self.policy_net.train(False)
                    self.value_net.train(False)

                    # reset buffer to continue collecting rollouts
                    self.rollout_buffer.reset()

                obs = new_obs

            if instances % len(self.env.data) == len(self.env.data) - 1:
                mean_training_reward = np.mean(self.env.episodes_rewards)
                mean_training_makespan = np.mean(self.env.episodes_makespans)
                if len(self.env.episodes_tardinesses) == 0:
                    mean_training_tardiness = 0
                else:
                    mean_training_tardiness = np.mean(self.env.episodes_tardinesses)
                self.logger.record(
                    {
                        'results_on_train_dataset/mean_reward': mean_training_reward,
                        'results_on_train_dataset/mean_makespan': mean_training_makespan,
                        'results_on_train_dataset/mean_tardiness': mean_training_tardiness
                    }
                )
                self.logger.dump()

        print("TRAINING DONE")
        self.logger.record(
            {
                'results_on_train_dataset/instances': instances,
                'results_on_train_dataset/num_timesteps': self.num_timesteps
            }
        )
        self.logger.dump()


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    From Stable-Baseline
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value

    :return: explained variance of ypred and y

    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y


if __name__ == "__main__":

    policy_net = PolicyNetwork(4, 10, 0.003)

    for name, para in policy_net.named_parameters():
        print('{}: {}'.format(name, para.shape))

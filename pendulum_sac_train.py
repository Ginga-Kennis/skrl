import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer


class Actor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False, clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        self.linear_layer_1 = nn.Linear(self.num_observations, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.action_layer = nn.Linear(300, self.num_actions)

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(inputs["states"]))
        x = F.relu(self.linear_layer_2(x))
        return torch.tanh(self.action_layer(x)), self.log_std_parameter, {}
    

class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        self.linear_layer_1 = nn.Linear(self.num_observations + self.num_actions, 400)
        self.linear_layer_2 = nn.Linear(400, 300)
        self.linear_layer_3 = nn.Linear(300, 1)

    def compute(self, inputs, role):
        x = F.relu(self.linear_layer_1(torch.cat([inputs["states"], inputs["taken_actions"]], dim = 1)))
        x = F.relu(self.linear_layer_2(x))
        return self.linear_layer_3(x), {}
    

# ENVIRONMENT
env = gym.make("Pendulum-v1")
env = wrap_env(env)

# CUDA DEVICE
device = env.device

# REPLAY BUFFER
memory = RandomMemory(memory_size=20000, num_envs=env.num_envs, device=device, replacement=False)


# AGENT
models = {}
models["policy"] = Actor(env.observation_space, env.action_space, device, clip_actions=True)
models["critic_1"] = Critic(env.observation_space, env.action_space, device)
models["critic_2"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_1"] = Critic(env.observation_space, env.action_space, device)
models["target_critic_2"] = Critic(env.observation_space, env.action_space, device)

for model in models.values():
    model.init_parameters(method_name="normal_", mean=0.0, std=0.1)

cfg = SAC_DEFAULT_CONFIG.copy()
cfg["batch_size"] = 64
cfg["discount_factor"] = 0.99
cfg["polyak"] = 0.005
cfg["actor_learning_rate"] = 1e-3
cfg["critic_learning_rate"] = 1e-3
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 1000
cfg["learn_entropy"] = True
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["base_directory"] = "runs/"
cfg["experiment"]["experiment_name"] = "exp1"
cfg["experiment"]["write_interval"] = 100
cfg["experiment"]["checkpoint_interval"] = 1000

agent = SAC(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 30000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=[agent])

# start training
trainer.train()

# start evaluation
trainer.eval()
    
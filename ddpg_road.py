import gym
import highway_env
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make("parking-v0")

# Create 4 artificial transitions per real transition
n_sampled_goal = 4



# Create the action noise object that will be used for exploration
n_actions = env.action_space.shape
noise_std = 0.2
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))


model = DDPG(
    "MultiInputPolicy",
    env,
    replay_buffer_class=HerReplayBuffer,
    replay_buffer_kwargs=dict(
      n_sampled_goal=n_sampled_goal,
      goal_selection_strategy="future",
      # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
      # we have to manually specify the max number of steps per episode
      max_episode_length=100,
      online_sampling=True,
    ),
    verbose=1,
    action_noise=action_noise,
    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=256,
    policy_kwargs=dict(net_arch=[256, 256, 256]),
)
# Train for 2e5 steps
model.learn(int(2e5))
# Save the trained agent
model.save('her_ddpg_highway')
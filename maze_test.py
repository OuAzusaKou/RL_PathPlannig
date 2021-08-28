import matplotlib
matplotlib.use('TkAgg')  # avoid non-GUI warning for matplotlib
import gym
import highway_env
import numpy as np

from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from IPython.display import display, HTML

from gym_maze.envs import MazeEnv
from gym_maze.envs.generators import SimpleMazeGenerator, RandomMazeGenerator, RandomBlockMazeGenerator, \
                                     UMazeGenerator, TMazeGenerator, WaterMazeGenerator
from gym_maze.envs.Astar_solver import AstarSolver


maze = RandomBlockMazeGenerator(maze_size=30, obstacle_ratio=0.2)
env = MazeEnv(maze)
print(env.action_space)
n_actions = env.action_space.shape[0]
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


import gym
import highway_env
import numpy as np
import torch
from maze_new_env import Maze_New_Env
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
torch.backends.cudnn.enabled = False
env = Maze_New_Env(grid_size=11)

print(env)
# The noise objects for DDPG
n_actions = env.action_space.shape
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = SAC("MlpPolicy", env, verbose=1,    buffer_size=int(1e6),
    learning_rate=1e-3,
    gamma=0.95,
    batch_size=256,tensorboard_log="./SAC_tensorboard/")
model.learn(total_timesteps=int(1e5),tb_log_name="first_run")
model.save("sac_pendulum")
env = model.get_env()

del model # remove to demonstrate saving and loading

model = SAC.load("sac_pendulum")

obs = env.reset()


# Evaluate the agent
episode_reward = 0
for _ in range(100):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    episode_reward += reward
    if done :
        print("Reward:", episode_reward)
        episode_reward = 0.0
        obs = env.reset()
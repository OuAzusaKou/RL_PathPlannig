import gym
import highway_env
import numpy as np
import torch
from maze_new_env import Maze_New_Env
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy import interpolate
import pylab as pl

from scipy import interpolate
torch.backends.cudnn.enabled = False
env = Maze_New_Env(grid_size=11)
def data_correct(x):
    X1=x.copy()
    X1[0:5]=np.linspace(X1[0],X1[5],5)
    X1[-3:] = np.linspace(X1[-3],X1[-1],3)
    return X1

# Load saved model
# Because it needs access to `env.compute_reward()`
# HER must be loaded with the env
model = DDPG.load("ddpg_pendulum")
model1 = SAC.load("sac_pendulum")
obs = env.reset()
#ddpg_parameter=model.get_parameters()
#sac_parameter=model1.get_parameters()
#print(ddpg_parameter)
#print(sac_parameter)

# Evaluate the agent
episode_reward = 0
x=[]
y=[]
x_=[]
y_=[]
x1=[]
y1=[]
coord=np.where(env.observation==1)
#print(coord)
amount=coord[0].size
#print(amount)
x__,y__=coord
for i in range(amount):
    x_.append(x__[i])
    y_.append(y__[i])
#print(x_)
#print(y_)

for _ in range(100000):
    x.append(env.agent_pos[0])
    y.append(env.agent_pos[1])
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #print(obs)
    #env.render()



    episode_reward += reward
    if done:
        x.append(env.agent_pos[0])
        y.append(env.agent_pos[1])
        print("Reward:", episode_reward)
        episode_reward = 0.0
        coord = np.where(env.observation == 1)
        # print(coord)
        amount = coord[0].size
        # print(amount)
        x__, y__ = coord
        for i in range(amount):
            x_.append(x__[i])
            y_.append(y__[i])
        obs = env.reset()

        #plt.scatter(x, y, color = "green", s = 1)
        #plt.plot(x,y,color = "green",linestyle=':')
        #plt.scatter(x_,y_,color = "blue", s =200)
        '''
        z1 = np.polyfit(x, y, 4)
        p1 = np.poly1d(z1)
        yvals = p1(x)
        # plot2 = plt.plot(x, yvals, 'b', label='polyfit1values')
        # plt.scatter(x_, y_, color="blue", s=200)
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        #axes.plot(x1, y2vals, color='blue', linewidth=2, linestyle="-")
        axes.plot(x, yvals, color='red', linewidth=2, linestyle="-")
        axes.scatter(x_, y_, color="blue", s=200)

        fig.savefig('./test_' + str(time.time()) + '.png')
        fig.show()
        #plt.savefig('./test_' + str(time.time()) + '.png')
        #plt.show()
        '''
        break

obs = env.reset()
for _ in range(100000):
    x1.append(env.agent_pos[0])
    y1.append(env.agent_pos[1])
    action, _ = model1.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    # print(obs)
    # env.render()



    episode_reward += reward
    if done:
        x1.append(env.agent_pos[0])
        y1.append(env.agent_pos[1])
        print("Reward:", episode_reward)
        episode_reward = 0.0
        # print(coord)
        # print(amount)
        obs = env.reset()
        #plt.scatter(x1, y1, color="red", s=1)
        #plt.plot(x1, y1, color="red", linestyle=':')
        x = data_correct(x)
        x1 = data_correct(x1)
        #print(x)
        #print(x1)
        z2 = np.polyfit(x1, y1, 4)
        p2 = np.poly1d(z2)
        y2vals = p2(x1)
        #plot3 = plt.plot(x1, y2vals, 'r', label='polyfit2values')
        z1 = np.polyfit(x, y, 4)
        p1 = np.poly1d(z1)
        yvals = p1(x)
        #plot2 = plt.plot(x, yvals, 'b', label='polyfit1values')
        #plt.scatter(x_, y_, color="blue", s=200)
        fig = plt.figure()
        axes = fig.add_subplot(1, 1, 1)
        axes.plot(x1, y2vals, color='blue', linewidth=2, linestyle="-")
        axes.plot(x, yvals, color='red', linewidth=2, linestyle="-")
        # axes.set_xlim(x.min(),x.max())
        axes.set_ylim(min(y2vals.min(), yvals.min()), max(y2vals.max(), yvals.max()))
        axes.scatter(x_, y_, color="blue", s=200)

        fig.savefig('./test_' + str(time.time()) + '.png')
        fig.show()

        break

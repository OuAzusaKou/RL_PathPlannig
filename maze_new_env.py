import numpy as np
import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env

class Maze_New_Env(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a simple env where the agent must learn to go always left.
    """


    def __init__(self, grid_size):
        super(Maze_New_Env, self).__init__()

        # Size of the 1D-grid
        self.grid_size = grid_size
        # Initialize the agent at the right of the grid
        self.agent_pos = np.array([0,0])
        self.goal = np .array([grid_size-1,grid_size-1])
        self.count=0
        self.observation=np.zeros((grid_size,grid_size))
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions, we have two: left and right
        n_actions = 2
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        # The observation will be the coordinate of the agent
        # this can be described both by Discrete and Box space
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(grid_size*grid_size,), dtype=np.float32)

    def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        self.observation=np.zeros((self.grid_size,self.grid_size))
        # Initialize the agent at the right of the grid
        self.agent_pos = np.array([0,0])
        self.observation[1,3] = 1
        self.observation[3,2] = 1
        self.observation[4,1] = 1
        self.observation[7,7] = 1
        self.observation[9,10] = 1
        self.observation[6,3] = 1
        self.observation[9,5] = 1
        self.observation[0,0] = 2
        self.count=0
        obs = self.observation.flatten()
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        return obs

    def step(self, action):
        self.count+=1
        #print(self.agent_pos)
        #print(self.observation[self.agent_pos[0],self.agent_pos[1]])
        self.observation[int(self.agent_pos[0]),int(self.agent_pos[1])] =0
        action=np.rint(action)
        agent_pos_buf = self.agent_pos + action
        agent_pos_buf = np.clip(agent_pos_buf, 0, self.grid_size-1)
        #print((agent_pos_buf))
        # Account for the boundaries of the grid
        if self.observation[int(agent_pos_buf[0]),int(agent_pos_buf[1])] == 1:
            self.agent_pos = self.agent_pos
        else:
            self.agent_pos = agent_pos_buf

        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size-1)
        #print(self.agent_pos)
        done = bool(np.linalg.norm(self.agent_pos - self.goal) == 0) or self.count>30

        # Null reward everywhere except when reaching the goal (left of the grid)
        reward = -1

        # Optionally we can pass additional info, we are not using that for now
        info = {}

        self.observation[int(self.agent_pos[0]),int(self.agent_pos[1])] = 2

        obs = self.observation.flatten()

        return obs, reward, done, info

    def render(self, mode='console'):
        pass

    def close(self):
        pass

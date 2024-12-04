import gym
from stable_baselines3 import PPO
import numpy as np

class DroneEnv(gym.Env):
    def __init__(self):
        super(DroneEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.state = np.zeros(2)

    def step(self, action):
        self.state += action
        reward = -np.sum(np.abs(self.state))
        done = False
        info = {}
        return self.state, reward, done, info

    def reset(self):
        self.state = np.zeros(2)
        return self.state

def train_ppo_agent(env, num_steps=10000):
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=num_steps)
    return model

def test_agent(model, env, num_episodes=10):
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)
            env.render()

if __name__ == "__main__":
    drone_env = DroneEnv()
    trained_model = train_ppo_agent(drone_env)
    test_agent(trained_model, drone_env)

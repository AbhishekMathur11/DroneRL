import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import numpy as np
import random
from collections import deque
# Define a simple image-based drone environment
class ImageDroneEnv(gym.Env):
    def __init__(self):
        super(ImageDroneEnv, self).__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(4)  # Assuming four discrete actions for simplicity

        # Initialize drone state, image, etc.

    def reset(self):
        # Reset drone state, image, etc.
        return self.get_observation()

    def step(self, action):
        # Perform action and update drone state, image, etc.
        reward = self.calculate_reward()
        done = self.check_done()
        observation = self.get_observation()

        return observation, reward, done, {}

    def get_observation(self):
        # Generate the current observation (image) of the environment
        observation = np.random.randint(0, 256, size=(3, 64, 64), dtype=np.uint8)
        return observation

    def calculate_reward(self):
        # Calculate the reward based on the drone's current state
        return np.random.uniform(-1, 1)

    def check_done(self):
        # Check if the episode is done
        return False


# Define a simple convolutional neural network for image processing
class ImageCNN(nn.Module):
    def __init__(self):
        super(ImageCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 4)  # Output size is the number of actions

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the Deep Q Network (DQN) agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.policy_net = ImageCNN()
        self.target_net = ImageCNN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 0.1

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)  # Random action for exploration
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def train(self, state, action, next_state, reward, done):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = torch.tensor([action])
        next_state = torch.from_numpy(next_state).float().unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float32)
        done = torch.tensor([done], dtype=torch.float32)

        q_values = self.policy_net(state)
        next_q_values = self.target_net(next_state).detach()
        target = reward + (1 - done) * self.gamma * next_q_values.max(1)[0]

        loss = self.criterion(q_values.gather(1, action.unsqueeze(1)), target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Main training loop
env = ImageDroneEnv()
agent = DQNAgent(state_size=(3, 64, 64), action_size=4)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.train(state, action, next_state, reward, done)
        total_reward += reward

        if done:
            agent.update_target_net()
            break

        state = next_state

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

# Save or use the trained agent for testing
# torch.save(agent.policy_net.state_dict(), 'drone_agent.pth')

# Define the Deep Q Network (DQN) agent with experience replay
class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=32):
        self.policy_net = ImageCNN()
        self.target_net = ImageCNN()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, 4)  # Random action for exploration
        with torch.no_grad():
            state = torch.from_numpy(state).float().unsqueeze(0)
            q_values = self.policy_net(state)
            return q_values.argmax().item()

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, next_states, rewards, dones = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.policy_net(states)
        next_q_values = self.target_net(next_states).detach()
        target = rewards + (1 - dones) * self.gamma * next_q_values.max(1)[0]

        loss = self.criterion(q_values.gather(1, actions.unsqueeze(1)), target.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

# ... (rest of the code)

# Main training loop
env = ImageDroneEnv()
agent = DQNAgent(state_size=(3, 64, 64), action_size=4)

num_episodes = 1000
update_target_net_freq = 10  # Update target network every 10 episodes

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    while True:
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.append((state, action, next_state, reward, done))
        agent.train()
        total_reward += reward

        if done:
            break

        state = next_state

    if episode % update_target_net_freq == 0:
        agent.update_target_net()

    agent.decay_epsilon()

    print(f"Episode: {episode + 1}, Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

# Save or use the trained agent for testing
# torch.save(agent.policy_net.state_dict(), 'drone_agent.pth')

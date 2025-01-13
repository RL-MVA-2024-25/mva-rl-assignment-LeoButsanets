import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from copy import deepcopy
from tqdm import tqdm
from evaluate import evaluate_HIV

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

env = TimeLimit(env=HIVPatient(domain_randomization=True), max_episode_steps=200)

config = {'nb_actions': env.action_space.n,
    'learning_rate': 0.001,
    'gamma': 0.99,
    'buffer_size': 100000,
    'epsilon_min': 0.01,
    'epsilon_max': 1.,
    'epsilon_decay_period': 20000,
    'epsilon_delay_decay': 20,
    'batch_size': 500,
    'gradient_steps': 5,
    'update_target_strategy': 'replace', # or 'ema'
    'update_target_freq': 700,
    'update_target_tau': 0.005,
    'criterion': torch.nn.SmoothL1Loss()}



class ReplayBuffer:
    "Replay Buffer from the course on DQN"
    def __init__(self, capacity, device):
        self.capacity = capacity
        self.data = []
        self.index = 0
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)
        
DQN = torch.nn.Sequential(
            torch.nn.Linear(env.observation_space.shape[0], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, env.action_space.n)
)

# ProjectAgent class to define agent logic
class ProjectAgent:
    def __init__(self):
        state_dim = env.observation_space.shape[0]
        n_action = env.action_space.n
        nb_neurons = 256
        num_layers = 5
   
        # Configuration for the agent
        self.model = DQN
        self.device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        self.model = self.model.to(self.device)
 
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        self.buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.memory = ReplayBuffer(self.buffer_size,self.device)

 
        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop

        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005

        
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        self.fine_tuning = config['fine_tuning'] if 'fine_tuning' in config.keys() else False

        self.target_model = deepcopy(self.model).to(self.device)
        self.lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001
        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self.best_model = deepcopy(self.model).to(self.device)


    def act(self, observation, use_random=False):
        with torch.no_grad():  # Disable gradient computation for action selection
            Q = self.model(torch.Tensor(observation).unsqueeze(0).to(self.device))
            return torch.argmax(Q).item()

    def save(self, path):
        path = os.path.join(os.getcwd(), path)
        torch.save(self.best_model.state_dict(), path)

    def load(self,path="protocol_agent.pkl"):
        # get the folder of the folder 
        # Correct implementation
        path = os.path.join(os.getcwd(), path)
        state_dict = torch.load(path, map_location=self.device)  # Load the state dict with map_location
        self.model.load_state_dict(state_dict)  # Load the state dict into the model
        

    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D = self.memory.sample(self.batch_size)
            QYmax = self.target_model(Y).max(1)[0].detach()
            #update = torch.addcmul(R, self.gamma, 1-D, QYmax)
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step() 

    def train(self, env, max_episode):
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        previous_val = 0
        while episode < max_episode:
            # Update epsilon
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon - self.epsilon_step)

            # Select epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample() 
            else:
                action = self.act(state) 

            # Step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            # Train 
            for _ in range(self.nb_gradient_steps):
                self.gradient_step()

            # Update target model periodically
            if self.update_target_strategy == 'replace':
                if step % self.update_target_freq == 0:
                    self.target_model.load_state_dict(self.model.state_dict())

            step += 1
            if done or trunc:
                episode += 1
                val_score = evaluate_HIV(agent=self, nb_episode=1)

                # Print training progress
                print(f"Episode {episode:3d} | "
                      f"Epsilon {epsilon:6.2f} | "
                      f"Batch Size {len(self.memory):5d} | "
                      f"Episode Return {episode_cum_reward:.2e} | "
                      f"Evaluation Score {val_score:.2e}")
                state, _ = env.reset()

                # Save model if evaluation score improves
                if val_score > previous_val:
                    previous_val = val_score
                    self.best_model = deepcopy(self.model).to(self.device)
                if episode % 10 ==0:
                    self.save("protocol_agent.pkl")
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        return episode_return           

if __name__ == "__main__":
    print("Training the agent...")
    agent = ProjectAgent()
    agent.train(env, max_episode=200)
    agent.save("protocol_agent.pkl")
    print("Training complete. Model saved as protocol_agent.pkl")
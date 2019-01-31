import gym
import matplotlib.pyplot as plt
import torch
import copy
import random
import numpy as np
import time
from collections import deque
from torch import nn
from torch import optim
import torch.nn.functional as F

# Create environment
env = gym.make("MountainCar-v0")
# Reset
state = env.reset()
print(state)

model = nn.Sequential(
    nn.Linear(2, 32),
    nn.ReLU(),
    nn.Linear(32, 32),
    nn.ReLU(),
    nn.Linear(32, 3)
)

target_model = copy.deepcopy(model)


def init_weights(layer):
    if type(layer) == nn.Linear:
        nn.init.xavier_normal(layer.weight)
        

model.apply(init_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.train()
target_model.train()
model.to(device)
target_model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.00003)
gamma = 0.99
epsilon = 0.1


def select_action(state):
    if random.random() < epsilon:
        return random.randint(0, 2)
    return model(torch.tensor(state).to(device).float().unsqueeze(0))[0].max(0)[1].view(1, 1).item()


def fit(batch):
        state, action, reward, next_state, done = batch
        state = torch.tensor(state).to(device).float()
        next_state = torch.tensor(next_state).to(device).float()
        reward = torch.tensor(reward).to(device).float()
        action = torch.tensor(action).to(device)

        target_q = torch.zeros(reward.size()[0]).float().to(device)
        with torch.no_grad():
            #Get predicted by target model Q-function
            target_q[done] = target_model(next_state).max(1)[0].detach()
        #Estimate current Q-function
        target_q = reward + target_q * gamma

        #Current approximation
        q = model(state).gather(1, action.unsqueeze(1))

        loss = F.smooth_l1_loss(q, target_q.unsqueeze(1))

        #Clear all gradients of network
        optimizer.zero_grad()
        #Backpropagate loss
        loss.backward()
        #Clip gradient
        for param in model.parameters():
            param.grad.data.clamp_(-1, 1)
        #Update network parameters
        optimizer.step()


class Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, element):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = element
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


memory = Memory(5000)

reward_by_percentage = []

steps = 0
max_step = 80001
t = time.time()
for step in range(1, max_step):
    epsilon = 1 - 0.9 * step / max_step
    action = select_action(state)
    new_state, reward, done, _ = env.step(action)
    steps += 1

    if done:
        memory.push((state, action, reward, state, done))
        state = env.reset()
        done = False
        steps = 0
    else:
        memory.push((state, action, reward + abs(new_state[1]), new_state, done))
        state = new_state
        
    if step > 128:
        fit(list(zip(*memory.sample(128))))

    if step % 2000 == 0:
        target_model = copy.deepcopy(model)
        print("Progress", step // 2000, "of", max_step // 2000, time.time() - t)
        state = env.reset()
        r = 0.
        r2 = 0.
        done = False
        while not done:
            epsilon = 0.
            if (step // 2000) % 2 == 0:
                env.render()
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            r += reward
            r2 += reward + abs(state[1])
            steps += 1
        print("Reward:", r, r2)
        reward_by_percentage.append(r)
        steps = 0
plt.plot(list(range(len(reward_by_percentage))), reward_by_percentage)
plt.show()

import gym
import matplotlib.pyplot as plt
import torch
import copy
import random
import numpy as np
from collections import deque
from torch import nn
from torch import optim
import torch.nn.functional as F

# Create environment
env = gym.make("MsPacman-v0")
# Reset
frame = env.reset()


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x


def to_grayscale(image):
    #Constants from https://en.wikipedia.org/wiki/Grayscale
    return 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]


def preprocesing(image):
    grayscale = to_grayscale(image)
    #Assume that image is 210x160
    return [[grayscale[i * 160 // 84, j * 160 // 84]for j in range(84)] for i in range(84)]


plt.imshow(preprocesing(frame), cmap="gray")

model = nn.Sequential(
    nn.Conv2d(4, 32, 8, 4),
    nn.ReLU(),
    nn.Conv2d(32, 64, 4, 2),
    nn.ReLU(),
    nn.Conv2d(64, 64, 3, 1),
    nn.ReLU(),
    Flatten(),
    nn.Linear(3136, 512),
    nn.ReLU(),
    nn.Linear(512, 5)
)

target_model = copy.deepcopy(model)


def init_weights(layer):
    if type(layer) == nn.Linear or type(layer) == nn.Conv2d:
        nn.init.xavier_normal(layer.weight)
        

model.apply(init_weights)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.train()
target_model.train()
model.to(device)
target_model.to(device)

optimizer = optim.RMSprop(model.parameters(), lr=0.00025, alpha=0.95, eps=0.01)
gamma = 0.99
epsilon = 0.1


def select_action(state):
    if random.random() < epsilon:
        return random.randint(0, 4)
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


memory = Memory(10000)
last_four_frames = deque(maxlen=4)

for i in range(4):
    last_four_frames.append(np.expand_dims(preprocesing(frame), 2))

summary_reward = 0.

actual_state = np.swapaxes(np.concatenate(last_four_frames, axis=2), 2, 0)
for step in range(1, 100001):
    action = select_action(actual_state)
    state, reward, done, _ = env.step(action)
    last_four_frames.append(np.expand_dims(preprocesing(state), 2))

    summary_reward += reward

    if done:
        memory.push((actual_state, action, reward, None, done))
        frame = env.reset()
        for i in range(4):
            last_four_frames.append(np.expand_dims(preprocesing(frame), 2))
        done = False
        actual_state = np.swapaxes(np.concatenate(last_four_frames, axis=2), 2, 0)
    else:
        new_state = np.swapaxes(np.concatenate(last_four_frames, axis=2), 2, 0)
        memory.push((actual_state, action, reward, new_state, done))
        actual_state = new_state

    if step > 32:
        fit(list(zip(*memory.sample(32))))

    if step % 1000 == 0:
        target_model = copy.deepcopy(model)
        print("Progress", step // 1000, "%, average reward", summary_reward / 1000)
        summary_reward = 0

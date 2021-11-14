"""
Trains an Agent With Stochastic Policy Gradient Ascent to Solve the Lunar Lander
Challenge From OpenAI
"""

import csv
import gym
import numpy as np
import os
import pickle
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

# SOME HYPERPARAMETERS
envName = "LunarLander-v2"
batchSize = 10  # every how many episodes do we update parameters?
gamma = 0.99    # discount factor for rewards
resume = True   # resume from previous checkpoint if possible?
render = True   # render out the game on-screen?

# π_θ(a|s) is approximated by a neural network
class PolicyNetwork(nn.Module):
    def __init__(self, inputSize, outputSize, hiddenSize=200, alpha=1e-4):
        super(PolicyNetwork, self).__init__()
        # specify where the network is saved to
        self.savePath = envName + '.pickle'
        
        # retrieve or initialize policy
        self.network = pickle.load(open(self.savePath, 'rb'))\
            if resume and os.path.exists(self.savePath) else nn.Sequential(
                nn.Linear(inputSize, hiddenSize),
                nn.ReLU(),
                nn.Linear(hiddenSize, hiddenSize),
                nn.ReLU(),
                nn.Linear(hiddenSize, outputSize),
                nn.Softmax(dim=-1)
            ).cuda()
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward(self, state):
        return self.network(state)
    
    def save(self):
        pickle.dump(self.network, open(self.savePath, 'wb'))

# take numpy array of rewards and compute discounted reward at each time step
def discountRewards(r: np.ndarray):
    discountedRewards = np.zeros(r.size)
    # G_0 = r_0 + γ*r_1 + (γ**2)*r_2 + (γ**3)*r_3 + ...
    #     = r_0 + γ(r_1 + γ*r_2 + (γ**2)*r_3 + ...)
    #     = r_0 + γ*G_1
    # G_t = r_t + γ*G_{t + 1}
    valueNext = 0
    for t in reversed(range(0, r.size)):
        discountedRewards[t] = r[t] + gamma*valueNext
        valueNext = discountedRewards[t]
    return discountedRewards

env = gym.make(envName)
policy = PolicyNetwork(env.observation_space.shape[0], env.action_space.n)
state = env.reset()
states, actions, rewards = [], [], []
epStates, epActions, epRewards = [], [], []
runningReward = None
sumRewards = 0
numEpisode = 0

while True:
    if render and numEpisode % 10 == 0:
        env.render()
    
    # sample an action from the policy network's output, which is a probability
    # distribution --- in other words, a_t ~ π_θ(a|s_t)
    actionProbs = policy.forward(T.tensor(state).cuda())
    actionDist = Categorical(actionProbs)
    action = actionDist.sample()

    epStates.append(state)
    epActions.append(action.item())
    # take a step and collect information
    state, reward, done, info = env.step(action.item())
    # record reward after calling step() to get reward *due to* action
    epRewards.append(reward)
    sumRewards += reward
    
    if done:
        # an episode's finished
        numEpisode += 1
        # compute the discounted r(t) backwards through time
        discountedRewards = discountRewards(np.array(epRewards))
        # cache all state-action-reward tuples for this episode
        states.extend(epStates)
        actions.extend(epActions)
        rewards.extend(discountedRewards)
        # reset episodic memory
        epStates, epActions, epRewards = [], [], []
        
        if numEpisode % batchSize == 0:
            # consider *all* state-action-reward tuples for this batch of episodes
            batchStates = T.tensor(np.vstack(states), requires_grad=True).cuda()
            batchActions = T.tensor(actions).cuda()
            batchProbs = policy.forward(batchStates)
            batchRewards = T.tensor(rewards, requires_grad=True).cuda()
            
            # multiply the gradient in the direction of the taken actions by their
            # respective rewards (the heart of policy gradient)
            batchDist = Categorical(batchProbs)
            logPs = batchDist.log_prob(batchActions)
            # create a "loss" function, L, to "minimize" by maximizing
            # logPs*discountedRewards
            L = T.sum(-logPs*batchRewards).cuda()
            
            # perform a step of backpropagation
            policy.optimizer.zero_grad()
            L.backward()
            policy.optimizer.step()
            policy.save()
            
            # restart for the next batch of episodes
            states, actions, rewards = [], [], []

        # keep a running average to show how well we're doing
        runningReward = sumRewards\
            if runningReward is None\
            else runningReward*0.99 + sumRewards*0.01
        # episode tracking
        print(
            f"episode {numEpisode:6d} --- total reward: {sumRewards:7.2f} --- running average: {runningReward:7.2f}"
        )
        # logging in csv
        fields = [numEpisode, sumRewards, runningReward]
        with open('iter-data.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
        # reset environment
        sumRewards = 0
        state = env.reset()
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.convolutional import Convolution1D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
import random
import numpy as np
import matplotlib.pyplot as plt
import gym
import copy
from collections import deque

# Game is initialized using open ai gym
# sequential neural net is created with 4 input variables
# two hidden layers of 20 nodes each and an output layer with 2 nodes
# representing the q values of each action (0 or 1 meaning going left or right)
# Uses leaky Relu activation function
env = gym.make('CartPole-v0')
env.reset()
net = Sequential()
net.add(Dense(8,init='uniform',input_shape=(4,)))
net.add(LeakyReLU(alpha=0.5))
net.add(Dense(12, init='uniform',input_shape=(20,)))
net.add(LeakyReLU(alpha=0.5))
net.add(Dense(2, init='uniform'))
net.add(Activation('linear'))
# Optimizer is RMSprop. Learning rate could be changed.
rms = RMSprop(lr=0.004)
net.compile(loss='mse', optimizer=rms)


target_net = copy.deepcopy(net)
epsilon = 1
buffer = 1000
batch_size = 32
gamma = 0.9
maxTimeStep = 0
replay = []
total_timestep = 0
steps = 0
random_episodes = 200
episodes = []
average = []

# Train method trains the neural network.
# Creates a mini batch from the replay memory.
# Using the tuples creates two different arrays of inputs and outputs.
# y array is the output array. x array is the observation
# y values are updated by adding the current reward to the maximum value of next state
def train():
    if (len(replay) >= buffer):
        del replay[0]
        minibatch = random.sample(replay, batch_size)
        X_train = []
        y_train = []
        for memory in minibatch:
            old_state, action, reward, new_state,done = memory
            old_qval = net.predict(old_state.reshape(1,4),batch_size=1)
            # gets the maximum q value of next state
            maxQ = np.max(target_net.predict(new_state.reshape(1,4),batch_size=1))
            y = np.zeros((1, 2))
            y[:] = old_qval[:]
            # if its terminal state update value is only reward
            if (done):
                update = reward
            else:
                update = (reward + (gamma * maxQ))
            y[0][action] = update
            y_train.append(y.reshape(2,))
            X_train.append(old_state.reshape(4,))
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        net.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,verbose=0 )

# Before starting training, game is played 200 times to fill the replay memory
last_100 = deque(maxlen=100)
for i in range(random_episodes):
    observation = env.reset()
    total_reward = 0
    t = 0
    done = False
    while (done == False):
        steps += 1
        t += 1
        action = np.random.randint(0, 2)
        old_observation = observation
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            total_timestep += t
            replay.append((old_observation, action, total_reward, observation, done))
            last_100.append(total_reward)

# This is where game is played and trained
# Game is played until the last 100 agents average a score of 200 or game is played 3300 times.
# Score is determined by the total time steps until losing
episode = 0
max = 0
env.monitor.start("/tmp/gym-results",force = True)
# np.mean(last_100) < 200
while episode < 3300:
    episode += 1
    observation = env.reset()
    total_reward = 0
    done = False
    t =0
    # Each agent plays the game until it loses
    while(done == False):
        steps += 1
        t += 1
        # env.render()
        q_values = net.predict(observation.reshape(1,4), batch_size=1)
        if (random.random() > epsilon):
            action = np.argmax(q_values)
        else:
            action = np.random.randint(0, 2)
        old_observation = observation
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            total_timestep += t
            replay.append((old_observation, action, total_reward, observation,done))
            last_100.append(total_reward)
            train()
            # print("Episode finished after {} timesteps".format(t+1))
            break
        replay.append((old_observation, action, total_reward, observation,done))
        train()
        if steps % 100 == 0:
            target_net = copy.deepcopy(net)

    episodes.append(episode)
    average.append(np.mean(last_100))
    # Epsilon value is decremented slowly throughout training
    if epsilon > 0.1 and len(replay) >= buffer-1:
        epsilon -= (1 / 3000.0)
    if(np.mean(last_100) > max):
        max = np.mean(last_100)
    if episode % 100 == 0:
        ave_num = np.mean(last_100)
        print "current episode: ", episode
        print "epsilon is ", epsilon
        print "Average reward for last 100 episodes: ", ave_num

env.monitor.close()

print "ended with episode ", episode
print "max of avarage 100 was ",max
plt.figure(1)
plt.plot(episodes,average)
plt.xlabel('Episodes')
plt.ylabel('Average timestep of last 100 games')

plt.show()



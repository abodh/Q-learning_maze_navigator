import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import time
from datetime import timedelta
start_time = time.monotonic()

nr = 3                # number of rows in the maze
nc = 3                # number of columns in the maze
total_states = nr*nc  # total number of states
initial_state = 6    # Initial state
goal_state = 2        # goal State
gamma = 0.9           # discount factor
maxQItr = 1000        # maximum number of iteration
E = 0.5              # exploration rate
run = 100             # total number of runs

#defining the maze

maze = np.ones((nr,nc))
# maze[1:4,2]= -1
# maze [0:3,7] = -1
# maze [4,5] = -1

# implementing
state_actions = np.zeros((total_states, total_states))
for i in range(0, total_states):
    state_actions[i, :] = maze.reshape((1, total_states))

for i in range(0, total_states):
    for j in range(0, total_states):
        if j != i - nc and j != i + nc and j != i - 1 and j != i + 1:
            state_actions[i, j] = -1

        if (i + 1) % nc == 0 and j == i + 1 or (j + 1) % nc == 0 and i == j + 1:
            state_actions[i, j] = -1

        if i == goal_state:
            state_actions[i, :] = -1

q = np.zeros(state_actions.shape)
v = np.zeros([total_states,1])
count = np.zeros((1,maxQItr))
avg_steps = np.zeros ((1,maxQItr))
avg_cum_rew = np.zeros ((1,maxQItr))
iteration = np.zeros ((1,maxQItr))
cum_rew = np.zeros ((1,maxQItr))

for r in range(0, run):
    E = 0.5
    for i in range(0, maxQItr):

        # starting from initial position
        current_state = initial_state  # initializes the current state
        reward = 0  # reward for every non-goal action
        power = 0  # index for gamma for cummulative reward
        cum_reward = 0  # to calculate cummulative reward
        steps = 0  # to calculate the steps to go

        # Repeat until the goal state is reached
        while (1):
            # Check whether the episode has completed i.e Goal has been reached
            n_actions = np.where(state_actions[current_state, :] >= 0)
            n_actions = n_actions[random.randint(0, len(n_actions) - 1)]
            # print ('current possible n_actions', n_actions)

            if ((i+1) % 5) == 0:
                E = E / 1.005

            if np.random.rand() <= E:
                # Choose the next state as an act of random action

                ns = n_actions[random.randint(0, len(n_actions) - 1)]
                # print ('random next state selection',ns)
            else:
                # find the maximum Q-value for next state
                c = 0
                b = np.amax(q[current_state, :])
                for j in range(0, len(n_actions)):
                    if q[current_state, n_actions[j]] == b: #room for improvement
                        c = c + 1

                if c == 1:
                    for j in range(0, len(n_actions)):
                        if q[current_state, n_actions[j]] == b:
                            ns = n_actions[j]
                            # print ('max q so next state selection',ns)
                elif c != 1:
                    temp_n_actions = np.where(q[current_state, :] == b)
                    temp_n_actions = temp_n_actions[random.randint(0, len(temp_n_actions) - 1)]
                    # print ('multiple max q so temp actions',temp_n_actions)
                    if len(temp_n_actions) > 4:
                        ns = n_actions[random.randint(0, len(n_actions) - 1)]
                        # print ('impossible temp so random next state',ns)
                    else:
                        ns = temp_n_actions[random.randint(0, len(temp_n_actions) - 1)]
                        # print ('random ns from multiple max q',ns)

            n_actions = np.where(state_actions[ns, :] >= 0)
            n_actions = n_actions[random.randint(0, len(n_actions) - 1)]
            # print ('possible n_actions from next state', n_actions)

            # print (q)
            max_q = 0
            for j in range(0, len(n_actions)):
                # print (q[ns, n_actions[j]])
                max_q = max(max_q, q[ns, n_actions[j]])

            # print ('max q', max_q)

            # Update q-values as per Bellman's equation

            if (ns == goal_state):
                reward = 100

            q[current_state, ns] = reward + (gamma * max_q)
            # print ('q value',q[current_state, ns])
            cum_reward = cum_reward + ((gamma ** (power)) * reward)
            cum_rew[0, i] = cum_reward
            power = power + 1

            for x in range(0, total_states):
                v[x, 0] = max(q[x, :])

            steps = steps + 1

            # Set current state as next state
            current_state = ns
            if (current_state == goal_state):
                count[0, i] = steps
                break

    avg_steps = avg_steps + count
    avg_cum_rew = avg_cum_rew + cum_rew

c = 0
V = np.zeros((nr, nc))
for i in range(0, nr):
    for j in range(0, nc):
        V[i, j] = v[c]
        c = c + 1

avg_steps = (avg_steps) / run
avg_cum_rew = (avg_cum_rew) / run

print (V)

x = avg_steps[0,:]
plt.plot(x)
plt.grid()
plt.ylabel("Steps to go")
plt.xlabel("Number of iteration")
plt.title("Steps to go vs number of iteration ")
# plt.xlim([0, 1000])

plt.yticks(np.arange(int(min(x)), int(max(x))+1, 1))
plt.show()

y = avg_cum_rew[0,:]
plt.plot(y)
plt.grid()
plt.ylabel("cumulative reward")
plt.xlabel("Number of iteration")
plt.title("cumulative reward vs number of iteration ")
# plt.xlim([0, 1000])
plt.yticks(np.arange(int(min(y)), int(max(y))+1, 1))
plt.show()

# Writing data to csv
# dfs=pd.DataFrame(state_actions)
# np.savetxt('dfs.csv', dfs, delimiter=',', fmt='%s')
#
# dfv = pd.DataFrame(V)
# np.savetxt('dfv.csv', dfv, delimiter=',', fmt='%s')

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
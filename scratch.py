import numpy as np
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import tabulate

class TowersOfHanoi:
    def __init__(self, state):
        self.state = state              # "State" is a tuple of length N, where N is the number of discs, and the elements are peg indices in [0,1,2]
        self.discs = len(self.state)

    def discs_on_peg(self, peg):
        return filter(lambda disc: self.state[disc] == peg, range(self.discs))

    def move_allowed(self, move):
        discs_from = self.discs_on_peg(move[0])
        discs_to = self.discs_on_peg(move[1])
        if discs_from:
            return (min(discs_to) > min(discs_from)) if discs_to else True
        else:
            return False

    def get_moved_state(self, move):
        if self.move_allowed(move):
            disc_to_move = min(self.discs_on_peg(move[0]))
        moved_state = list(self.state)
        moved_state[disc_to_move] = move[1]
        return tuple(moved_state)

def generate_reward_matrix(N):      # N is the number of discs
    states = list(itertools.product(range(3), repeat=N))
    moves = list(itertools.permutations(range(3), 2))
    R = pd.DataFrame(index=states, columns=states, data=-np.inf)
    for state in states:
        tower = TowersOfHanoi(state=state)
        for move in moves:
            if tower.move_allowed(move):
                next_state = tower.get_moved_state(move)
                R[state][next_state] = 0
    final_state = tuple([2]*N)          # Define final state as all discs being on the last peg
    R[final_state] += 100               # Add a reward for all moves leading to the final state
    return R

def learn_Q(R, gamma=0.8, alpha=1.0, N_episodes=1000):
    Q = np.zeros(R.shape)
    states=range(R.shape[0])
    for n in range(N_episodes):
        Q_previous = Q
        state = np.random.choice(states)                # Randomly select initial state
        next_states = np.where(R[state,:] >= 0)[0]      # Generate a list of possible next states
        next_state = np.random.choice(next_states)      # Randomly select next state from the list of possible next states
        V = np.max(Q[next_state,:])                     # Maximum Q-value of the states accessible from the next state
        Q[state, next_state] = (1-alpha)*Q[state, next_state] + alpha*(R[state, next_state] + gamma*V)      # Update Q-values
    if np.max(Q) > 0:
        Q /= np.max(Q)      # Normalize Q to its maximum value
    return Q

R = generate_reward_matrix(2)
# print(tabulate.tabulate(R, headers='keys'))

Q=learn_Q(R.values)

Q_df = pd.DataFrame(index=R.index, columns=R.columns, data=Q)
print(tabulate.tabulate(Q_df, headers='keys', floatfmt=".3f"))

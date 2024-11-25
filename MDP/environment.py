'''
Implementation of a basic RL environment.

Rewards are all normal.
Transitions are multinomial.

author: iosband@stanford.edu
'''

import numpy as np

#-------------------------------------------------------------------------------


class Environment(object):
    '''General RL environment'''

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        '''
        Moves one step in the environment.

        Args:
            action

        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        return 0, 0, 0


#-------------------------------------------------------------------------------


class TabularMDP(Environment):
    '''
    Tabular MDP

    R - dict by (s,a) - each R[s,a] = (meanReward, sdReward)
    P - dict by (s,a) - each P[s,a] = transition vector size S
    '''

    def __init__(self, nState, nAction, epLen):
        '''
        Initialize a tabular episodic MDP

        Args:
            nState  - int - number of states
            nAction - int - number of actions
            epLen   - int - episode length

        Returns:
            Environment object
        '''

        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen

        self.timestep = 0
        self.state = 0

        # Now initialize R and P
        self.R = {}
        self.P = {}
        for state in range(nState):
            for action in range(nAction):
                self.R[state, action] = (1, 1)
                self.P[state, action] = np.ones(nState) / nState


    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = 0

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action

        Returns:
        reward - double - reward
        newState - int - new state
        pContinue - 0/1 - flag for end of the episode
        '''
        if self.R[self.state, action][1] < 1e-9:
            # Hack for no noise
            reward = self.R[self.state, action][0]
        else:
            reward = np.random.normal(loc=self.R[self.state, action][0],
                                      scale=self.R[self.state, action][1])
        newState = np.random.choice(self.nState, p=self.P[self.state, action])

        # Update the environment
        self.state = newState
        self.timestep += 1

        if self.timestep == self.epLen:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1

        return reward, newState, pContinue

    def compute_qVals(self):
        '''
        Compute the Q values for the environment

        Args:
            NULL - works on the TabularMDP

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    qVals[s, j][a] = self.R[s, a][0] + np.dot(self.P[s, a], qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])
        return qVals, qMax


#-------------------------------------------------------------------------------
# Benchmark environments
def make_map(epLen=100, n=10):
    #0: up y+1
    #1: down y-1
    #2: left x-1
    #3: right x+1
    f = lambda x,y: y*n+x
    nAction = 4
    R_true = {}
    P_true = {}
    nState = n ** 2
    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[f(n-1,n),3] = 0
    R_true[f(n,n-1),0] = 0

    # Transitions
    for x in range(n):
        for y in range(n):
            if y+1 < n:
                P_true[f(x,y),0][f(x,y+1)] = 1
            else:
                P_true[f(x,y),0][f(x,y)] = 1
            if y-1 >= 0:
                P_true[f(x,y),1][f(x,y-1)] = 1
            else:
                P_true[f(x,y),1][f(x,y)] = 1
            if x+1 < n:
                P_true[f(x,y),3][f(x+1,y)] = 1
            else:
                P_true[f(x,y),3][f(x,y)] = 1
            if x-1 >= 0:
                P_true[f(x,y),2][f(x-1,y)] = 1
            else:
                P_true[f(x,y),2][f(x,y)] = 1

    Map = TabularMDP(nState, nAction, epLen)
    Map.R = R_true
    Map.P = P_true
    Map.reset()
    return Map


def make_FourRooms(epLen=100, n=11):
    #0: up y+1
    #1: down y-1
    #2: left x-1
    #3: right x+1
    f = lambda x,y: y*n+x
    nAction = 4
    R_true = {}
    P_true = {}
    nState = n ** 2
    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[f(n-1,n),3] = 0
    R_true[f(n,n-1),0] = 0

    # Transitions
    for x in range(n):
        for y in range(n):
            if y+1 < n:
                P_true[f(x,y),0][f(x,y+1)] = 1
            else:
                P_true[f(x,y),0][f(x,y)] = 1
            if y-1 >= 0:
                P_true[f(x,y),1][f(x,y-1)] = 1
            else:
                P_true[f(x,y),1][f(x,y)] = 1
            if x+1 < n:
                P_true[f(x,y),3][f(x+1,y)] = 1
            else:
                P_true[f(x,y),3][f(x,y)] = 1
            if x-1 >= 0:
                P_true[f(x,y),2][f(x-1,y)] = 1
            else:
                P_true[f(x,y),2][f(x,y)] = 1
    wall1 = [0,1,3,4,6,7,9,10]
    for y in wall1:
        P_true[f(4,y),3][f(5,y)] = 0
        P_true[f(4,y),3][f(4,y)] = 1
        P_true[f(6, y), 2][f(5, y)] = 0
        P_true[f(6, y), 2][f(6, y)] = 1
    wall2 = [0, 1, 3, 4, 6, 7, 9, 10]
    for x in wall2:
        P_true[f(x,4),0][f(x,5)] = 0
        P_true[f(x,4),0][f(x,1)] = 1
        P_true[f(x, 6), 1][f(x, 5)] = 0
        P_true[f(x, 6), 1][f(x, 6)] = 1
    P_true[f(5, 2), 0][f(5, 3)] = 0
    P_true[f(5, 2), 0][f(5, 2)] = 1
    P_true[f(5, 2), 1][f(5, 1)] = 0
    P_true[f(5, 2), 1][f(5, 2)] = 1

    P_true[f(5, 8), 0][f(5, 9)] = 0
    P_true[f(5, 8), 0][f(5, 8)] = 1
    P_true[f(5, 8), 1][f(5, 7)] = 0
    P_true[f(5, 8), 1][f(5, 8)] = 1

    P_true[f(2, 5), 2][f(1, 5)] = 0
    P_true[f(2, 5), 2][f(2, 5)] = 1
    P_true[f(2, 5), 3][f(3, 5)] = 0
    P_true[f(2, 5), 3][f(2, 5)] = 1

    P_true[f(8, 5), 2][f(7, 5)] = 0
    P_true[f(8, 5), 2][f(8, 5)] = 1
    P_true[f(8, 5), 3][f(9, 5)] = 0
    P_true[f(8, 5), 3][f(8, 5)] = 1

    Map = TabularMDP(nState, nAction, epLen)
    Map.R = R_true
    Map.P = P_true
    Map.reset()
    return Map
def make_riverSwim(epLen=20, nState=6):
    '''
    Makes the benchmark RiverSwim MDP.

    Args:
        NULL - works for default implementation

    Returns:
        riverSwim - Tabular MDP environment
    '''
    nAction = 2
    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (5. / 1000, 0)
    R_true[nState - 1, 1] = (1, 0)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

    for s in range(1, nState - 1):
        P_true[s, 1][min(nState - 1, s + 1)] = 0.35
        P_true[s, 1][s] = 0.6
        P_true[s, 1][max(0, s-1)] = 0.05

    P_true[0, 1][0] = 0.4
    P_true[0, 1][1] = 0.6
    P_true[nState - 1, 1][nState - 1] = 0.6
    P_true[nState - 1, 1][nState - 2] = 0.4

    riverSwim = TabularMDP(nState, nAction, epLen)
    riverSwim.R = R_true
    riverSwim.P = P_true
    riverSwim.reset()

    return riverSwim

def make_riverSwim_origin(epLen=20, nState=6):
    '''
    Makes the benchmark RiverSwim MDP.

    Args:
        NULL - works for default implementation

    Returns:
        riverSwim - Tabular MDP environment
    '''
    nAction = 2
    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (5. , 0)
    R_true[nState - 1, 1] = (10000, 0)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

    for s in range(1, nState - 1):
        P_true[s, 1][min(nState - 1, s + 1)] = 0.3
        P_true[s, 1][s] = 0.6
        P_true[s, 1][max(0, s-1)] = 0.1

    P_true[0, 1][0] = 0.3
    P_true[0, 1][1] = 0.7
    P_true[nState - 1, 1][nState - 1] = 0.7
    P_true[nState - 1, 1][nState - 2] = 0.3

    riverSwim = TabularMDP(nState, nAction, epLen)
    riverSwim.R = R_true
    riverSwim.P = P_true
    riverSwim.reset()

    return riverSwim

def make_riverSwim_small_reward(epLen=6, nState=6):
    '''
    Makes the benchmark RiverSwim MDP.

    Args:
        NULL - works for default implementation

    Returns:
        riverSwim - Tabular MDP environment
    '''
    nAction = 2
    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0 , 0)
    R_true[nState - 1, 1] = (10000, 0)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

    for s in range(1, nState - 1):
        P_true[s, 1][min(nState - 1, s + 1)] = 1
        P_true[s, 1][s] = 0
        P_true[s, 1][max(0, s-1)] = 0

    P_true[0, 1][0] = 0
    P_true[0, 1][1] = 1
    P_true[nState - 1, 1][nState - 1] = 1
    P_true[nState - 1, 1][nState - 2] = 0

    riverSwim = TabularMDP(nState, nAction, epLen)
    riverSwim.R = R_true
    riverSwim.P = P_true
    riverSwim.reset()

    return riverSwim
def make_SixArms(epLen=20, nState=7):
    '''
    Makes the benchmark RiverSwim MDP.

    Args:
        NULL - works for default implementation

    Returns:
        riverSwim - Tabular MDP environment
    '''
    nAction = 6
    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[1, 0] = (50., 0)
    R_true[1, 1] = (50., 0)
    R_true[1, 2] = (50., 0)
    R_true[1, 3] = (50., 0)
    R_true[1, 5] = (50., 0)
    R_true[2, 1] = (133., 0)
    R_true[3, 2] = (300., 0)
    R_true[4, 3] = (800., 0)
    R_true[5, 4] = (1666, 0)
    R_true[6, 5] = (6000., 0)

    # Transitions
    P_true[0, 0][1] = 1.
    P_true[0, 1][2] = 0.15
    P_true[0, 1][0] = 0.85
    P_true[0, 2][3] = 0.1
    P_true[0, 2][0] = 0.9
    P_true[0, 3][4] = 0.05
    P_true[0, 3][0] = 0.95
    P_true[0, 4][5] = 0.03
    P_true[0, 4][0] = 0.97
    P_true[0, 5][6] = 0.01
    P_true[0, 5][0] = 0.99

    P_true[1, 0][1] = 1.
    P_true[1, 1][1] = 1.
    P_true[1, 2][1] = 1.
    P_true[1, 3][1] = 1.
    P_true[1, 5][1] = 1.
    P_true[1, 4][0] = 1.

    P_true[2, 0][0] = 1.
    P_true[2, 1][2] = 1.
    P_true[2, 2][0] = 1.
    P_true[2, 3][0] = 1.
    P_true[2, 4][0] = 1.
    P_true[2, 5][0] = 1.

    P_true[3, 0][0] = 1.
    P_true[3, 1][0] = 1.
    P_true[3, 2][3] = 1.
    P_true[3, 3][0] = 1.
    P_true[3, 4][0] = 1.
    P_true[3, 5][0] = 1.

    P_true[4, 0][0] = 1.
    P_true[4, 1][0] = 1.
    P_true[4, 2][0] = 1.
    P_true[4, 3][4] = 1.
    P_true[4, 4][0] = 1.
    P_true[4, 5][0] = 1.

    P_true[5, 0][0] = 1.
    P_true[5, 1][0] = 1.
    P_true[5, 2][0] = 1.
    P_true[5, 3][0] = 1.
    P_true[5, 4][5] = 1.
    P_true[5, 5][0] = 1.

    P_true[6, 0][0] = 1.
    P_true[6, 1][0] = 1.
    P_true[6, 2][0] = 1.
    P_true[6, 3][0] = 1.
    P_true[6, 4][0] = 1.
    P_true[6, 5][6] = 1.


    SixArms = TabularMDP(nState, nAction, epLen)
    SixArms.R = R_true
    SixArms.P = P_true
    SixArms.reset()

    return SixArms

def make_deterministicChain(nState, epLen):
    '''
    Creates a deterministic chain MDP with two actions.

    Args:
        nState - int - number of states
        epLen - int - episode length

    Returns:
        chainMDP - Tabular MDP environment
    '''
    nAction = 2

    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0, 1)
    R_true[nState - 1, 1] = (1, 1)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.
        P_true[s, 1][min(nState - 1, s + 1)] = 1.

    chainMDP = TabularMDP(nState, nAction, epLen)
    chainMDP.R = R_true
    chainMDP.P = P_true
    chainMDP.reset()

    return chainMDP

def make_stochasticChain(chainLen):
    '''
    Creates a difficult stochastic chain MDP with two actions.

    Args:
        chainLen - int - total number of states

    Returns:
        chainMDP - Tabular MDP environment
    '''
    nState = chainLen
    epLen = chainLen
    nAction = 2
    pNoise = 1. / chainLen

    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0, 1)
    R_true[nState - 1, 1] = (1, 1)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

        P_true[s, 1][min(nState - 1, s + 1)] = 1. - pNoise
        P_true[s, 1][max(0, s-1)] += pNoise

    stochasticChain = TabularMDP(nState, nAction, epLen)
    stochasticChain.R = R_true
    stochasticChain.P = P_true
    stochasticChain.reset()

    return stochasticChain

def make_bootDQNChain(nState=6, epLen=15, nAction=2):
    '''
    Creates the chain from Bootstrapped DQN

    Returns:
        bootDQNChain - Tabular MDP environment
    '''
    R_true = {}
    P_true = {}

    for s in range(nState):
        for a in range(nAction):
            R_true[s, a] = (0, 0)
            P_true[s, a] = np.zeros(nState)

    # Rewards
    R_true[0, 0] = (0.01, 1)
    R_true[nState - 1, 1] = (1, 1)

    # Transitions
    for s in range(nState):
        P_true[s, 0][max(0, s-1)] = 1.

        P_true[s, 1][min(nState - 1, s + 1)] = 0.5
        P_true[s, 1][max(0, s-1)] = 0.5

    bootDQNChain = TabularMDP(nState, nAction, epLen)
    bootDQNChain.R = R_true
    bootDQNChain.P = P_true
    bootDQNChain.reset()

    return bootDQNChain


def make_hardBanditMDP(epLen, gap=0.01, nAction=2, pSuccess=0.5):
    '''
    Creates a difficult bandit-style MDP which is hard to distinguish.

    Args:
        epLen - int
        gap - double - how much better is best arm
        nAction - int - how many actions

    Returns:
        hardBanditMDP - Tabular MDP environment
    '''
    nState = 3

    R_true = {}
    P_true = {}

    for a in range(nAction):
        # Rewards are independent of action
        R_true[0, a] = (0.5, 1)
        R_true[1, a] = (1, 0)
        R_true[2, a] = (0, 0)

        # Transitions are like a bandit
        P_true[0, a] = np.array([0, pSuccess, 1 - pSuccess])
        P_true[1, a] = np.array([0, 1, 0])
        P_true[2, a] = np.array([0, 0, 1])

    # The first action is a better action though
    P_true[0, 0] = np.array([0, pSuccess + gap, 1 - (pSuccess + gap)])

    hardBanditMDP = TabularMDP(nState, nAction, epLen)
    hardBanditMDP.R = R_true
    hardBanditMDP.P = P_true
    hardBanditMDP.reset()

    return hardBanditMDP




def make_stateBanditMDP(stateMul, gap=0.1):
    '''
    Creates a bandit-style MDP which examines dependence on states.

    Args:
        epLen - int
        gap - double - how much better is best arm
        nAction - int - how many actions

    Returns:
        stateBanditMDP - Tabular MDP environment
    '''
    epLen = 2
    nAction = 2
    nState = 1 + 2 * stateMul

    R_true = {}
    P_true = {}

    for a in range(nAction):
        R_true[0, a] = (0, 0)
        P_true[0, a] = np.zeros(nState)

        for k in range(stateMul):
            for i in range(2):
                s = 1 + (2 * k) + i
                P_true[s, a] = np.zeros(nState)
                P_true[s, a][s] = 1
                R_true[s, a] = (1-i, 0)

    # Important piece is where the transitions go
    P_true[0, 0] = np.ones(nState) / (nState - 1)
    P_true[0, 0][0] = 0

    # Rewarding states
    inds = (np.arange(nState) % 2) > 0
    P_true[0, 1][inds] = (0.5 + gap) / stateMul
    P_true[0, 1][-inds] = (0.5 - gap) / stateMul
    P_true[0, 1][0] = 0

    stateBanditMDP = TabularMDP(nState, nAction, epLen)
    stateBanditMDP.R = R_true
    stateBanditMDP.P = P_true
    stateBanditMDP.reset()

    return stateBanditMDP

def make_confidenceMDP(stateMul, gap=0.1):
    '''
    Creates a bandit-style MDP which examines dependence on states.

    Args:
        epLen - int
        gap - double - how much better is best arm
        nAction - int - how many actions

    Returns:
        confidenceMDP - Tabular MDP environment
    '''
    epLen = 2
    nAction = 1
    nState = 1 + 2 * stateMul

    R_true = {}
    P_true = {}

    for a in range(nAction):
        R_true[0, a] = (0, 0)
        P_true[0, a] = np.zeros(nState)

        for k in range(stateMul):
            for i in range(2):
                s = 1 + (2 * k) + i
                P_true[s, a] = np.zeros(nState)
                P_true[s, a][s] = 1
                R_true[s, a] = (1-i, 0)

    # Important piece is where the transitions go
    P_true[0, 0] = np.ones(nState) / (nState - 1)
    P_true[0, 0][0] = 0

    # Rewarding states
    inds = (np.arange(nState) % 2) > 0

    confidenceMDP = TabularMDP(nState, nAction, epLen)
    confidenceMDP.R = R_true
    confidenceMDP.P = P_true
    confidenceMDP.reset()

    return confidenceMDP


def make_HconfidenceMDP(epLen):
    '''
    Creates a H-dependence bandit confidence.

    Args:
        epLen - int
        gap - double - how much better is best arm
        nAction - int - how many actions

    Returns:
        hardBanditMDP - Tabular MDP environment
    '''
    nState = 3

    R_true = {}
    P_true = {}

    # Rewards are independent of action
    R_true[0, 0] = (0.5, 0)
    R_true[1, 0] = (1, 0)
    R_true[2, 0] = (0, 0)

    # Transitions are like a bandit
    P_true[0, 0] = np.array([0, 0.5, 0.5])
    P_true[1, 0] = np.array([0, 1, 0])
    P_true[2, 0] = np.array([0, 0, 1])

    hardBanditMDP = TabularMDP(nState, 1, epLen)
    hardBanditMDP.R = R_true
    hardBanditMDP.P = P_true
    hardBanditMDP.reset()

    return hardBanditMDP

if __name__ == '__main__':
    env = make_riverSwim()
    print(env.P)

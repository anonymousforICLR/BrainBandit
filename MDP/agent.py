'''
Finite horizon tabular agents.

This is a collection of some of the classic benchmark algorithms for efficient
reinforcement learning in a tabular MDP with little/no prior knowledge.
We provide implementations of:

- PSRL
- Gaussian PSRL
- UCBVI
- BEB
- BOLT
- UCRL2
- Epsilon-greedy

author: iosband@stanford.edu
'''

import numpy as np
import sys
sys.path.append('..')
from model.Lyapunov_Worm_deconstruction import Lyapunov_Worm_Deconstruction
class Agent(object):

    def __init__(self):
        pass

    def update_obs(self, obs, action, reward, newObs):
        '''Add observation to records'''

    def update_policy(self, h):
        '''Update internal policy based upon records'''

    def pick_action(self, obs):
        '''Select an observation based upon the observation'''


class FiniteHorizonAgent(Agent):
    pass

class EpisodicAgent(Agent):
    pass

class DiscountedAgent(Agent):
    pass

class FiniteHorizonTabularAgent(FiniteHorizonAgent):
    '''
    Simple tabular Bayesian learner from Tabula Rasa.

    Child agents will mainly implement:
        update_policy

    Important internal representation is given by qVals and qMax.
        qVals - qVals[state, timestep] is vector of Q values for each action
        qMax - qMax[timestep] is the vector of optimal values at timestep

    '''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1e-6, tau=1e-3, **kwargs):
        '''
        Tabular episodic learner for time-homoegenous MDP.
        Must be used together with true state feature extractor.

        Args:
            nState - int - number of states
            nAction - int - number of actions
            alpha0 - prior weight for uniform Dirichlet
            mu0 - prior mean rewards
            tau0 - precision of prior mean rewards
            tau - precision of reward noise

        Returns:
            tabular learner, to be inherited from
        '''
        # Instantiate the Bayes learner
        self.nState = nState
        self.nAction = nAction
        self.epLen = epLen
        self.alpha0 = alpha0
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau

        self.qVals = {}
        self.qMax = {}

        # Now make the prior beliefs
        self.R_prior = {}
        self.P_prior = {}

        self.state_visitation = [0 for _ in range(nState)]
        for state in range(nState):
            for action in range(nAction):
                self.R_prior[state, action] = (self.mu0, self.tau0)
                self.P_prior[state, action] = (
                    self.alpha0 * np.ones(self.nState, dtype=np.float32))

    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''
        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

    def egreedy(self, state, timestep, epsilon=0.):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        Q = self.qVals[state, timestep]
        nAction = Q.size
        noise = np.random.rand()

        if noise < epsilon:
            action = np.random.choice(nAction)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])

        return action

    def pick_action(self, state, timestep):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep)
        return action
    def count_state(self, state):

        self.state_visitation[state] += 1

    def sample_mdp(self):
        '''
        Returns a single sampled MDP from the posterior.

        Args:
            NULL

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        '''
        R_samp = {}
        P_samp = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                mu, tau = self.R_prior[s, a]
                R_samp[s, a] = mu + np.random.normal() * 1./np.sqrt(tau)
                P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])

        return R_samp, P_samp

    def map_mdp(self):
        '''
        Returns the maximum a posteriori MDP from the posterior.

        Args:
            NULL

        Returns:
            R_hat - R_hat[s, a] is the MAP mean reward for (s,a)
            P_hat - P_hat[s, a] is the MAP transition vector for (s,a)
        '''
        R_hat = {}
        P_hat = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_hat[s, a] = self.R_prior[s, a][0]
                P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        return R_hat, P_hat

    def compute_qVals(self, R, P):
        '''
        Compute the Q values for a given R, P estimates

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = R[s, a] + np.dot(P[s, a], qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_opt(self, R, P, R_bonus, P_bonus):
        '''
        Compute the Q values for a given R, P estimates + R/P bonus

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_bonus - R_bonus[s,a] = bonus for rewards
            P_bonus - P_bonus[s,a] = bonus for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}

        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)

                for a in range(self.nAction):
                    qVals[s, j][a] = (R[s, a] + R_bonus[s, a]
                                      + np.dot(P[s, a], qMax[j + 1])
                                      + P_bonus[s, a] * i)
                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

    def compute_qVals_EVI(self, R, P, R_slack, P_slack):
        '''
        Compute the Q values for a given R, P by extended value iteration

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions
            R_slack - R_slack[s,a] = slack for rewards
            P_slack - P_slack[s,a] = slack for transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
                # Extended value iteration
        qVals = {}
        qMax = {}
        qMax[self.epLen] = np.zeros(self.nState)

        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState)

            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction)

                for a in range(self.nAction):
                    rOpt = R[s, a] + R_slack[s, a]

                    # form pOpt by extended value iteration, pInd sorts the values
                    pInd = np.argsort(qMax[j + 1])
                    pOpt = P[s, a]
                    if pOpt[pInd[self.nState - 1]] + P_slack[s, a] * 0.5 > 1:
                        pOpt = np.zeros(self.nState)
                        pOpt[pInd[self.nState - 1]] = 1
                    else:
                        pOpt[pInd[self.nState - 1]] += P_slack[s, a] * 0.5

                    # Go through all the states and get back to make pOpt a real prob
                    sLoop = 0
                    while np.sum(pOpt) > 1:
                        worst = pInd[sLoop]
                        pOpt[worst] = max(0, 1 - np.sum(pOpt) + pOpt[worst])
                        sLoop += 1

                    # Do Bellman backups with the optimistic R and P
                    qVals[s, j][a] = rOpt + np.dot(pOpt, qMax[j + 1])

                qMax[j][s] = np.max(qVals[s, j])

        return qVals, qMax

#-----------------------------------------------------------------------------
# PSRL
#-----------------------------------------------------------------------------

class PSRL(FiniteHorizonTabularAgent):
    '''
    Posterior Sampling for Reinforcement Learning
    '''

    def update_policy(self, h=False):
        '''
        Sample a single MDP from the posterior and solve for optimal Q values.

        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_samp, P_samp)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax
    def sample_mdp(self):
        '''
        Returns a single sampled MDP from the posterior.

        Args:
            NULL

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        '''
        R_samp = {}
        P_samp = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                mu, tau = self.R_prior[s, a]
                R_samp[s, a] = mu + np.random.normal() * 1./np.sqrt(tau)
                P_samp[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        return R_samp, P_samp
class UCRL2(FiniteHorizonTabularAgent):
    '''Classic benchmark optimistic algorithm'''

    def __init__(self, nState, nAction, epLen,
                 delta=0.05, scaling=1., **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        super(UCRL2, self).__init__(nState, nAction, epLen,
                                    alpha0=1e-5, tau0=0.0001)
        self.delta = delta
        self.scaling = scaling


    def get_slack(self, time):
        '''
        Returns the slackness parameters for UCRL2

        Args:
            time - int - grows the confidence sets

        Returns:
            R_slack - R_slack[s, a] is the confidence width for UCRL2 reward
            P_slack - P_slack[s, a] is the confidence width for UCRL2 transition
        '''
        R_slack = {}
        P_slack = {}
        delta = self.delta
        scaling = self.scaling
        for s in range(self.nState):
            for a in range(self.nAction):
                nObsR = max(self.R_prior[s, a][1] - self.tau0, 1.)
                R_slack[s, a] = scaling * np.sqrt((4 * np.log(2 * self.nState * self.nAction * (time + 1) / delta)) / float(nObsR))

                nObsP = max(self.P_prior[s, a].sum() - self.alpha0, 1.)
                P_slack[s, a] = scaling * np.sqrt((4 * self.nState * np.log(2 * self.nState * self.nAction * (time + 1) / delta)) / float(nObsP))
        return R_slack, P_slack

    def update_policy(self, time=100):
        '''
        Compute UCRL2 Q-values via extended value iteration.
        '''
        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Compute the slack parameters
        R_slack, P_slack = self.get_slack(time)

        # Perform extended value iteration
        qVals, qMax = self.compute_qVals_EVI(R_hat, P_hat, R_slack, P_slack)

        self.qVals = qVals
        self.qMax = qMax

class OptimisticPSRL(PSRL):
    '''
    Optimistic Posterior Sampling for Reinforcement Learning
    '''
    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1., tau=1., nSamp=10, **kwargs):
        '''
        Just like PSRL but we take optimistic over multiple samples

        Args:
            nSamp - int - number of samples to use for optimism
        '''
        super(OptimisticPSRL, self).__init__(nState, nAction, epLen,
                                             alpha0, mu0, tau0, tau)
        self.nSamp = nSamp

    def update_policy(self, h=False):
        '''
        Take multiple samples and then take the optimistic envelope.

        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()
        qVals, qMax = self.compute_qVals(R_samp, P_samp)
        self.qVals = qVals
        self.qMax = qMax

        for i in range(1, self.nSamp):
            # Do another sample and take optimistic Q-values
            R_samp, P_samp = self.sample_mdp()
            qVals, qMax = self.compute_qVals(R_samp, P_samp)

            for timestep in range(self.epLen):
                self.qMax[timestep] = np.maximum(qMax[timestep],
                                                 self.qMax[timestep])
                for state in range(self.nState):
                    self.qVals[state, timestep] = np.maximum(qVals[state, timestep],
                                                             self.qVals[state, timestep])

    def sample_mdp(self):
        '''
        Returns a single sampled MDP from the posterior.

        Args:
            NULL

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        '''
        R_samp = {}
        P_samp = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_samp[s, a] = self.R_prior[s, a][0]
                P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])

        return R_samp, P_samp
class OTS_MDP(FiniteHorizonTabularAgent):
    '''
    Posterior Sampling for Reinforcement Learning
    '''
    def __init__(self, nState, nAction, epLen,**kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            delta - double - probability scale parameter
            scaling - double - rescale default confidence sets
        '''
        super(OTS_MDP, self).__init__(nState, nAction, epLen)
    def update_policy(self, h=False):
        '''
        Sample a single MDP from the posterior and solve for optimal Q values.

        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.sample_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_samp, P_samp)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax
    def sample_mdp(self):
        '''
        Returns a single sampled MDP from the posterior.

        Args:
            NULL

        Returns:
            R_samp - R_samp[s, a] is the sampled mean reward for (s,a)
            P_samp - P_samp[s, a] is the sampled transition vector for (s,a)
        '''
        R_samp = {}
        P_samp = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                mu, tau = self.R_prior[s, a]
                random_n = max(np.random.normal(), 0)
                R_samp[s, a] = mu + random_n * 1./np.sqrt(tau)
                P_samp[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        return R_samp, P_samp

class UBE(PSRL):
    '''
    Optimistic Posterior Sampling for Reinforcement Learning
    '''
    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1e-6, tau=1e-3, nSamp=1, **kwargs):
        '''
        Just like PSRL but we take optimistic over multiple samples

        Args:
            nSamp - int - number of samples to use for optimism
        '''
        super(UBE, self).__init__(nState, nAction, epLen,
                                  alpha0, mu0, tau0, tau)

        self.qVals = {}
        self.qMax = {}
        self.qVar = {}
    def map_mdp(self):
        '''
        Returns the maximum a posteriori MDP from the posterior.

        Args:
            NULL

        Returns:
            R_hat - R_hat[s, a] is the MAP mean reward for (s,a)
            P_hat - P_hat[s, a] is the MAP transition vector for (s,a)
        '''
        R_hat = {}
        P_hat = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                R_hat[s, a] = (self.R_prior[s, a][0], self.R_prior[s, a][1])
                P_hat[s, a] = self.P_prior[s, a] / np.sum(self.P_prior[s, a])

        return R_hat, P_hat
    def update_policy(self, h=False):
        '''
        Take multiple samples and then take the optimistic envelope.

        Works in place with no arguments.
        '''
        # Sample the MDP
        R_samp, P_samp = self.map_mdp()
        qVals, qMax, qVar = self.compute_qVals(R_samp, P_samp)
        self.qVals = qVals
        self.qMax = qMax
        self.qVar = qVar




    def compute_qVals(self, R, P):
        '''
        Compute the Q values for a given R, P estimates

        Args:
            R - R[s,a] = mean rewards
            P - P[s,a] = probability vector of transitions

        Returns:
            qVals - qVals[state, timestep] is vector of Q values for each action
            qMax - qMax[timestep] is the vector of optimal values at timestep
        '''
        qVals = {}
        qMax = {}
        qVar = {}
        qVar_pi = {}
        for s in range(self.nState):
            qVar[s,self.epLen] = np.zeros(self.nAction, dtype=np.float32)
            qVals[s,self.epLen] = np.zeros(self.nAction, dtype=np.float32)
        qMax[self.epLen] = np.zeros(self.nState, dtype=np.float32)
        qVar_pi[self.epLen] = np.zeros(self.nState, dtype=np.float32)


        for i in range(self.epLen):
            j = self.epLen - i - 1
            qMax[j] = np.zeros(self.nState, dtype=np.float32)
            qVar_pi[j] = np.zeros(self.nState, dtype=np.float32)
            for s in range(self.nState):
                qVals[s, j] = np.zeros(self.nAction, dtype=np.float32)
                qVar[s, j] = np.zeros(self.nAction, dtype=np.float32)
                for a in range(self.nAction):
                    qVals[s, j][a] = R[s, a][0] + np.dot(P[s, a], qMax[j + 1])
                    qVar[s, j][a] = 1. / R[s, a][1] + np.dot(P[s, a], qVar_pi[j+1])
                qMax[j][s] = np.max(qVals[s, j])
                qVar_pi[j][s] = qVar[s, j][np.argmax(qVals[s, j])]

        return qVals, qMax, qVar

    def pick_action(self, state, timestep):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep)
        return action

    def egreedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''

        qVals = self.qVals[state, timestep]
        qVar = self.qVar[state, timestep]
        qCov = np.diag(qVar)
        Q = qVals
        # print(f"Qsample: {np.random.multivariate_normal(qVals, qCov)}")
        nAction = Q.size
        noise = np.random.rand()

        if noise < epsilon:
            action = np.random.choice(nAction)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])
        return action
    def update_obs(self, oldState, action, reward, newState, pContinue, h):
        '''
        Update the posterior belief based on one transition.

        Args:
            oldState - int
            action - int
            reward - double
            newState - int
            pContinue - 0/1
            h - int - time within episode (not used)

        Returns:
            NULL - updates in place
        '''
        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

class UBE_TS(UBE):
    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1e-6, tau=1e-3, **kwargs):
        super(UBE_TS, self).__init__(nState, nAction, epLen,
                                               alpha0, mu0, tau0, tau)

    def egreedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''

        qVals = self.qVals[state, timestep]
        qVar = self.qVar[state, timestep]
        qCov = np.diag(qVar)
        Q = np.random.multivariate_normal(qVals, qCov)
        nAction = Q.size
        noise = np.random.rand()

        if noise < epsilon:
            action = np.random.choice(nAction)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])
        return action


class UBE_UCB(UBE):
    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1e-6, tau=1e-3, c=1., **kwargs):
        super(UBE_UCB, self).__init__(nState, nAction, epLen,
                                     alpha0, mu0, tau0, tau)
        self.c = c

    def egreedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''

        qVals = self.qVals[state, timestep]
        qVar = self.qVar[state, timestep]
        qCov = np.diag(qVar)
        Q = qVals + self.c * np.sqrt(qCov)
        nAction = Q.size
        noise = np.random.rand()

        if noise < epsilon:
            action = np.random.choice(nAction)
        else:
            action = np.random.choice(np.where(Q == Q.max())[0])
        return action
class UBE_BBN(UBE):
    '''
    Optimistic Posterior Sampling for Reinforcement Learning
    '''

    def __init__(self, nState, nAction, epLen,
                 alpha0=1., mu0=0., tau0=1e-6, tau=1e-3, BBN_dim=None, BBN_para=None, BBN_scale=False, **kwargs):

        super(UBE_BBN, self).__init__(nState, nAction, epLen,
                                      alpha0, mu0, tau0, tau)
        self.qVals = {}
        self.qMax = {}
        self.qVar = {}
        self.dim = BBN_dim
        self.para = BBN_para
        self.scale = BBN_scale
    def softmax_scale(self,x):

        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum() * 2

    def egreedy(self, state, timestep, epsilon=0):

        qVals = self.qVals[state, timestep]
        qVar = self.qVar[state, timestep]
        net = Lyapunov_Worm_Deconstruction(varargin=self.para, dim=self.dim)
        if self.scale == True:
            uncertainty = self.softmax_scale(np.sqrt(qVar))
        else:
            uncertainty = np.sqrt(qVar)
        action = net.decide_simulation_multi_dim(qVals, uncertainty, save_history=True, init=True)
        return action

class EpsilonGreedy(FiniteHorizonTabularAgent):
    '''Epsilon greedy agent'''

    def __init__(self, nState, nAction, epLen, epsilon=0.1, **kwargs):
        '''
        As per the tabular learner, but prior effect --> 0.

        Args:
            epsilon - double - probability of random action
        '''
        super(EpsilonGreedy, self).__init__(nState, nAction, epLen,
                                            alpha0=0.0001, tau0=0.0001)
        self.epsilon = epsilon

    def update_policy(self, time=False):

        # Output the MAP estimate MDP
        R_hat, P_hat = self.map_mdp()

        # Solve the MDP via value iteration
        qVals, qMax = self.compute_qVals(R_hat, P_hat)

        # Update the Agent's Q-values
        self.qVals = qVals
        self.qMax = qMax

    def pick_action(self, state, timestep):
        '''
        Default is to use egreedy for action selection
        '''
        action = self.egreedy(state, timestep, self.epsilon)
        return action

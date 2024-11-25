import os.path
from environment import make_riverSwim,make_riverSwim_origin, make_SixArms, make_map, make_FourRooms
from feature_extractor import FeatureTrueState
from agent import PSRL, UCRL2, OptimisticPSRL, OTS_MDP, EpsilonGreedy,UBE, UBE_TS, UBE_UCB, UBE_BBN
from experiment import run_finite_tabular_experiment
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
def visualize_vistation(state_visitation, agent_name, ep):


    n_state = len(state_visitation)
    n = int(np.sqrt(n_state))
    visitation_map = np.zeros((n, n),dtype=int)
    for i in range(n_state):
        visitation_map[int(i/n), i%n] = state_visitation[i]

    sns.heatmap(visitation_map, vmax=300)
    plt.title(f"Visitation counts for {agent_name} at ep {ep}")
    plt.show()

def visualize_global_uncertainty(qVar, timesteps, n, action):

    global_uncertainty = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            global_uncertainty[i,j] = qVar[i*n+j, timesteps][action]
    sns.heatmap(global_uncertainty)
    plt.title(f"Global Uncertainty (timesteps={timesteps}, actions={action})")
    plt.show()

def visualize_local_uncertainty(R, timesteps, n, action):

    local_uncertainty = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            local_uncertainty[i,j] = 1 / R[i*n+j,action][1]

    sns.heatmap(local_uncertainty)
    plt.title(f"Local Uncertainty (timesteps={timesteps}, actions={action})")
    plt.show()

def visualize_value(Q, timesteps, n, action):

    value = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            value[i,j] = Q[i*n+j,action][action]

    sns.heatmap(value)
    plt.title(f"Q alue (timesteps={timesteps}, actions={action})")
    plt.show()

def coverage_rate(state_visitation, state_number):
    coverage_state = np.sum(np.array(state_visitation) !=0 )
    return coverage_state/float(state_number)
if __name__ == '__main__':

    regret_list = []
    base_path = 'saved/map'
    # seed = np.random.randint(100)
    seed = 0
    nEps = 300
    length = 150
    save = False
    agent_name = 'UBE_BBN'
    T = 10
    Visualize = True
    VisualizeFrequency = 50
    coverage = np.zeros(shape=(T,nEps))
    dim = 4
    BBN_para = {
            'step': 100,  # how many steps to run the brain circuit before executing the next movement
            'tau': np.ones(dim),  # decay time constant
            'weights_in': np.ones(dim) * 1,  # input weights
            'rs': np.ones(dim) * .5,  #
            'w': np.ones(dim) * 4,  # weight of mutual inhibition
            'k': 7. * np.ones(dim),  # sigmoid center
            'n': 1.5 * np.ones(dim),  # sigmoid slope
            'bi': np.ones(dim) * 6.25,  # baseline production
            'dt': 0.4,  # size of timesteps
            'nsf': 0.,  # noise level
            'w_avg_comp': 1.,
            'w_std_comp': 1.
        }
    for i in tqdm(range(T)):
        env = make_map(epLen=length)
        f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
        alpha0 = 1. / env.nState
        agent = eval(agent_name)(nState=env.nState, nAction=env.nAction, epLen=env.epLen,  BBN_dim=dim,BBN_para=BBN_para ,BBN_scale=True)
        targetPath = base_path + f'_seed{seed}.csv'
        data = []
        qVals, qMax = env.compute_qVals()
        np.random.seed(seed)
        cumRegret = 0
        cumReward = 0
        empRegret = 0
        for ep in range(0, nEps):
            # Reset the environment
            env.reset()
            epMaxVal = qMax[env.timestep][env.state]
            agent.update_policy(ep)
            epReward = 0
            epRegret = 0
            pContinue = 1
            while pContinue > 0:
                h, oldState = f_ext.get_feat(env)
                agent.count_state(oldState)
                action = agent.pick_action(oldState, h)
                epRegret += qVals[oldState, h].max() - qVals[oldState, h][action]
                reward, newState, pContinue = env.advance(action)
                epReward += reward
                agent.update_obs(oldState, action, reward, newState, pContinue, h)
            coverage[i][ep] = coverage_rate(agent.state_visitation, 100)
            cumReward += epReward
            cumRegret += epRegret
            empRegret += (epMaxVal - epReward)
            data.append([ep, epReward, cumReward, cumRegret, empRegret])
            if Visualize and ep % VisualizeFrequency == 0:
                visualize_vistation(agent.state_visitation, agent_name, ep+1)
            seed+=1
        if save:
            dt = pd.DataFrame(data,
                              columns=['episode', 'epReward', 'cumReward',
                                       'cumRegret', 'empRegret'])
            # print('Writing to file ' + targetPath)
            dt.to_csv(targetPath, index=False, float_format='%.2f')
            print(f'saved file to {targetPath}')
        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')
        regret = cumRegret
        regret_list.append(regret)

        print(coverage_rate(agent.state_visitation, 100))
        # agent.state_visitation

    path = f"saved/MapL{length}"
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(f"saved/MapL{length}/coverage_rate_" + agent_name + 'optimistic', coverage)
    if 'UBE' in agent_name:
        visualize_global_uncertainty(agent.qVar,0,10,0)
        visualize_local_uncertainty(agent.R_prior,0,10,0)


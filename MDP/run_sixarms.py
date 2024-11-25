
from environment import make_riverSwim,make_riverSwim_origin, make_SixArms
from feature_extractor import FeatureTrueState
from agent import PSRL, UCRL2, OptimisticPSRL, OTS_MDP, EpsilonGreedy, UBE_TS, UBE_UCB, UBE_BBN
from experiment import run_finite_tabular_experiment
import numpy as np
import os
if __name__ == '__main__':
    dim = 6
    BBN_para = {
        'step': 100,  # how many steps to run the brain circuit before executing the next movement
        'tau': np.ones(dim),  # decay time constant
        'weights_in': np.ones(dim) * 1.,  # input weights
        'rs': np.ones(dim) * .5,  #
        'w': np.ones(dim) * 4,  # weight of mutual inhibition
        'k': 7. * np.ones(dim),  # sigmoid center
        'n': 1.5 * np.ones(dim),  # sigmoid slope
        'bi': np.ones(dim) * 6.25,  # baseline production
        'dt': 0.4,  # size of timesteps
        'nsf': 0.,  # noise level
        'w_avg_comp': 1e-2,
        'w_std_comp': 1e-1
    }
    regret_list = []
    agent_name = 'UBE_BBN'
    root_path = 'saved/sixarms/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    base_path = 'saved/sixarms/'+agent_name
    seed = 0
    for i in range(10):
        env = make_SixArms()
        f_ext = FeatureTrueState(env.epLen, env.nState, env.nAction, env.nState)
        agent = eval(agent_name)(nState=env.nState, nAction=env.nAction, epLen=env.epLen, alpha0=1 / env.nState, BBN_dim=dim,BBN_para=BBN_para,BBN_scale=False)
        seed += 1
        targetPath = base_path + f'_seed{seed}.csv'
        regret = run_finite_tabular_experiment(agent, env, f_ext, 1000, seed,
                        recFreq=100, fileFreq=1, targetPath=targetPath, save=True)
        regret_list.append(regret)
    print(np.mean(regret_list))
    print(np.std(regret_list))
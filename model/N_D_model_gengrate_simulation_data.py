import time  # @Start date : 2023/1/5

import numpy as np
import sys
from multiprocessing import  Process
sys.path.append('..')
from model.Lyapunov_Worm import Lyapunov_Worm_multi_D
from tqdm import  tqdm

def gengrate_simulation_data_n_D(n:int):
    dim = n
    varargin = {
        'step': 20000,  # how many steps to run the brain circuit before executing the next movement
        'tau': np.ones(dim),  # decay time constant
        'weights_in': np.ones(dim) * 1.,  # input weights
        'rs': np.ones(dim) * 0.5,  #
        'w': np.ones(dim) * 4.,  # weight of mutual inhibitionstd[0]
        'k': 7. * np.ones(dim),  # sigmoid center
        'n': 1.5 * np.ones(dim),  # sigmoid slope
        'bi': np.ones(dim) * 5.5,  # baseline production
        'dt': 0.4,  # size of timesteps
        'nsf': 0.1,  # noise level
    }

    r_std_med = [np.sqrt(0.75) for _ in range(dim-2)]
    r_std = [np.sqrt(2.5)] + r_std_med + [np.sqrt(0.25)]
    r_mean = [0 for _ in range(dim)]
    r_list = []
    total_n = 30


    for t in tqdm(range(total_n)):
        net = Lyapunov_Worm_multi_D(varargin=varargin, dim=dim)
        net.decide_simulation_multi_dim(r_mean, r_std, save_history=True, init=True)
        r1 = net.get_choice_probability(0)
        r2 = net.get_choice_probability(dim - 2)
        r3 = net.get_choice_probability(dim - 1)
        r_list.append(r1)
        r_list.append(r2)
        r_list.append(r3)
    r_list = np.array(r_list).reshape(total_n, 3)
    np.save(f'saved/{dim}D_model_simulation_neutral', r_list)
if __name__ == '__main__':


    process_list = []
    for i in range(2,11,1):
        p = Process(target=gengrate_simulation_data_n_D,args=(i,))
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()







import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from BanditGame import Play
from Expected_Value import CalculateMeanValue
import pandas as pd

bandit_mean = 0
mean_std = 1

dim = 3
varargin = {
"seeking":{
    'step': 400,  # how many steps to run the brain circuit before executing the next movement
    'tau': np.ones(dim),  # decay time constant
    'weights_in': np.ones(dim) * 1.,  # input weights
    'rs': np.ones(dim) * .5,  #
    'w': np.ones(dim) * 4,  # weight of mutual inhibition
    'k': 7 * np.ones(dim),  # sigmoid center
    'n': 2 * np.ones(dim),  # sigmoid slope
    'bi': np.ones(dim) * 6.25,  # baseline production
    'dt': 0.4,  # size of timesteps
    'nsf': 0.1,  # noise level
},
"neutral":{
    'step': 400,  # how many steps to run the brain circuit before executing the next movement
    'tau': np.ones(dim),  # decay time constant
    'weights_in': np.ones(dim) * 1.,  # input weights
    'rs': np.ones(dim) * .5,  #
    'w': np.ones(dim) * 4,  # weight of mutual inhibition
    'k': 7 * np.ones(dim),  # sigmoid center
    'n': 2 * np.ones(dim),  # sigmoid slope
    'bi': np.ones(dim) * 5.5,  # baseline production
    'dt': 0.4,  # size of timesteps
    'nsf': 0.1,  # noise level
},
"averse":{
    'step': 400,  # how many steps to run the brain circuit before executing the next movement
    'tau': np.ones(dim),  # decay time constant
    'weights_in': np.ones(dim) * 1.,  # input weights
    'rs': np.ones(dim) * .5,  #
    'w': np.ones(dim) * 4,  # weight of mutual inhibition
    'k': 7 * np.ones(dim),  # sigmoid center
    'n': 2 * np.ones(dim),  # sigmoid slope
    'bi': np.ones(dim) * 4.75,  # baseline production
    'dt': 0.4,  # size of timesteps
    'nsf': 0.1,  # noise level
}
}


bandit_std_list = [3,1,0.5]
prior = [[-1,1] for _ in range(dim) ]
print(bandit_std_list)
p = Play(varargin=varargin['seeking'], force_times=0, dim=dim, prior=prior,init=True, bandit_mean=bandit_mean, mean_std=mean_std,
         bandit_std_list=bandit_std_list, subject=1, block=10000, trial=30, save=True,
         save_path=f'saved/bandit_performance_raw_data_{bandit_std_list}.csv', save_mode='w')
p.play()
k = CalculateMeanValue(raw_path=f'saved/bandit_performance_raw_data_{bandit_std_list}.csv', save_path=f'saved/bandit_performance_analysed_data_{bandit_std_list}.csv',
                  trial=30, dim=dim)
k.run()

import sys
sys.path.append("..")
import numpy as np
import matplotlib.pyplot as plt
from BanditGame import Play
from Expected_Value import CalculateMeanValue
import pandas as pd

bandit_mean = 0
mean_std = 1

dim = 2
varargin = {
"seeking":{
    'step': 400,  # how many steps to run the brain circuit before executing the next movement
    'tau': np.ones(dim),  # decay time constant
    'weights_in': np.ones(dim) * 1.,  # input weights
    'rs': np.ones(dim) * .5,  #
    'w': np.ones(dim) * 4,  # weight of mutual inhibition
    'k': 7 * np.ones(2),  # sigmoid center
    'n': 2 * np.ones(2),  # sigmoid slope
    'bi': np.ones(2) * 6.25,  # baseline production
    'dt': 0.4,  # size of timesteps
    'nsf': 0.1,  # noise level
},
"neutral":{
    'step': 400,  # how many steps to run the brain circuit before executing the next movement
    'tau': np.ones(dim),  # decay time constant
    'weights_in': np.ones(dim) * 1.,  # input weights
    'rs': np.ones(dim) * .5,  #
    'w': np.ones(dim) * 4,  # weight of mutual inhibition
    'k': 7 * np.ones(2),  # sigmoid center
    'n': 2 * np.ones(2),  # sigmoid slope
    'bi': np.ones(2) * 5.5,  # baseline production
    'dt': 0.4,  # size of timesteps
    'nsf': 0.1,  # noise level
},
"averse":{
    'step': 400,  # how many steps to run the brain circuit before executing the next movement
    'tau': np.ones(dim),  # decay time constant
    'weights_in': np.ones(dim) * 1.,  # input weights
    'rs': np.ones(dim) * .5,  #
    'w': np.ones(dim) * 4,  # weight of mutual inhibition
    'k': 7 * np.ones(2),  # sigmoid center
    'n': 2 * np.ones(2),  # sigmoid slope
    'bi': np.ones(2) * 4.75,  # baseline production
    'dt': 0.4,  # size of timesteps
    'nsf': 0.1,  # noise level
}
}

RU = 1
std_list = [3]
mode_list = ['seeking','neutral','averse']
for mode in mode_list:
    for std in std_list:
        bandit_std_list = [std, std-RU]
        print(bandit_std_list)
        prior = [[-1,1],[-1,1]]
        p = Play(varargin=varargin[mode], force_times=0, prior=prior,init=True, bandit_mean=bandit_mean, mean_std=mean_std,
                 bandit_std_list=bandit_std_list, subject=1, block=10000, trial=20, save=True,
                 save_path=f'bandit_performance_raw_data_std={std}_RU={RU}_{mode}.csv', save_mode='w')
        p.play()
        k = CalculateMeanValue(raw_path=f'bandit_performance_raw_data_std={std}_RU={RU}_{mode}.csv', save_path=f'bandit_performance_analysed_data_std={std}_RU={RU}_{mode}.csv',
                          trial=20)
        k.run()

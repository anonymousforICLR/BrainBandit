
import numpy as np
import pandas as pd
from tqdm import tqdm
import copy
def kalman(mean, var, tau, reward):
    lr = var / (var+tau)
    mean = mean + lr * (reward - mean)
    var = var - lr * var
    return mean, var

class Kalman_filter:
    def __init__(self, raw_path, save_path, trial:int=20, dim:int=2, tau:list= [16,16], var_0:list = [100,100]):
        self.raw_path = raw_path
        self.save_path = save_path
        self.trial = trial
        self.dim = dim
        self.tau = tau
        self.var_0 = var_0
    def run(self):
        Data = pd.read_csv(self.raw_path)
        mean_list_block = [[] for _ in range(self.dim)]
        var_list_block = [[] for _ in range(self.dim)]
        total_trial = self.trial
        optimal_list = []
        for block_i in tqdm(range(0, len(Data), total_trial)):
            optimal_choice = np.argmax([Data[f'mu{dim+1}'][block_i] for dim in range(self.dim)])
            if Data['trial'][block_i] == 1 :
                mean_list_trial = np.zeros(self.dim)
                var_list_trial = self.tau
                tau_list = self.var_0
                if Data['cond'][block_i] == 1:
                    tau_list = [16,1e-6]
                elif Data['cond'][block_i] == 2:
                    tau_list = [1e-6, 16]
                elif Data['cond'][block_i] == 3:
                    tau_list = [16, 16]
                elif Data['cond'][block_i] == 4:
                    tau_list = [1e-6, 1e-6]
                for trial_i in range(total_trial):
                    for mean_i in range(self.dim):
                        mean_list_block[mean_i].append(mean_list_trial[mean_i])
                        var_list_block[mean_i].append(var_list_trial[mean_i])
                    choice =  Data['choice'][block_i+trial_i] - 1
                    is_optimal = True
                    if choice != optimal_choice:
                        is_optimal = False
                    optimal_list.append(is_optimal)
                    mean_list_trial[choice],var_list_trial[choice] = kalman(mean_list_trial[choice], var_list_trial[choice], tau_list[choice], Data['reward'][block_i + trial_i])

        for i in range(self.dim):
            Data[f'Q{i}'] = mean_list_block[i]
            Data[f'Var{i}'] = var_list_block[i]
        Data['optimal'] = optimal_list
        Data = Data.round(4)
        Data.to_csv(self.save_path, index=False)
        print(f"successfully save data to {self.save_path}")

class CalculateMeanValue:
    def __init__(self, raw_path, save_path, trial:int=20, dim:int=2, prior=None):
        self.raw_path = raw_path
        self.save_path = save_path
        self.trial = trial
        self.dim = dim
        self.prior = prior
    def run(self):
        Data = pd.read_csv(self.raw_path)
        mean_list_block = [[] for _ in range(self.dim)]
        total_trial = self.trial
        optimal_list = []
        for block_i in tqdm(range(0, len(Data), total_trial)):
            optimal_choice = np.argmax([Data[f'mu{dim}'][block_i] for dim in range(self.dim)])
            if Data['trial'][block_i] == 1 :
                mean_list_trial = np.zeros(self.dim)
                reward_list = [self.prior[_].copy() for _ in range(self.dim)] if self.prior else  [[] for _ in range(self.dim)]
                for trial_i in range(total_trial):
                    for mean_i in range(self.dim):
                        mean_list_block[mean_i].append(mean_list_trial[mean_i])
                    choice =  Data['choice'][block_i+trial_i] - 1
                    is_optimal = True
                    if choice != optimal_choice:
                        is_optimal = False
                    optimal_list.append(is_optimal)
                    reward_list[choice].append(Data['reward'][block_i + trial_i])
                    for mean_i in range(self.dim):

                        mean_list_trial[mean_i] = np.mean(reward_list[mean_i]) if len(reward_list[mean_i]) > 0 else 0

        for i in range(self.dim):
            Data[f'Q{i}'] = mean_list_block[i]
            Data[f'Q{i}'] = Data[f'Q{i}']
        Data['optimal'] = optimal_list
        Data = Data.round(4)
        Data.to_csv(self.save_path, index=False)
        print(f"successfully save data to {self.save_path}")



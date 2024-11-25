# @Start date : 2022/12/12

import sys
sys.path.append("..")
import numpy as np
from BanditGame import Game
from scipy import stats
import pandas as pd
from tqdm import tqdm


class Optimistic_Thompson_Sampling_Model:
    def __init__(self, n_choice, tau, mean = 0, mean_std=1, std_list=None, prior_std = None):

        self.n = n_choice
        self.mean_std = mean_std
        self.mean = mean
        self.std_list = std_list
        self.expected_mean = np.zeros(n_choice)
        self.expected_var = np.ones(n_choice) * mean_std * mean_std if prior_std is None else np.array(prior_std)
        self.game =  Game(n_bandits=self.n, mean=self.mean, std_mean=self.mean_std, std_list=self.std_list)
        self.mu = self.game.mean_list
        self.std = self.game.std
        self.times = 0
        self.each_times = np.zeros(n_choice)
        self.tau = tau
        self.choice_history = []
        self.std_gaussian = stats.norm(0, 1)
        self.result = []

    def choose(self, index):
        self.choice_history.append(index)
        reward = self.game.choose(index)
        self.times += 1
        self.each_times[index] += 1
        updating_rate = self.expected_var[index] / (self.expected_var[index] + self.tau[index])
        self.expected_mean[index] = self.expected_mean[index] + updating_rate * (reward - self.expected_mean[index])
        self.expected_var[index] = self.expected_var[index] - updating_rate * self.expected_var[index]
        return reward

    def play(self, total_epochs, block_id=None, optimal_choice=None, sample_number=1):
        for i in range(total_epochs):
            expected_mean = self.expected_mean.copy()
            expected_var = self.expected_var.copy()
            sample_list = []
            for index in range(self.n):
                tmp = np.random.normal(expected_mean[index],np.sqrt(expected_var[index]),size=sample_number)
                tmp[tmp < expected_mean[index]] = expected_mean[index]
                sample_list.append(np.mean(tmp))

            choice = np.argmax(sample_list)
            reward = self.choose(choice)
            is_optimal = True
            if choice != optimal_choice:
                is_optimal = False
            self.result.append([block_id+1, i + 1, *self.mu, *self.std,*expected_mean,*expected_var,choice,is_optimal, reward])

    def get_mean_list(self):
        return self.game.get_mean()


class Play_OTS:
    def __init__(self, subject: int = 100, block: int = 10, trial: int = 10, sample_number: int = 10, save: bool = True, save_path: str = None,
                 save_mode: str = 'w',dim = 2, mean = 0, mean_std = 1, std_list=None,tau_list=None, prior_std = None):
        self.subject = subject
        self.block = block
        self.trial = trial
        self.dim = dim
        self.optimal_probability = []
        self.result = []
        self.save = save
        self.save_path = save_path
        self.save_mode = save_mode
        self.std_list = std_list
        self.tau_list = tau_list
        if self.tau_list is None :
            self.tau_list = np.array(std_list) ** 2 + np.ones(len(std_list)) * 1e-6
        self.mean = mean
        self.mean_std = mean_std
        self.sample_number = sample_number
        self.prior_std = prior_std

    def play(self):
        for subject in range(self.subject):
            block_optimal_times = 0
            for block in tqdm(range(self.block)):
                model = Optimistic_Thompson_Sampling_Model(n_choice=self.dim, tau=self.tau_list, mean=self.mean, mean_std=self.mean_std, std_list=self.std_list, prior_std=self.prior_std)
                mean_list = model.get_mean_list()
                optimal = np.argmax(mean_list)
                model.play(self.trial, block_id=block,optimal_choice = optimal, sample_number=self.sample_number)
                block_optimal_times += sum(model.choice_history == optimal)
                results = model.result
                self.result.append(results)
            self.optimal_probability.append(block_optimal_times / (self.trial * self.block))
        if self.save:
            self.save_csv()

    def get_optimal_probability(self):
        return np.mean(self.optimal_probability)

    def save_csv(self):
        result = np.array(self.result)
        result = result.reshape(self.block * self.trial * self.subject, -1)
        mu_list = [f'mu{i }' for i in range(self.dim)]
        std_list = [f'std{i }' for i in range(self.dim)]
        Q_list = [f'Q{i }' for i in range(self.dim)]
        U_list = [f'U{i }' for i in range(self.dim)]
        result = pd.DataFrame(data=result,
                              columns=['block', 'trial', *mu_list,*std_list, *Q_list, *U_list,'choice','optimal', 'reward'])
        result[[ 'block', 'trial', 'choice']] = result[
            [ 'block', 'trial', 'choice']].astype(int)
        result[['optimal']] = result[
            ['optimal']].astype(bool)
        if self.save_mode == 'a':
            header = False
        else:
            header = True
        result.to_csv(self.save_path, header=header, index=False, mode=self.save_mode)
        print("successfully save data to " + self.save_path)


if __name__ == '__main__':

    dim = 3
    std_list = [3,1,0.5]
    prior_std = [1., 1., 1.]
    tau_list = [1., 1., 1.]
    game1 = Play_OTS(subject=1, block=10000, trial=dim*10, dim=dim,mean_std=1, sample_number=1,save=True,save_path=f"saved/OTS_{std_list}.csv", save_mode='w',std_list=std_list, prior_std=prior_std, tau_list=tau_list)
    game1.play()
    p = game1.get_optimal_probability()
    print(p)


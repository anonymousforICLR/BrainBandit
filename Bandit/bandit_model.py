# @Start date : 2022/12/12
import random
import sys
sys.path.append("..")
import numpy as np
from BanditGame import Game
from scipy import stats
import pandas as pd
from tqdm import tqdm


class Model:
    def __init__(self, n_choice, tau, mean = 0, mean_std=1, std_list=None, prior_std = None):

        self.n = n_choice
        self.mean_std = mean_std
        self.mean = mean
        self.std_list = std_list
        self.expected_mean = np.zeros(n_choice)
        self.expected_var = np.ones(n_choice) * mean_std * mean_std if prior_std is None else prior_std.copy()
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

    def play(self, total_epochs, v=0., ru=0., vtu=0., block_id=None, greedy_mode=None, greedy_ratio=0.1,optimal_choice=None):
        for i in range(total_epochs):
            expected_mean0 = self.expected_mean[0]
            expected_mean1 = self.expected_mean[1]
            expected_var0 = self.expected_var[0]
            expected_var1 = self.expected_var[1]
            V = self.expected_mean[0] - self.expected_mean[1]
            RU = np.sqrt(self.expected_var[0]) - np.sqrt(self.expected_var[1])
            VTU = (self.expected_mean[0] - self.expected_mean[1]) / np.sqrt(
                self.expected_var[0] + self.expected_var[1])
            f = v * V + ru * RU + vtu * VTU
            p = self.std_gaussian.cdf(f)
            sto = np.random.rand(1)[0]

            if sto > p:
                choice = 1
            else:
                choice = 0

            if greedy_mode == 'decaying_greedy':
                if sto < (total_epochs-i)/total_epochs/2:
                    choice = np.random.choice([0,1],size=1)[0]
            elif greedy_mode == 'greedy':
                if sto < greedy_ratio:
                    choice = np.random.choice([0, 1], size=1)[0]
            reward = self.choose(choice)

            is_optimal = True
            if choice != optimal_choice:
                is_optimal = False
            self.result.append([block_id+1, i + 1, self.mu[0], self.mu[1], self.std[0], self.std[1],expected_mean0,expected_mean1,expected_var0,expected_var1,choice, reward, V, RU, VTU,is_optimal])

    def get_mean_list(self):
        return self.game.get_mean()


class Play:
    def __init__(self, subject: int = 100, block: int = 10, trial: int = 10, save: bool = True, save_path: str = None,
                 save_mode: str = 'w',mean = 0, mean_std = 1, v=0,ru=0,vtu=0, std_list=None,tau_list=None,prior_std=None,greedy_mode=None, greedy_ratio=0.1):
        self.subject = subject
        self.block = block
        self.trial = trial
        self.optimal_probability = []
        self.result = []
        self.save = save
        self.save_path = save_path
        self.save_mode = save_mode
        self.v = v
        self.ru = ru
        self.vtu = vtu
        self.std_list = std_list
        self.tau_list = tau_list
        if self.tau_list is None :
            self.tau_list = np.array(std_list) ** 2 + np.ones(len(std_list)) * 1e-6
        self.mean = mean
        self.mean_std = mean_std
        self.prior_std = prior_std
        self.greedy = greedy_mode
        self.greedy_ratio = greedy_ratio

    def play(self):
        for subject in range(self.subject):
            block_optimal_times = 0
            for block in tqdm(range(self.block)):
                model = Model(n_choice=2, tau=self.tau_list, mean=self.mean, mean_std=self.mean_std, std_list=self.std_list, prior_std=self.prior_std)
                mean_list = model.get_mean_list()
                optimal = np.argmax(mean_list)
                model.play(self.trial, v=self.v, ru=self.ru, vtu=self.vtu, block_id=block, greedy_mode=self.greedy, greedy_ratio=self.greedy_ratio,optimal_choice=optimal)
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
        result = pd.DataFrame(data=result,
                              columns=['block', 'trial', 'mu1', 'mu2','std1', 'std2', 'Q1', 'Q2', 'var1','var2','choice', 'reward','V', 'RU', 'VTU','optimal'])
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
    p = stats.norm(0, 1).cdf(1/4*0.25)
    print(p)

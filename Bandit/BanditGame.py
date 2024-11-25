# @author : Chen
# @Start date : 2022/11/22
import sys
sys.path.append("..")
import numpy as np
import pandas as pd
from model.Lyapunov_Worm import Lyapunov_Worm_multi_D
from model.Lyapunov_Worm_deconstruction import Lyapunov_Worm_Deconstruction
from tqdm import tqdm

class Bandit:
    def __init__(self, mean: float = 0, std: float = 0):
        self.mean = mean
        self.std = std
        self.reward_history = []

    def choose(self):
        reward = np.random.normal(loc=self.mean, scale=self.std, size=1)[0]
        self.reward_history.append(reward)
        return reward


class Game:
    def __init__(self, n_bandits: int = 2, mean: float = 0, std_mean: float = 1, std_list: list = None):
        self.n_bandits = n_bandits
        self.mean_list = np.random.normal(loc=mean, scale=std_mean, size=self.n_bandits)
        self.std = std_list
        self.bandits = [Bandit(mean=self.mean_list[i], std=self.std[i]) for i in range(0, self.n_bandits)]

    def choose(self, index: int):
        return self.bandits[index].choose()

    def get_mean(self):
        return self.mean_list
    def get_std(self):
        return self.std


class Play:
    def __init__(self, varargin, force_times, bandit_std_list, net= None, prior = None, init = True, bandit_mean=0,mean_std=1, dim:int = 2, subject: int = 1, block: int = 10, trial: int = 10, save: bool = True, save_path: str = 'result',
                 save_mode: str = 'w'):
        '''

        :param subject: index
        :param block: number of blocks
        :param trial: number of trials
        :param save: whether to save
        :param save_path: save path
        :param save_mode: 'a' for 'add' , 'w' for rewrite
        :param  varargin


        '''

        self.subject = subject
        self.block = block
        self.trial = trial
        self.save = save
        self.save_path = save_path
        self.save_mode = save_mode
        self.dim = dim
        self.varargin = varargin
        self.force_times = force_times
        self.mean = bandit_mean
        self.mean_std = mean_std
        self.bandit_std_list = bandit_std_list
        self.prior = prior
        self.init = init
        self.net = net

    def one_game(self, dim: int, game: Game, block_index: int):
        if self.net is None:
            net = Lyapunov_Worm_multi_D(varargin=self.varargin, dim=self.dim)
        elif self.net == 'deconstruction':
            net = Lyapunov_Worm_Deconstruction(varargin=self.varargin, dim=self.dim)
        result_list = []
        mu = game.mean_list
        std = game.std
        if self.prior is not None:
            for (i,bandit) in enumerate(game.bandits):
                bandit.reward_history += self.prior[i]
        for i in range(self.trial):
            if i < dim * self.force_times:
                choice = i % dim
            else:
                reward_history = []
                for bandit in game.bandits:
                    reward_history.append(bandit.reward_history)
                reward_history = np.array(reward_history)
                choice = net.decide(history=reward_history,init=self.init)
            reward = game.choose(choice)
            result_list.append([self.subject,
                                block_index + 1,
                                i + 1,
                                *mu,
                                *std,
                                choice+1,
                                reward]
                               )

        return result_list

    def play(self):
        result = []
        for j in range(self.subject):
            for i in tqdm(range(self.block)):
                game = Game(n_bandits=self.dim, mean=self.mean, std_mean=self.mean_std, std_list=self.bandit_std_list)
                result_list = self.one_game(dim=self.dim,game=game, block_index=i)
                result.append(result_list)
            # print(result)
            result = np.array(result)
            result = result.reshape(self.block * self.trial, -1)
            mu_list = [f'mu{i }' for i in range(self.dim)]
            std_list = [f'std{i}' for i in range(self.dim)]
            result = pd.DataFrame(data=result,
                                  columns=['subject', 'block', 'trial', *mu_list, *std_list, 'choice', 'reward'])

            result[['subject', 'block', 'trial', 'choice']] = result[
                ['subject', 'block', 'trial', 'choice']].astype(int)
            result = result.round(4)
        if self.save:
            self.save_csv(result)
    def save_csv(self, result):

        if self.save_mode == 'a':
            header = False
        else:
            header = True
        result.to_csv(self.save_path, header=header, index=False, mode=self.save_mode)
        print("successfully save data to " + self.save_path)

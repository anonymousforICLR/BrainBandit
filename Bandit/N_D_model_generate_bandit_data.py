# @Start date : 2023/3/15
import numpy as np
import pandas as pd
import sys
sys.path.append("..")
from model.Lyapunov_Worm import Lyapunov_Worm_multi_D
from Expected_Value import CalculateMeanValue
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
    def __init__(self, varargin, force_times, bandit_std_list, prior = None, init = True, bandit_mean=0,mean_std=1, dim:int = 2, subject: int = 1, block: int = 10, trial: int = 10, save: bool = True, save_path: str = 'result',
                 save_mode: str = 'w'):
        '''

        :param subject: index
        :param block: number of blocks
        :param trial: number of trials
        :param save: whether to save
        :param save_path: save path
        :param save_mode: 'a' for 'add' , 'w' for rewrite
        :param  varargin
                varargin = {
            'step_num': 400,  # how many steps to run the brain circuit before executing the next movement

            # FROM NI'S MODEL
            'tau': np.ones(dim),  # decay time constant
            'weights_in': np.ones(dim) * 1.,  # input weights
            # self.weights_in = np.ones(2) * 2

            'rs': np.ones(dim) * .5,  #
            'w': np.ones(dim) * 4.,  # weight of mutual inhibition
            'k': 7 * np.ones(2),  # sigmoid center
            'n': 1. * np.ones(2),  # sigmoid slope
            'bi': np.ones(2) * 5.5,  # baseline production
            'dt': 0.5,  # size of timesteps
            'nsf': 0.8,  # noise level}
        }

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

    def one_game(self, dim: int, game: Game, block_index: int):

        net = Lyapunov_Worm_multi_D(varargin=self.varargin, dim=self.dim)
        result_list = []
        mu = game.mean_list
        std = game.std
        if self.prior is not None:
            for bandit in game.bandits:
                bandit.reward_history += self.prior
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
                                choice,
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
            mu_list = [f'mu{i + 1}' for i in range(self.dim)]
            std_list = [f'std{i + 1}' for i in range(self.dim)]
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


if __name__ == '__main__':
    dim = 2
    varargin = {
        'step': 1000,  # how many steps to run the brain circuit before executing the next movement
        'tau': np.ones(dim),  # decay time constant
        'weights_in': np.ones(dim) * 1.,  # input weights
        'rs': np.ones(dim) * 0.5,  #
        'w': np.ones(dim) * 4,  # weight of mutual inhibition
        'k': 7. * np.ones(dim),  # sigmoid center
        'n': 1.5 * np.ones(dim),  # sigmoid slope
        'bi': np.ones(dim) * 5.5,  # baseline production
        'dt': 0.5,  # size of timesteps
        'nsf': 0.1,  # noise level
    }
    bandit_std_list  = [2.6,.7]
    print(f'saved/analysed_{dim}D')
    p = Play(varargin=varargin, dim=dim, force_times=5, prior=None, init=True, bandit_mean=0, mean_std=0, bandit_std_list=bandit_std_list, subject=1, block=5000*dim, trial=10*dim, save=True, save_path=f'saved/raw_{dim}D_neutral_dimension', save_mode='w')
    p.play()
    k = CalculateMeanValue(f'saved/raw_{dim}D_neutral_dimension', f'saved/analysed_{dim}D_neutral_dimension', trial=10*dim, dim=dim)
    k.run()



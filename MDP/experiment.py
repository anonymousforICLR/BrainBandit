'''
Script to run simple tabular RL experiments.

author: iosband@stanford.edu
'''

import numpy as np
import pandas as pd

from shutil import copyfile

def run_finite_tabular_experiment(agent, env, f_ext, nEps, seed=1,
                    recFreq=100, fileFreq=1000, targetPath='tmp.csv', save=True):
    '''
    A simple script to run a finite tabular MDP experiment

    Args:
        agent - finite tabular agent
        env - finite TabularMDP
        f_ext - trivial FeatureTrueState
        nEps - number of episodes to run
        seed - numpy random seed
        recFreq - how many episodes between logging
        fileFreq - how many episodes between writing file
        targetPath - where to write the csv

    Returns:
        NULL - data is output to targetPath as csv file
    '''
    data = []
    qVals, qMax = env.compute_qVals()
    np.random.seed(seed)

    cumRegret = 0
    cumReward = 0
    empRegret = 0

    for ep in range(1, nEps + 1):
        # Reset the environment
        env.reset()
        epMaxVal = qMax[env.timestep][env.state]

        agent.update_policy(ep)

        epReward = 0
        epRegret = 0
        pContinue = 1

        while pContinue > 0:
            # Step through the episode

            h, oldState = f_ext.get_feat(env)
            # print(f'h: {h}')
            agent.count_state(oldState)
            action = agent.pick_action(oldState, h)
            epRegret += qVals[oldState, h].max() - qVals[oldState, h][action]

            reward, newState, pContinue = env.advance(action)
            epReward += reward

            agent.update_obs(oldState, action, reward, newState, pContinue, h)

        cumReward += epReward
        cumRegret += epRegret
        empRegret += (epMaxVal - epReward)

        # Variable granularity

        # Logging to dataframe

        data.append([ep, epReward, cumReward, cumRegret, empRegret])
            # print('episode:', ep, 'cumReward:', cumReward, 'cumRegret:', cumRegret, 'empRegret:', empRegret, 'epReward',epReward)
            # print(f'Q: {agent.qVals}')
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

    return cumRegret



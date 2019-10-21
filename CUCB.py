# coding=UTF-8
__author__ = 'sanglinwei'
'''
this code is following b
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_cucb(_knowledge, _user_prob, _event_num, _target):
    k = _knowledge[['index', 'sit1', 'UCB1', 'times', 'signal']]
    # plot the graph
    arm1 = list([])
    for j in range(_event_num):
        # CUCB
        # calculate UCB
        ucb1 = k['sit1'] + np.sqrt(alpha * np.log(j+1) / (2 * k['times']))
        k.loc[:, 'UCB1'] = ucb1
        # index
        # need to be revised
        k.sort_values(by=['UCB1'], ascending=False, inplace=True)
        m = k['sit1'].tolist()
        accumulate = list([])
        for i in range(user_num):
            accumulate.append(sum(m[0:i + 1]))

        # choose demand
        signal = list([x < _target for x in accumulate])
        k.loc[:, 'signal'] = signal
        # need to be revised
        k.sort_values(by=['index'], ascending=True, inplace=True)

        # deploy and update
        for i in range(user_num):
            if k.loc[i, 'signal']:
                # feedback
                _feedback = np.random.binomial(1, _user_prob['sit1'][i])
                # update
                k.loc[i, 'sit1'] = (k.loc[i, 'sit1'] * k.loc[i, 'times'] + _feedback) / (k.loc[i, 'times'] + 1)
                k.loc[i, 'times'] = k.loc[i, 'times'] + 1
        arm1.append(k.loc[1, 'sit1'])
    # update the knowledge
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(range(event_num), arm1)
    plt.show()
    _knowledge.loc[:, 'sit1'] = k['sit1']
    _knowledge.loc[:, 'times'] = k['times']

    return _knowledge


def run_concucb():


    return 0


def prob_generate(num):
    prob = 10*np.random.randn(num)+90
    prob[prob > 90] = np.random.randn(sum(prob > 90))*2+90
    return prob/100


user_num = 5
event_num = 200
alpha = 2  # UCB parameter

# initial configuration
# user's configuration
User_PROB = pd.DataFrame(columns=['index', "sit1", "sit2", "sit3"])
User_PROB["sit1"] = prob_generate(user_num)
User_PROB["sit2"] = prob_generate(user_num)
User_PROB["sit3"] = prob_generate(user_num)
User_PROB['index'] = range(user_num)

# power system command configuration in
target = 3  # fixed target

# demand aggregator configuration
KNOWLEDGE_INIT = pd.DataFrame(columns=['index', "sit1", 'UCB1', "sit2", 'UCB2', "sit3", 'UCB3',
                                       "power", "times", 'signal'])
KNOWLEDGE_INIT['sit1'] = np.zeros(user_num)+0.2
KNOWLEDGE_INIT['sit2'] = prob_generate(user_num)
KNOWLEDGE_INIT['sit3'] = prob_generate(user_num)
KNOWLEDGE_INIT['power'] = 200
KNOWLEDGE_INIT['times'] = 4
KNOWLEDGE_INIT['index'] = range(user_num)
KNOWLEDGE_INIT['signal'] = 0

knowledge = run_cucb(KNOWLEDGE_INIT, User_PROB, event_num, target)


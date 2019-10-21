# coding=UTF-8
__author__ = 'sanglinwei'
'''
this code is following b
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def run_cucb(_knowledge, _user_prob, _event_num, _target):
    k = _knowledge[['index', 'sit1', 'UCB1', 'time1', 'signal']]
    # plot the graph
    arm1 = list([])
    for j in range(_event_num):
        # CUCB
        # calculate UCB
        ucb1 = k['sit1'] + np.sqrt(alpha * np.log(j+1) / (2 * k['time1']))
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
                k.loc[i, 'sit1'] = (k.loc[i, 'sit1'] * k.loc[i, 'time1'] + _feedback) / (k.loc[i, 'time1'] + 1)
                k.loc[i, 'time1'] = k.loc[i, 'time1'] + 1
        arm1.append(k.loc[1, 'sit1'])
    # update the knowledge
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(range(event_num), arm1)
    plt.show()
    _knowledge.loc[:, 'sit1'] = k['sit1']
    _knowledge.loc[:, 'time1'] = k['time1']
    return _knowledge


def run_concucb(_kn, _user, _event, ):


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
                                       "power", "time1", 'time2', 'time3', 'signal'])
KNOWLEDGE_INIT['sit1'] = np.zeros(user_num)+0.2
KNOWLEDGE_INIT['sit2'] = np.zeros(user_num)+0.2
KNOWLEDGE_INIT['sit3'] = np.zeros(user_num)+0.2
KNOWLEDGE_INIT['power'] = 200
KNOWLEDGE_INIT['time1'] = 4
KNOWLEDGE_INIT['time2'] = 4
KNOWLEDGE_INIT['time3'] = 4
KNOWLEDGE_INIT['index'] = range(user_num)
KNOWLEDGE_INIT['signal'] = 0
knowledge = KNOWLEDGE_INIT

# knowledge = run_cucb(KNOWLEDGE_INIT, User_PROB, event_num, target)
k = knowledge

# generate random sequence
events = list(np.random.permutation(range(event_num)) % 3 + 1)

# events = [1, 3, 2, 3]
events_str = list(map(str, events))
num_sit = np.zeros(3)

for event_id in events_str:
    # different situation
    sit_choose = 'sit' + event_id
    time_choose = 'time' + event_id
    ucb_choose = 'UCB' + event_id
    # index
    ucb = k[sit_choose] + np.sqrt(alpha*np.log(num_sit[int(event_id)-1]+1)/(2*k[time_choose]))
    k.loc[:, ucb_choose] = ucb
    k.sort_values(by=[ucb_choose], ascending=False, inplace=True)
    m = k[sit_choose].tolist()
    accumulate = list([])
    for i in range(user_num):
        accumulate.append(sum(m[0:i+1]))
    signal = list([x < target for x in accumulate])
    k.sort_values(by=['index'], ascending=True, inplace=True)
    k.loc[:, 'signal'] = signal
    print(event_id)

    for i in range(user_num):
        if k.loc[i, 'signal']:
            # feedback
            feedback = np.random.binomial(1, User_PROB[sit_choose][i])
            # update
            k.loc[i, sit_choose] = (k.loc[i, sit_choose] * k.loc[i, time_choose] + feedback) / \
                                   (k.loc[i, time_choose] + 1)
            k.loc[i, time_choose] = k.loc[i, time_choose] + 1
    num_sit[int(event_id) - 1] = num_sit[int(event_id) - 1]+1
    # knowledge update
    knowledge.loc[:, sit_choose] = k[sit_choose]
    knowledge.loc[:, time_choose] = k[time_choose]








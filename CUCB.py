# coding=UTF-8
__author__ = 'sanglinwei'
'''
this code 
'''

import pandas as pd
import numpy as np


def run_ucb(knowledge, user_prob, _event_num, _target):
    k = knowledge.copy()
    # plot the graph
    arm1 = list([])
    for j in range(_event_num):
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
                _feedback = np.random.binomial(1, user_prob['sit1'][i])
                # update
                k.loc[i, 'sit1'] = (k.loc[i, 'sit1'] * k.loc[i, 'time1'] + _feedback) / (k.loc[i, 'time1'] + 1)
                k.loc[i, 'time1'] = k.loc[i, 'time1'] + 1
        arm1.append(k.loc[1, 'sit1'])

    return k


def run_ucb_in_contextual(knowledge, user_prob, _events, _target):
    k = knowledge.copy()

    events_str = list(map(str, _events))
    num = 1
    for event_id in events_str:
        # different situation
        sit_choose = 'sit' + event_id
        # index
        ucb = k['sit1'] + np.sqrt(alpha * np.log(num) / (2 * k['time1']))
        k.loc[:, 'UCB1'] = ucb
        k.sort_values(by=['UCB1'], ascending=False, inplace=True)
        m = k['sit1'].tolist()
        accumulate = list([])

        for i in range(user_num):
            accumulate.append(sum(m[0:i + 1]))
        signal = list([x < _target for x in accumulate])
        k.loc[:, 'signal'] = signal
        k.sort_values(by=['index'], ascending=True, inplace=True)

        for i in range(user_num):
            if k.loc[i, 'signal']:
                # feedback
                feedback = np.random.binomial(1, user_prob[sit_choose][i])
                # update
                k.loc[i, 'sit1'] = (k.loc[i, 'sit1'] * k.loc[i, 'time1'] + feedback) / (k.loc[i, 'time1'] + 1)
                k.loc[i, 'time1'] = k.loc[i, 'time1'] + 1
        num = num + 1

    return k


def run_contextual_ucb(knowledge, user_prob, _events, _target):
    k = knowledge.copy()

    events_str = list(map(str, _events))
    num_sit = np.zeros(3)

    for event_id in events_str:
        # different situation
        sit_choose = 'sit' + event_id
        time_choose = 'time' + event_id
        ucb_choose = 'UCB' + event_id
        # index
        ucb = k[sit_choose] + np.sqrt(alpha * np.log(num_sit[int(event_id) - 1] + 1) / (2 * k[time_choose]))
        k.loc[:, ucb_choose] = ucb
        k.sort_values(by=[ucb_choose], ascending=False, inplace=True)
        m = k[sit_choose].tolist()
        accumulate = list([])

        for i in range(user_num):
            accumulate.append(sum(m[0:i + 1]))
        signal = list([x < _target for x in accumulate])
        k.loc[:, 'signal'] = signal
        k.sort_values(by=['index'], ascending=True, inplace=True)

        for i in range(user_num):
            if k.loc[i, 'signal']:
                # feedback
                feedback = np.random.binomial(1, user_prob[sit_choose][i])
                # update
                k.loc[i, sit_choose] = (k.loc[i, sit_choose] * k.loc[i, time_choose] + feedback) / \
                                       (k.loc[i, time_choose] + 1)
                k.loc[i, time_choose] = k.loc[i, time_choose] + 1
        num_sit[int(event_id) - 1] = num_sit[int(event_id) - 1] + 1
    return k


# scale the probability
def prob_generate(num):
    prob = 10*np.random.randn(num)+90
    prob[prob > 90] = np.random.randn(sum(prob > 90))*2+90
    return prob/100


if __name__ == '__main__':
    # fundamental parameters
    user_num = 10  # the number of participated customers
    event_num = 200  # the number of demand response event
    alpha = 2  # UCB parameter

    # initial configuration
    # user's configuration
    User_PROB = pd.DataFrame(columns=['index', "sit1", "sit2", "sit3"])
    User_PROB["sit1"] = prob_generate(user_num)
    User_PROB["sit2"] = prob_generate(user_num)
    User_PROB["sit3"] = prob_generate(user_num)
    User_PROB['index'] = range(user_num)

    # power system command configuration in
    target = 3  # fixed target which can be time-varying

    # demand aggregator configuration
    KNOWLEDGE_INIT = pd.DataFrame(columns=['index', "sit1", 'UCB1', "sit2", 'UCB2', "sit3", 'UCB3',
                                           "power", "time1", 'time2', 'time3', 'signal'])
    KNOWLEDGE_INIT['sit1'] = np.zeros(user_num)+0.2
    KNOWLEDGE_INIT['sit2'] = np.zeros(user_num)+0.2
    KNOWLEDGE_INIT['sit3'] = np.zeros(user_num)+0.2
    KNOWLEDGE_INIT['power'] = 200
    KNOWLEDGE_INIT['time1'] = 1
    KNOWLEDGE_INIT['time2'] = 1
    KNOWLEDGE_INIT['time3'] = 1
    KNOWLEDGE_INIT['index'] = range(user_num)
    KNOWLEDGE_INIT['signal'] = 0
    k1 = KNOWLEDGE_INIT.copy()
    k2 = KNOWLEDGE_INIT.copy()
    k3 = KNOWLEDGE_INIT.copy()

    # generate random sequence
    events = list(np.random.permutation(range(event_num)) % 3 + 1)
    knowledge_common = run_ucb(k1, User_PROB, event_num, target)
    knowledge_common_in_contextual = run_ucb_in_contextual(k2, User_PROB, events, target)
    knowledge_contextual = run_contextual_ucb(k3, User_PROB, events, target)



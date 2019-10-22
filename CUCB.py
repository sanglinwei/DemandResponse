# coding=UTF-8
__author__ = 'sanglinwei'
'''
this code 
'''

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt


# just for testing the ubb
def run_ucb(knowledge, user_prob, _event_num, _target, _user_num):
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
        for i in range(_user_num):
            accumulate.append(sum(m[0:i + 1]))

        # choose demand
        signal = list([x < _target for x in accumulate])
        k.loc[:, 'signal'] = signal
        # need to be revised
        k.sort_values(by=['index'], ascending=True, inplace=True)

        # deploy and update
        for i in range(_user_num):
            if k.loc[i, 'signal']:
                # feedback
                _feedback = np.random.binomial(1, user_prob['sit1'][i])
                # update
                k.loc[i, 'sit1'] = (k.loc[i, 'sit1'] * k.loc[i, 'time1'] + _feedback) / (k.loc[i, 'time1'] + 1)
                k.loc[i, 'time1'] = k.loc[i, 'time1'] + 1
        arm1.append(k.loc[1, 'sit1'])

    return k


# run common ucb in contextual
def run_ucb_in_contextual(knowledge, user_prob, _events, _target, _user_num):
    k = knowledge.copy()
    events_str = list(map(str, _events))
    num = 1
    regret = list()
    for event_id in events_str:
        # different situation
        sit_choose = 'sit' + event_id
        # index
        ucb = k['sit1'] + np.sqrt(alpha * np.log(num) / (2 * k['time1']))
        k.loc[:, 'UCB1'] = ucb
        k.sort_values(by=['UCB1'], ascending=False, inplace=True)
        m = k['sit1'].tolist()
        accumulate = list([])

        for i in range(_user_num):
            accumulate.append(sum(m[0:i + 1]))
        signal = list([x < _target+0.5 for x in accumulate])
        k.loc[:, 'signal'] = signal
        k.sort_values(by=['index'], ascending=True, inplace=True)
        sum_feedback = 0
        for i in range(_user_num):
            if k.loc[i, 'signal']:
                # feedback
                feedback = np.random.binomial(1, user_prob[sit_choose][i])
                # update
                k.loc[i, 'sit1'] = (k.loc[i, 'sit1'] * k.loc[i, 'time1'] + feedback) / (k.loc[i, 'time1'] + 1)
                k.loc[i, 'time1'] = k.loc[i, 'time1'] + 1
                sum_feedback = sum_feedback + feedback
        regret.append(sum_feedback)
        num = num + 1
    return k, regret


# run contextual ucb
def run_contextual_ucb(knowledge, user_prob, _events, _target, _user_num):
    k = knowledge.copy()

    events_str = list(map(str, _events))
    num_sit = np.zeros(3)
    regret = list()
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

        for i in range(_user_num):
            accumulate.append(sum(m[0:i + 1]))

        # choose demand
        # discuss the target???????
        signal = list([x < _target for x in accumulate])
        k.loc[:, 'signal'] = signal
        k.sort_values(by=['index'], ascending=True, inplace=True)

        # deploy and update
        sum_feedback = 0
        for i in range(_user_num):
            if k.loc[i, 'signal']:
                # feedback
                feedback = np.random.binomial(1, user_prob[sit_choose][i])
                # update the probability
                k.loc[i, sit_choose] = (k.loc[i, sit_choose] * k.loc[i, time_choose] + feedback) / \
                                       (k.loc[i, time_choose] + 1)
                k.loc[i, time_choose] = k.loc[i, time_choose] + 1
                sum_feedback = sum_feedback + feedback
        regret.append(sum_feedback)
        num_sit[int(event_id) - 1] = num_sit[int(event_id) - 1] + 1
    return k, regret


# the oracle play results
def oracle_results(user_prob, _events, _target, _user_num):
    events_str = list(map(str, _events))
    regret = list([])
    k = user_prob.copy()
    for event_id in events_str:
        sit_choose = 'sit' + event_id
        k.sort_values(by=sit_choose, ascending=False, inplace=True)
        m = k[sit_choose].tolist()
        accumulate = list([])
        for i in range(_user_num):
            accumulate.append(sum(m[0:i+1]))

        # choose the demand
        signal = list([x < _target for x in accumulate])
        k['signal'] = signal
        k.sort_values(by='index', ascending=True, inplace=True)
        sum_feedback = 0

        # deploy demand
        for i in range(_user_num):
            if k.loc[i, 'signal']:
                feedback = np.random.binomial(1, user_prob[sit_choose][i])
                sum_feedback = sum_feedback+feedback
        regret.append(sum_feedback)
    return k, regret


# scale the probability
def prob_generate(num):
    prob = 30*np.random.randn(num)+70
    prob[prob > 70] = np.random.randn(sum(prob > 70))*2+70
    prob[prob > 100] = np.ones(sum(prob > 100))*100
    prob[prob < 0] = np.zeros(sum(prob < 0))
    return prob/100


# plot the results
def plot_results(ucb_res, contextual_res, oracle_res, _target):
    x = range(len(ucb_res))
    target_line = np.ones(len(ucb_res))*_target
    fig = plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, ucb_res, color='orange')
    ax.plot(x, contextual_res, color='blue')
    ax.plot(x, oracle_res, color='red')
    ax.plot(x, target_line, color='black', linestyle='-')
    ax.set_xlabel('round')
    ax.set_ylabel('reward')
    ax.axis('on')
    plt.legend(labels=['UCB', 'CUCB', 'Oracle', 'target'])
    plt.grid(True)
    plt.show()
    return 0


# plot regret
def plot_regret(ucb_res, contextual_res, oracle_res):
    regret1 = list()
    regret2 = list()
    accumulate_regret1 = list()
    accumulate_regret2 = list()
    for i in range(len(ucb_res)):
        regret1.append(np.square(ucb_res[i]-oracle_res[i]))
        regret2.append(np.square(contextual_res[i]-oracle_res[i]))
    for i in range(len(ucb_res)):
        accumulate_regret1.append(sum(regret1[0:i+1]))
        accumulate_regret2.append(sum(regret2[0:i+1]))
    rounds = range(len(ucb_res))
    fig = plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rounds, accumulate_regret1, color='orange', marker='*')
    ax.plot(rounds, accumulate_regret2, color='blue', marker='x')
    ax.set_xlabel('round')
    ax.set_ylabel('regret')
    ax.axis('on')
    plt.legend(labels=['UCB', 'CUCB'])
    plt.grid(True)
    plt.show()
    return 0


# plot the optimal choose
def plot_optimal_choose():
    pass


# fundamental parameter
alpha = 2  # UCB parameter


if __name__ == '__main__':
    start = time.clock()
    # fundamental parameters
    user_num = 10  # the number of participated customers
    event_num = 1000  # the number of demand response event

    # initial configuration
    # user's configuration
    User_PROB = pd.DataFrame(columns=['index', "sit1", "sit2", "sit3"])
    User_PROB["sit1"] = prob_generate(user_num)
    User_PROB["sit2"] = prob_generate(user_num)
    User_PROB["sit3"] = prob_generate(user_num)
    User_PROB['index'] = range(user_num)

    # power system command configuration in
    target = 5  # fixed target which can be time-varying

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
    # knowledge_common, results1 = run_ucb(k1, User_PROB, event_num, target, user_num)
    knowledge_common_in_contextual, results1 = run_ucb_in_contextual(k2, User_PROB, events, target, user_num)
    knowledge_contextual, results2 = run_contextual_ucb(k3, User_PROB, events, target, user_num)
    oracle_result = oracle_results(User_PROB, events, target, user_num)
    plot_results(results1, results2, oracle_result, target)
    plot_regret(results1, results2, oracle_result)
    elapsed = time.clock()-start
    print("time used", elapsed)




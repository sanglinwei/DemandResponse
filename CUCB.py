# coding=UTF-8
__author__ = 'sanglinwei'
'''
this code 
'''

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

mpl.style.use('default')


# run common ucb in contextual
def run_ucb_in_contextual(knowledge, user_prob, _truth, _events, _target, _user_num):

    # copy the user knowledge and oracle knowledge
    k = knowledge.copy()
    oracle_k = user_prob.copy()

    events_str = list(map(str, _events))
    num = 1
    results = list()
    regret = list()
    rate_choose = list()
    j = 0
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
            accumulate.append(sum(m[0:i+1]))

        # choose demand
        signal = list([x < _target+0.5 for x in accumulate])
        k.loc[:, 'signal'] = signal
        k.sort_values(by=['index'], ascending=True, inplace=True)

        # oracle index and generate signal
        oracle_k.sort_values(by=[sit_choose], ascending=False, inplace=True)
        oracle_m = oracle_k[sit_choose].tolist()
        oracle_accumulate = list()
        for i in range(_user_num):
            oracle_accumulate.append(sum(oracle_m[0:i+1]))
        oracle_signal = list([x < _target+0.5 for x in oracle_accumulate])
        oracle_k['signal'] = oracle_signal
        oracle_k.sort_values(by=['index'], ascending=True, inplace=True)

        # calculate the regret
        regret.append(abs(accumulate[sum(signal)-1]-oracle_accumulate[sum(oracle_signal)-1]))

        # calculate the rate
        same_choose = oracle_k['signal'] & k['signal']
        rate_choose.append(same_choose.sum() / k['signal'].sum())

        # deploy and update
        sum_feedback = 0

        for i in range(_user_num):
            if k.loc[i, 'signal']:

                # feedback
                feedback = _truth[i, j]

                # update
                k.loc[i, 'sit1'] = (k.loc[i, 'sit1'] * k.loc[i, 'time1'] + feedback) / ((k.loc[i, 'time1']) + 1)
                k.loc[i, 'time1'] = k.loc[i, 'time1'] + 1
                sum_feedback = sum_feedback + feedback

        j = j + 1
        results.append(sum_feedback)
        num = num + 1
    return k, results, rate_choose, regret


# run contextual ucb
def run_contextual_ucb(knowledge, user_prob, _truth, _events, _target, _user_num, _sit_num):

    # copy the user knowledge and oracle knowledge
    k = knowledge.copy()
    oracle_k = user_prob.copy()

    events_str = list(map(str, _events))
    num_sit = np.ones(_sit_num)
    results = list()
    regret = list()
    rate_choose = list()
    j = 0
    for event_id in events_str:

        # different situation
        sit_choose = 'sit' + event_id
        time_choose = 'time' + event_id
        ucb_choose = 'UCB' + event_id

        # index
        ucb = k[sit_choose] + np.sqrt(alpha * np.log(num_sit[int(event_id) - 1]) / (2 * k[time_choose]))
        k.loc[:, ucb_choose] = ucb
        k.sort_values(by=[ucb_choose], ascending=False, inplace=True)
        m = k[sit_choose].tolist()
        accumulate = list()

        for i in range(_user_num):
            accumulate.append(sum(m[0:i + 1]))

        # choose demand
        signal = list([x < _target+0.5 for x in accumulate])
        k.loc[:, 'signal'] = signal
        k.sort_values(by=['index'], ascending=True, inplace=True)

        # oracle index and generate signal
        oracle_k.sort_values(by=[sit_choose], ascending=False, inplace=True)
        oracle_m = oracle_k[sit_choose].tolist()
        oracle_accumulate = list()
        for i in range(_user_num):
            oracle_accumulate.append(sum(oracle_m[0:i+1]))
        oracle_signal = list([x < _target+0.5 for x in oracle_accumulate])
        oracle_k['signal'] = oracle_signal
        oracle_k.sort_values(by=['index'], ascending=True, inplace=True)

        # calculate the regret
        regret.append(abs(accumulate[sum(signal)-1]-oracle_accumulate[sum(oracle_signal)-1]))

        # calculate the rate
        same_choose = oracle_k['signal'] & k['signal']
        rate_choose.append(same_choose.sum()/k['signal'].sum())

        # deploy and update
        sum_feedback = 0

        for i in range(_user_num):
            if k.loc[i, 'signal']:

                # feedback
                feedback = _truth[i, j]

                # update the probability
                k.loc[i, sit_choose] = (k.loc[i, sit_choose] * k.loc[i, time_choose] + feedback) / \
                                       (k.loc[i, time_choose] + 1)
                k.loc[i, time_choose] = k.loc[i, time_choose] + 1
                sum_feedback = sum_feedback + feedback

        j = j + 1
        results.append(sum_feedback)
        num_sit[int(event_id) - 1] = num_sit[int(event_id) - 1] + 1
    return k, results, rate_choose, regret


# the oracle play results
def oracle_play(user_prob, _truth, _events, _target, _user_num):
    events_str = list(map(str, _events))
    results = list()
    k = user_prob.copy()
    j = 0
    for event_id in events_str:
        sit_choose = 'sit' + event_id
        k.sort_values(by=[sit_choose], ascending=False, inplace=True)
        m = k[sit_choose].tolist()
        accumulate = list()
        for i in range(_user_num):
            accumulate.append(sum(m[0:i+1]))

        # choose the demand
        signal = list([x < _target+0.5 for x in accumulate])
        k['signal'] = signal
        k.sort_values(by='index', ascending=True, inplace=True)
        sum_feedback = 0

        # deploy demand
        for i in range(_user_num):
            if k.loc[i, 'signal']:
                feedback = _truth[i, j]
                sum_feedback = sum_feedback+feedback
        j = j + 1
        results.append(sum_feedback)
    return k, results


# scale the probability
def prob_generate(num):
    prob = 50*np.random.randn(num)+50
    # prob[prob > 70] = np.random.randn(sum(prob > 70))*25+50
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
    ax.plot(x, ucb_res)
    ax.plot(x, contextual_res)
    ax.plot(x, oracle_res)
    ax.plot(x, target_line)
    ax.set_xlabel('round')
    ax.set_ylabel('reward')
    ax.axis('on')
    plt.tight_layout()
    plt.legend(labels=['CUCB', 'ConCUCB', 'offline', 'target'])
    plt.grid(True)
    plt.show()
    return 0


# define ground truth
def run_truth(_events, user_prob, _user_num, _event_num):
    events_str = list(map(str, _events))
    truth_feedback = np.zeros([_user_num, _event_num])
    j = 0
    for event_id in events_str:
        sit_choose = 'sit' + event_id
        for i in range(_user_num):
            truth_feedback[i, j] = np.random.binomial(1, user_prob[sit_choose][i])
        j = j + 1
    return truth_feedback


# plot mismatch
def plot_mismatch(ucb_res, contextual_res, oracle_res, _target):
    ucb_mismatch = list()
    contextual_mismatch = list()
    oracle_mismatch = list()
    num = len(ucb_res)
    for i in range(num):
        ucb_mismatch.append(np.square(ucb_res[i]-_target))
        contextual_mismatch.append(np.square(contextual_res[i]-_target))
        oracle_mismatch.append(np.square(oracle_res[i]-_target))
    rounds = range(num)
    plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    plt.subplot(111)
    plt.scatter(rounds, ucb_mismatch, marker='o')
    plt.scatter(rounds, contextual_mismatch, marker='x')
    plt.scatter(rounds, oracle_mismatch, marker='+')
    plt.xlabel('round')
    plt.ylabel('mismatch')
    plt.axis('on')
    plt.tight_layout()
    plt.legend(labels=['CUCB', 'ConCUCB', 'oracle'])
    plt.grid(True)

    plt.savefig('mismatch.svg')
    plt.savefig('mismatch.pdf')

    plt.show()
    return 0


# plot regret
def plot_true_regret(reg1, reg2, _events):
    rounds = range(len(reg1))
    fig = plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax = fig.add_subplot(1, 1, 1)
    seq = 0
    for event in _events:
        if event == 1:
            ax.scatter(rounds[seq], reg1[seq], color='orange', marker='1')
            ax.scatter(rounds[seq], reg2[seq], color='blue', marker='1')
        if event == 2:
            ax.scatter(rounds[seq], reg1[seq], color='orange', marker='+')
            ax.scatter(rounds[seq], reg2[seq], color='blue', marker='+')
        if event == 3:
            ax.scatter(rounds[seq], reg1[seq], color='orange', marker='*')
            ax.scatter(rounds[seq], reg2[seq], color='blue', marker='*')
        if event == 4:
            ax.scatter(rounds[seq], reg1[seq], color='orange', marker='o')
            ax.scatter(rounds[seq], reg2[seq], color='blue', marker='o')
        seq = seq + 1
    ax.set_xlabel('round')
    ax.set_ylabel('instantaneous regret')
    ax.axis('on')
    plt.tight_layout()
    plt.legend(labels=['CUCB', 'ConCUCB'])
    plt.grid(True)

    plt.savefig('instantaneous_regret.svg')
    plt.savefig('instantaneous_regret.pdf')
    plt.show()
    return 0


# plot accumulate regret
def plot_acc_regret(reg1, reg2):
    acc_reg1 = list()
    acc_reg2 = list()
    rounds = range(len(reg1))
    for i in rounds:
        acc_reg1.append(sum(reg1[0:i+1]))
        acc_reg2.append(sum(reg2[0:i+1]))

    fig = plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rounds, acc_reg1, label='CUCB')
    ax.plot(rounds, acc_reg2, label='ConCUCB')
    ax.set_xlabel('round')
    ax.set_ylabel('accumulate regret')
    ax.axis('on')
    plt.legend(labels=['UCB', 'CUCB'])
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.savefig('acc_regret.svg')
    plt.savefig('acc_regret.pdf')
    plt.show()
    return 0


# plot accumulate bias
def plot_bias(ucb_res, contextual_res, oracle_res):
    _bias1 = list()
    _bias2 = list()
    accumulate_bias1 = list()
    accumulate_bias2 = list()
    for i in range(len(ucb_res)):
        _bias1.append(np.square(ucb_res[i]-oracle_res[i]))
        _bias2.append(np.square(contextual_res[i]-oracle_res[i]))
    for i in range(len(ucb_res)):
        # sum bias
        accumulate_bias1.append(sum(_bias1[0:i+1]))
        accumulate_bias2.append(sum(_bias2[0:i+1]))
    rounds = range(len(ucb_res))
    fig = plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(rounds, accumulate_bias1, color='orange', marker='*')
    ax.plot(rounds, accumulate_bias2, color='blue', marker='x')
    ax.set_xlabel('round')
    ax.set_ylabel('sum_difference')
    ax.axis('on')
    plt.tight_layout()
    plt.legend(labels=['UCB', 'CUCB'])
    plt.grid(True)
    plt.show()
    return 0


# plot the optimal choose
def plot_optimal_choose(_rate1, _rate2):
    rounds = range(len(_rate1))
    fig = plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(rounds, _rate1, color='orange', marker='*', label='UCB')
    ax.scatter(rounds, _rate2, color='blue', marker='x', label='CUCB')
    ax.set_xlabel('round')
    ax.set_ylabel('right choose rate')
    ax.axis('on')
    plt.tight_layout()
    plt.legend()
    plt.show()


def plot_user_profile(user_prob):

    fig = plt.figure()
    plt.rc('font', family='Times New Roman', style='normal', size=13)
    plt.subplots_adjust()
    sns.set()
    ax = sns.heatmap(user_prob[['sit1', 'sit2', 'sit3']], annot=True, cmap="YlGnBu")
    plt.tight_layout()
    plt.legend()
    plt.savefig('user_heatmap.svg')
    plt.savefig('user_heatmap.pdf')
    plt.show()
    return 0


# fundamental parameter
alpha = 2  # UCB confidence parameter


if __name__ == '__main__':
    start = time.clock()
    # fundamental parameters
    user_num = 20  # the number of participated customers
    event_num = 300  # the number of demand response event
    sit_num = 3  # the number of situations

    # power system command configuration
    # according to the user expectation to choose the target
    target = 5  # fixed target which can be time-varying

    # initial configuration
    # user's configuration
    User_PROB = pd.DataFrame(columns=['index', "sit1", "sit2", "sit3"])
    User_PROB['index'] = range(user_num)

    # demand aggregator configuration
    KNOWLEDGE_INIT = pd.DataFrame(columns=['index', "sit1", 'UCB1', "sit2", 'UCB2', "sit3", 'UCB3',
                                           "power", "time1", 'time2', 'time3', 'signal'])
    KNOWLEDGE_INIT['power'] = 600
    KNOWLEDGE_INIT['index'] = range(user_num)
    KNOWLEDGE_INIT['signal'] = 0

    for i_e in range(sit_num):
        User_PROB['sit'+str(i_e+1)] = prob_generate(user_num)
        KNOWLEDGE_INIT['sit'+str(i_e+1)] = np.zeros(user_num)+0.5
        KNOWLEDGE_INIT['time'+str(i_e+1)] = 1

    # generate random sequence
    events = list(np.random.permutation(range(event_num)) % sit_num + 1)

    User_Expectation = User_PROB.apply(np.sum, axis=0)

    # play the truth
    truth = run_truth(events, User_PROB, user_num, event_num)

    # play the common UCB in multi-contexts based on the truth
    knowledge_common_in_contextual, results1, rate1, regret1 = run_ucb_in_contextual(KNOWLEDGE_INIT, User_PROB, truth,
                                                                                     events, target, user_num)
    # play the contextual-UCB algorithm based on the truth
    knowledge_contextual, results2, rate2, regret2 = run_contextual_ucb(KNOWLEDGE_INIT, User_PROB, truth, events,
                                                                        target, user_num, sit_num)
    # play the oracle choose based on the truth
    oracle, oracle_results = oracle_play(User_PROB, truth, events, target, user_num)
    plot_results(results1, results2, oracle_results, target)
    plot_mismatch(results1, results2, oracle_results, target)
    plot_acc_regret(regret1, regret2)
    plot_true_regret(regret1, regret2, events)
    plot_user_profile(User_PROB)
    # plot_bias(results1, results2, oracle_results)
    elapsed = time.clock()-start
    print("time used", elapsed)


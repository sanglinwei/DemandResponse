import pandas as pd
import numpy as np


def run_cucb(knowledge, target):
    return 0


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
                                       "power", "times", 'signal', 'feedback'])
KNOWLEDGE_INIT['sit1'] = np.zeros(user_num)+0.2
KNOWLEDGE_INIT['sit2'] = prob_generate(user_num)
KNOWLEDGE_INIT['sit3'] = prob_generate(user_num)
KNOWLEDGE_INIT['power'] = 200
KNOWLEDGE_INIT['times'] = 4
KNOWLEDGE_INIT['index'] = range(user_num)
KNOWLEDGE_INIT['signal'] = 0
KNOWLEDGE_INIT['feedback'] = np.zeros(user_num)
knowledge = KNOWLEDGE_INIT

k = knowledge[['index', 'sit1', 'UCB1', 'times', 'signal', 'feedback']]
for j in range(event_num):
    # contextual CUCB
    # calculate UCB
    UCB1 = k['sit1']+np.sqrt(alpha*np.log(j)/(2*k['times']))
    k.loc[:, 'UCB1'] = UCB1
    # index
    k.sort_values(by=['UCB1'], ascending=False, inplace=True)
    m = k['sit1'].tolist()
    accumulate = list([])
    for i in range(user_num):
        accumulate.append(sum(m[0:i+1]))
    # choose demand
    signal = list([x < target for x in accumulate])
    k.loc[:, 'signal'] = signal
    k.sort_values(by=['index'], ascending=True, inplace=True)

    # deploy and update
    for i in range(user_num):
        if k.loc[i, 'signal']:
            # feedback
            feedback = np.random.binomial(1, User_PROB['sit1'][i])
            # update
            k.loc[i, 'sit1'] = (k.loc[i, 'sit1']*k.loc[i, 'times']+feedback)/(k.loc[i, 'times']+1)
            k.loc[i, 'times'] = k.loc[i, 'times']+1

# update the knowledge
knowledge.loc[:, 'sit1'] = k['sit1']
knowledge.loc[:, 'times'] = k['times']



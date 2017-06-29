import nltk
from nltk.corpus import wordnet as wn
from math import *
import random
import numpy as np
STABILITY = 0.00001
network = {}

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
tot = 3
def newline(p1, p2):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()
    if(p2[0] == p1[0]):
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmax-p1[0])
        ymin = p1[1]+(p2[1]-p1[1])/(p2[0]-p1[0])*(xmin-p1[0])
    l = mlines.Line2D([xmin,xmax], [ymin,ymax])
    ax.add_line(l)
    return l

def plotnow(pos1, pos2, negfp):
    import matplotlib.pyplot as plt
    plt.plot(emb[pos1][0], emb[pos1][1], marker='o', color = [0,1,0], ls='')
    plt.plot(emb[pos2][0], emb[pos2][1], marker='o', color = [0,0,1], ls='')
    for a in negfp:
        plt.plot(emb[a][0], emb[a][1], marker='o', color = [0.5,0.5,0.5], ls='')
    lim = 0.5
    plt.ylim([-1*lim, lim])
    plt.xlim([-1*lim, lim])
    plt.show()  


def plotall(ii="def"):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for a in emb:
        plt.plot(emb[a][0], emb[a][1], marker = 'o', color = [levelOfNode[a]/(tot+1),levelOfNode[a]/(tot+1),levelOfNode[a]/(tot+1)])
    for a in network:
        for b in network[a]:
            plt.plot([emb[a][0], emb[b][0]], [emb[a][1], emb[b][1]], color = [levelOfNode[a]/(tot+1),levelOfNode[a]/(tot+1),levelOfNode[a]/(tot+1)])
    plt.show()
    # fig.savefig(str(ii) + '.png', dpi=fig.dpi)

levelOfNode = {}
def get_hyponyms(synset, level):
    if (level == tot):
        levelOfNode[str(synset)] = level
        return
    if not str(synset) in network:
        network[str(synset)] = [str(s) for s in synset.hyponyms()]
        levelOfNode[str(synset)] = level
    for hyponym in synset.hyponyms():
        get_hyponyms(hyponym, level + 1)

mammal = wn.synset('mammal.n.01')
get_hyponyms(mammal, 0)
levelOfNode[str(mammal)] = 0

emb = {}

for a in network:
    for b in network[a]:
        emb[b] = np.random.uniform(low=-0.001, high=0.001, size=(2,))
    emb[a] = np.random.uniform(low=-0.001, high=0.001, size=(2,))

vocab = list(emb.keys())
random.shuffle(vocab)

for a in emb:
    if not a in network:
        network[a] = []

### Now the tough part


def partial_der(theta, x, gamma): #eqn4
    alpha = (1.0-np.dot(theta, theta))
    norm_x = np.dot(x, x)
    beta = (1-norm_x)
    gamma = gamma
    return 4.0/(beta * sqrt(gamma*gamma - 1) + STABILITY)*((norm_x- 2*np.dot(theta, x)+1)/(pow(alpha,2)+STABILITY)*theta - x/(alpha + STABILITY))

lr = 0.1
def update(emb, error_): #eqn5
    global lr
    try:
        # update =  lr*pow((1 - np.dot(emb,emb)), 2)*error_/4
        # if (np.dot(update, update) >= 0.01):
        #     update = 0.1*update/sqrt(np.dot(update, update))
        # # print (update)
        emb = emb + error_*lr
        if (np.dot(emb, emb) >= 1):
            emb = emb/sqrt(np.dot(emb, emb)) - 0.1
        return emb
    except Exception as e:
        print (e)
        temp = input()

def dist(vec1, vec2): # eqn1
    return 1 + 2*np.dot(vec1 - vec2, vec1 - vec2)/ \
             ((1-np.dot(vec1, vec1))*(1-np.dot(vec2, vec2)) + STABILITY)



J = 2

def calc_dist_safe(v1, v2):
    tmp = dist(v1, v2)
    if (tmp > 700 or tmp < -700):
        tmp = 700
    return cosh(tmp)

j = 0
pre_emb = emb.copy()
# plotall("init")

running_mean = [1.0, 1.0, 1.0, 1.0, 1.0]


import autograd.numpy as np
from autograd import grad
from math import *

def act_dist(vec1, vec2): # eqn1
    return np.cosh(1 + 2*np.dot(vec1 - vec2, vec1 - vec2)/ \
             ((1-np.dot(vec1, vec1))*(1-np.dot(vec2, vec2)) + STABILITY))


def loss(x, a, b):
    return np.log(np.exp(-1*x) / (np.exp(-1*a) + np.exp(-1*b) ))

grad_0 = grad(loss, 0)
grad_1 = grad(loss, 1)
grad_2 = grad(loss, 2)


pre_emb = emb.copy()
lr = 0.05
# for ii in range(30):
#     # tmp = input()
#     print ("epoch", ii)
#     for k in range(20):
for pos1 in vocab:
    if not network[pos1]:
        continue
    pos2 = random.choice(network[pos1])
    # print ("--------------START-------")
    negs = []
    while (len(negs) < J):
        neg = random.choice(vocab)
        if not (neg in network[pos1] or pos1 in network[neg] or neg == pos1):
            negs.append(neg1)
    dist_pos = act_dist(emb[pos1], emb[pos2])
    dist_negs = []
    for a in negs:
        dist_negs.append(act_dist(emb[pos1], emb[a]))
    print ("loss", loss(dist_pos, dist_negs[0], dist_negs[1]))
    emb[pos1] = update(emb[pos1], grad_0(emb[pos1], emb[pos2], emb[negs[0]], emb[negs[1]]))
    emb[pos2] = update(emb[pos2], grad_1(emb[pos1], emb[pos2], emb[negs[0]], emb[negs[1]]))
    emb[negs[0]] = update(emb[negs[0]], grad_2(emb[pos1], emb[pos2], emb[negs[0]], emb[negs[1]]))
    emb[negs[1]] = update(emb[negs[1]], grad_3(emb[pos1], emb[pos2], emb[negs[0]], emb[negs[1]]))
    print ("--------------------------------------------new loss", loss(emb[pos1], emb[pos2], emb[negs[0]], emb[negs[1]]))


            # print ("dist_p_final", calc_dist_safe(emb[pos1], emb[pos2]))
            # for a in negs:
            #     print ("dist_neg", calc_dist_safe(emb[a[0]], emb[a[1]]))
            # print ("-----------END------")
            # plotnow(pos1, pos2, neg_for_plot)
            # print (i, j, loss, pos1, pos2)
        # print (loss_hist)
    # pre_emb = emb
    # if ((ii+1) % 5 == 0):
    #     lr /= 2

# plotall()

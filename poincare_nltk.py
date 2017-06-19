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


def plotall(ii):
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

lr = 0.01
def update(emb, error_): #eqn5
    try:
        update =  lr*pow((1 - np.dot(emb,emb)), 2)*error_/4
        # print (update)
        emb = emb - update
        if (np.dot(emb, emb) >= 1):
            emb = emb/sqrt(np.dot(emb, emb)) - 0.00001
        return emb
    except Exception as e:
        print (e)
        temp = input()

def dist(vec1, vec2): # eqn1
    return 1 + 2*np.dot(vec1 - vec2, vec1 - vec2)/ \
             ((1-np.dot(vec1, vec1))*(1-np.dot(vec2, vec2)) + STABILITY)



J = 5

def calc_dist_safe(v1, v2):
    tmp = dist(v1, v2)
    if (tmp > 700 or tmp < -700):
        tmp = 700
    return cosh(tmp)

j = 0
pre_emb = emb.copy()
plotall("init")

running_mean = [1.0, 1.0, 1.0, 1.0, 1.0]

for ii in range(1):
    # tmp = input()
    print (ii)
    for k in range(20):
        for pos1 in vocab:
            j += 1
            if not network[pos1]:
                continue
            pos2 = random.choice(network[pos1])
            i = random.choice([0,1])
            # print ("--------------START-------")
            poss = []
            negs = []
            dist_p_init = dist(emb[pos1], emb[pos2])
            if (dist_p_init > 700): # cant exclude it their distance should be less
                print ("got one very high")
                dist_p_init = 700
            if (dist_p_init < -700): # cant exclude it their distance should be less
                dist_p_init = -700
            dist_p = cosh(dist_p_init) # this is +ve

            print ("dist_p_init", dist_p)
            dist_negs_init = []
            dist_negs = []
            neg_for_plot = []
            while (len(negs) < J):
                neg1 = pos1
                neg2 = random.choice(vocab)
                if not (neg2 in network[neg1] or neg1 in network[neg2]):
                    dist_neg_init = dist(emb[neg1], emb[neg2])
                    if (dist_neg_init > 700 or dist_neg_init < -700): #already dist is good, leave it
                        continue
                    negs.append([neg1, neg2])
                    dist_neg = cosh(dist_neg_init)
                    # print ("dist_neg", dist_neg)
                    dist_negs_init.append(dist_neg_init)
                    dist_negs.append(dist_neg)
                    neg_for_plot.append(neg2)
            # plotnow(pos1, pos2, neg_for_plot)
            loss_den = 0.0
            for dist_neg in dist_negs:
                loss_den += exp(-1*dist_neg)
            loss = -1*dist_p - log(loss_den + STABILITY)
            der_p = -1 + exp(-1*dist_p)/(loss_den + STABILITY)
            der_negs = []
            for dist_neg in dist_negs:
                der_negs.append(exp(-1*dist_neg)/(loss_den + STABILITY))
            der_p_emb0 = der_p * partial_der(emb[pos1], emb[pos2], dist_p_init)
            der_p_emb1 = der_p * partial_der(emb[pos2], emb[pos1], dist_p_init)
            der_negs_final = []
            for (der_neg, neg, dist_neg_init) in zip(der_negs, negs, dist_negs_init):
                der_neg1 = der_neg * partial_der(emb[neg[1]], emb[neg[0]], dist_neg_init)
                der_neg0 = der_neg * partial_der(emb[neg[0]], emb[neg[1]], dist_neg_init)
                der_negs_final.append([der_neg0, der_neg1])
            emb[pos1] = update(emb[pos1], -1*der_p_emb0)
            emb[pos2] = update(emb[pos2], -1*der_p_emb1)
            for (neg, der_neg) in zip(negs, der_negs_final):
                # emb[neg[0]] = update(emb[neg[0]], -1*der_neg[0])
                emb[neg[1]] = update(emb[neg[1]], -1*der_neg[1])
            loss_hist = loss
            # print ("dist_p_final", calc_dist_safe(emb[pos1], emb[pos2]))
            # for a in negs:
            #     print ("dist_neg", calc_dist_safe(emb[a[0]], emb[a[1]]))
            # print ("-----------END------")
            # plotnow(pos1, pos2, neg_for_plot)
            # print (i, j, loss, pos1, pos2)
        # print (loss_hist)
    # pre_emb = emb
    plotall(ii)


pre_emb = emb.copy()

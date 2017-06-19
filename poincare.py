from math import *
import random
vocab = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
emb = {}
emb['a'] = np.array([0.1, 0.1])
emb['b'] = np.array([-0.1, 0.1])
emb['c'] = np.array([0.1, -0.1])
emb['d'] = np.array([-0.1, -0.1])
emb['e'] = np.array([0.1, 0])
emb['f'] = np.array([0, 0.1])
emb['g'] = np.array([0, -0.1])
emb['h'] = np.array([-0.1, 0])

network = [['a', 'b'], ['a', 'c'], ['b', 'd'], ['b', 'e'], ['c', 'f'], ['f', 'g'], ['f', 'h']]

def partial_der(theta, x, gamma):
    alpha = (1.0-np.dot(theta, theta))
    norm_x = np.dot(x, x)
    beta = (1-norm_x)
    gamma = gamma
    return 4.0/(beta * sqrt(gamma*gamma - 1))*((norm_x- 2*np.dot(theta, x)+1)/pow(alpha,2)*theta - x/alpha)

lr = 0.1
def update(emb, error_):
    emb = emb - lr*pow((1 - np.dot(emb,emb)), 2)*error_/4
    if (np.dot(emb, emb) >= 1):
        emb = emb/sqrt(np.dot(emb, emb)) - 0.00001
    return emb

def dist(vec1, vec2):
    return 1 + 2*np.dot(vec1 - vec2, vec1 - vec2)/ \
             ((1-np.dot(vec1, vec1))*(1-np.dot(vec2, vec2)))

J = 2
while(1):
    for a in network:
        negs = []
        dist_p_init = dist(emb[a[0]], emb[a[1]])
        dist_p = cosh(dist_p_init)
        while (len(negs) < J):
            neg = a[1]
            while ([a[0], neg] in network or [neg, a[0]] in network or neg == a[0] or neg in negs):
                neg = random.choice(vocab)
            negs.append(neg)
        print(a[0], '+', a[1], '-', negs[0], '-', negs[1])
        dist_negs_init = []
        dist_negs = []
        for neg in negs:
            dist_neg_init = dist(emb[a[0]], emb[neg])
            dist_neg = cosh(dist_neg_init)
            dist_negs_init.append(dist_neg_init)
            dist_negs.append(dist_neg)
        loss_den = 0.0
        for dist_neg in dist_negs:
            loss_den += exp(-1*dist_neg)
        loss = -1*dist_p - log(loss_den)
        der_p = -1
        der_negs = []
        for dist_neg in dist_negs:
            der_negs.append(1/loss_den*exp(-1*dist_neg))
        try:
            der_p_emb0 = der_p * partial_der(emb[a[0]], emb[a[1]], dist_p_init)
            der_p_emb1 = der_p * partial_der(emb[a[1]], emb[a[0]], dist_p_init)
            der_neg1_emb0 = der_negs[0] * partial_der(emb[a[0]], emb[negs[0]], dist_negs_init[0])
            der_neg1_neg1 = der_negs[0] * partial_der(emb[negs[0]], emb[a[0]], dist_negs_init[0])
            der_neg2_emb0 = der_negs[1] * partial_der(emb[a[0]], emb[negs[1]], dist_negs_init[1])
            der_neg2_neg2 = der_negs[1] * partial_der(emb[negs[1]], emb[a[0]], dist_negs_init[1])
            emb[a[0]] = update(emb[a[0]], der_p_emb0 + der_neg1_emb0 + der_neg2_emb0)
            emb[a[1]] = update(emb[a[1]], der_p_emb1)
            emb[neg1] = update(emb[neg1], der_neg1_neg1)
            emb[neg2] = update(emb[neg2], der_neg2_neg2)
        except Exception as e:
            print (e)
            continue
        print (loss)
    lis = []
    for a in emb:
        lis.append(emb[a])
    import matplotlib.pyplot as plt
    plt.plot(*zip(*lis), marker='o', color='r', ls='')
    plt.show()  
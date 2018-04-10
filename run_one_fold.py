"""
3/28/2018:
This is the script of directly learning from regions.
"""
import sys
import math
import json
import random
from util import read
from util import calc_KLD
import numpy as np
from sklearn.metrics import roc_auc_score
from classes import Sampling
from classes import Subspace
from scipy.stats import binom_test

def train_and_test(new_regions, model, test, auc, acc):
    """Do the regular training and testing."""
    def accuracy(probs, test):
        """Warning: just use bias = .5!!!"""
        predicted = [1 if pi > .5 else -1 for pi in probs]
        correct = 0
        for pred, true in zip(predicted, test[1]):
            if pred*true > 0:
                correct += 1
        return float(correct)/len(test[0])

    sys.stdout.write(str(len(auc)) + '.')
    sys.stdout.flush()
    model.fit(new_regions)

    probs = model.pred_prob(test[0])
    auc[-1] = roc_auc_score(test[1], probs)
    acc[-1] = accuracy(probs, test)

def select(partition, model, train):
    """Select a group to split from current partition."""
    candidates = [g for g in partition if g.all_same is False]
    if len(candidates) == 0: return None
    candidates.sort(key=lambda c:c.err, reverse=True)
    #del candidates[5:]
    return candidates[0]


def test_competition(compt):
    """Test if supervised heuristic is doing
    significantly better or not.
    
    Perform multiple testing of different windows
    and adjust by Bonferroni procedure."""
    win_sizes = [10,15]
    alpha = 0.05
    if len(compt) < max(win_sizes):
        return False
    for ws in win_sizes:
        bstat = sum(compt[-ws:])
        pval = binom_test(bstat,ws,0.5,alternative='greater')
        if pval >= alpha/len(win_sizes):
            return False
    return True


def run(fold, end):
    train = read(fold+'train')
    test = read(fold+'test')
    model = Sampling(train[0])
    partition = [Subspace(train[0])]
    frr = list()
    for g in partition:
        g.query(train[1])
        frr.append(Subspace.frr(g.region))
    auc, acc = [.5]*len(partition), [.5]*len(partition)
    train_and_test(partition, model, test, auc, acc)
    compt = list() #history of unsup and sup competition
    sup_only = False #whether to use sup split only

    while len(auc) < end:
        #print 'Competition:', compt[-15:]
        #print 'Sup only?', sup_only
        target = select(partition, model, train)
        if target is None:
            break
        if sup_only:
            sup_d, sup_v = target.test_split(train[0],model)
            children,_ = target.split(sup_d,sup_v,train)
            num_queries = 1
        else:
            sup_d, sup_v = target.test_split(train[0],model)
            unsup_d, unsup_v = target.test_split(train[0])
            if sup_d == unsup_d and abs(sup_v-unsup_v) < 1e-5:
                children,_ = target.split(sup_d,sup_v,train)
                num_queries = 1
            else:
                sup_children, sup_gini = target.split(sup_d,sup_v,train)
                unsup_children, unsup_gini = target.split(unsup_d,unsup_v,train)
                num_queries = 2
                #print 'Parent:', len(target), target.label
                #print 'Sup left:', len(sup_children[0]), sup_children[0].label,
                #print 'Sup right:', len(sup_children[1]), sup_children[1].label, sup_gini
                #print 'Unsup left:', len(unsup_children[0]), unsup_children[0].label,
                #print 'Unsup right:', len(unsup_children[1]), unsup_children[1].label, unsup_gini
                if sup_gini <= unsup_gini:
                    compt.append(1)
                    children = sup_children
                else:
                    compt.append(0)
                    children = unsup_children
                sup_only = test_competition(compt)

        for child in children:
            frr.append(Subspace.frr(child.region))
        auc.extend([auc[-1]]*num_queries)
        acc.extend([acc[-1]]*num_queries)
        partition.remove(target)
        partition.extend(children)
        train_and_test(children, model, test, auc, acc)


    with open(fold + 'frr', 'wb') as f:
        f.write(json.dumps(frr))
    with open(fold + 'auc', 'wb') as f:
        f.write(json.dumps(auc))
    with open(fold + 'acc', 'wb') as f:
        f.write(json.dumps(acc))

if __name__ == '__main__':
    fold = sys.argv[1]
    end = int(sys.argv[2])
    run(fold, end)
    print 'Done(' + fold + ')'

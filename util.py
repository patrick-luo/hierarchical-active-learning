import numpy as np
from math import log
import math
import sys
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from time import time
import json

def log2(x):
    return log(x) / log(2.0)

def entropy(p):
    if p < 1e-5 or p > 1-1e-5: return 0.0
    else: return -p*log2(p) - (1-p)*log2(1-p)

def gini(p):
    if p < 1e-5 or p > 1 - 1e-5:
        return 0.0
    return 1 - (p**2 + (1-p)**2)

def mean(data):
    return sum(data) / float(len(data))

def variance(data, avg=None):
    if len(data) <= 1: return 0.0
    if avg == None: avg = mean(data)
    return sum((d-avg)**2 for d in data)\
            / float(len(data)-1)

def read(file):
    def decompose(line):
        """Each line is in liblinear format."""
        attrs = line.strip().split()
        x = list()
        for a in attrs[1:]:
            pair = a.split(':')
            x.append(float(pair[1]))
        return np.array(x), float(attrs[0])

    x = list()
    y = list()
    with open(file) as f:
        for line in f:
            xi, yi = decompose(line)
            x.append(xi)
            y.append(yi)
    return np.array(x), np.array(y)


def calc_KLD(mu0, sig0, mu1, sig1):
    """Calculate the KL divergence from normal(mu1,sig1)
    to normal(mu0,sig0). I.e. KL(N0||N1)
    
    The formula is implemented following wikipedia
    KL diverence page section multivariate normal"""
    try: inv_sig1 = inv(sig1)
    except: inv_sig1 = pinv(sig1)
    res = np.trace(np.dot(inv_sig1, sig0))
    diff_mu = mu1 - mu0
    res += np.dot(np.dot(diff_mu.T,inv_sig1),diff_mu)
    res -= mu0.shape[0]
    if det(sig0) == 0 or det(sig1)/det(sig0) < 1e-5:
        return 0
    res += np.log(det(sig1)/det(sig0))
    res /= 2
    return float(res)

def log_comb(n, r):
    """The log of nCr"""
    r = min(r,n-r)
    if r == 0:
        return 0
    return sum(log(i) for i in xrange(n-r+1,n+1))\
        - sum(log(i) for i in xrange(1,r+1))

if __name__ == '__main__':
    x,y = read(sys.argv[1])
    print x.shape
    print y.shape

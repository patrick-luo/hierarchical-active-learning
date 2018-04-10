import numpy as np
import sys
import random
from scipy.stats import bernoulli
from sklearn.linear_model import LogisticRegression as LR
from sklearn.mixture import GaussianMixture as GM
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import det
from util import mean
from util import gini

class Sampling(object):
    """Work out one MLE model based on instance sampling and calculate
    the asymptotic normal distribution of the estimated parameters"""
    def __init__(self, xdata, sampling_num=10000):
        # the MLE parameters
        self.model = None
        # the mean of normal
        self.mu = None
        # the covariance of normal
        self.sigma = None

        x, y, xinfo = list(), list(), list()
        while len(xinfo) < sampling_num:
            idx = random.randint(0,xdata.shape[0]-1)
            x.append(xdata[idx])
            y.append(None)
            xinfo.append(idx)

        # the x data for training
        self.x = x
        # the y data for training
        self.y = y
        #the index in xdata for each instance.
        self.xinfo = xinfo

    @classmethod
    def train(cls, x, y):
        """Train a MLE model and return its distribution."""
        mdl = LR()
        mdl.fit(x, y)
        
        des_x = np.insert(x,0,1,axis=1)
        probs = mdl.predict_proba(x)
        w = np.diag(np.multiply(probs[:,0],probs[:,1]))
        fisher = np.dot(np.dot(des_x.T,w),des_x)
        try:
            sigma = inv(fisher)
        except:
            sigma = pinv(fisher)
        mu = np.insert(mdl.coef_,0,mdl.intercept_)
        mu = mu.reshape(mu.shape[0],1)
        return mdl, mu, sigma

    @classmethod
    def bern_samp(cls, prob):
        label = bernoulli.rvs(p=prob)
        return -1 if label == 0 else label

    def fit(self, new_regions):
        """Train a MLE estimator and its distribution given old labels
        and labels which are effected by the new_regions.
        
        In particular, assume each labeled region of instances
        follow binomial distribution, and sample labeled instances
        from this region. All the sampled instances will be aggregated
        together as one set of training data to work out the best
        MLE model.
        """
        for i,idx in enumerate(self.xinfo):
            for region in new_regions:
                if idx in region.index:
                    self.y[i] = Sampling.bern_samp(region.label)
                    break
        self.model, self.mu, self.sigma =\
            Sampling.train(np.array(self.x), np.array(self.y))

    def pred_prob(self, x):
        """Predict the positive probabilities."""
        assert self.model.classes_[1] == 1
        return self.model.predict_proba(x)[:,1]

    def pred_group_prob(self, group, xdata):
        x = [xdata[i] for i in group.index]
        probs = self.pred_prob(x)
        return mean(probs)

class Subspace(object):
    """Each is a subspace of the original feature space."""

    def __init__(self, xdata, region=None):
        """Fields:
            region is the hypercube description of its data.
            ###chiildren is a list of child nodes.
            index is a set of indices in training data.
            label is its group label
            err is the mean sample error
            all_same means if all instances are identical
        """
        def in_region(xi, region):
            """A region is a list of
            [lowerbound(exclusive), upperbound(inclusive)],
            where each indice is the dimension index."""
            for vj, rj in zip(xi, region):
                if vj <= rj[0] or vj > rj[1]:
                    return False
            return True

        def different(xi, xj):
            for vi, vj in zip(xi, xj):
                if vi != vj:
                    return True
            return False

        if region is None:
            self.region = [[-sys.float_info.max, sys.float_info.max]\
                for i in xrange(xdata.shape[1])]
        else:
            self.region = region
        #self.children = list()
        self.index = set()
        for idx, xi in enumerate(xdata):
            if in_region(xi, self.region):
                self.index.add(idx)
        self.label = None
        self.err = None
        self.all_same = True
        if len(self.index) > 0:
            i = min(self.index)
            for j in self.index:
                if different(xdata[i], xdata[j]):
                    self.all_same = False
                    break


    def test_split(self, xdata, model=None):
        """Test split from some value from one
        dimension to create two new children.
        Using unsupervised method when model is None,
        otherwise supervised"""
        x = [xdata[i] for i in self.index]
        if model is None:
            gm = GM(n_components=2,max_iter=500).fit(x)
            p = gm.predict_proba(x)[:,1]
        else:
            p = model.pred_prob(x)

        """(best dimension, best split value, lowest Gini score)"""
        d,v,score = None, None, sys.float_info.max
        for dj in range(len(self.region)):
            """dict(): key is distinct value on
            current dimension, value is a tuple
            of [number of points, number of '1's]"""
            val2cnts = dict()
            for xi, pi in zip(x, p):
                xij = xi[dj]
                if xij not in val2cnts:
                    val2cnts[xij] = [0, 0]
                val2cnts[xij][0] += 1
                val2cnts[xij][1] += pi
            val2cnts = sorted(val2cnts.items(), key=lambda t:t[0])
            if len(val2cnts) < 2: continue
            n = float(len(x))
            left_n, right_n = 0, n
            left_p, right_p = 0.0, float(sum(pi for pi in p))
            for (val, cnts) in val2cnts[:-1]:
                left_n += cnts[0]
                left_p += cnts[1]
                left_label = left_p/left_n
                right_n -= cnts[0]
                right_p -= cnts[1]
                right_label = right_p/right_n
                s = left_n/n*gini(left_label)+\
                        right_n/n*gini(right_label)
                if s < score:
                    d,v,score = dj, val, s
        assert (d is not None) and (v is not None)
        return d,v

    def split(self, d, v, train):
        (xdata,ydata) = train
        left_region = self.region[:]
        right_region = self.region[:]
        left_region[d] = [self.region[d][0], v]
        right_region[d] = [v, self.region[d][1]]
        left_space = Subspace(xdata, left_region)
        right_space = Subspace(xdata, right_region)
        assert len(self) == len(left_space) + len(right_space)
        left_space.query(ydata)
        right_space.query(ydata)
        final_gini = (gini(left_space.label)*len(left_space)+\
            gini(right_space.label)*len(right_space))/len(self)
        return [left_space, right_space], final_gini


    def __len__(self):
        return len(self.index)

    @classmethod
    def frr(cls, region):
        if len(region) == 0: return 0.0
        cnt = 0
        for rj in region:
            if rj[0] == -sys.float_info.max\
                    and rj[1] == sys.float_info.max:
                cnt += 1
        return float(cnt) / len(region)

    def query(self, ydata):
        pos, neg = 0, 0
        for idx in self.index:
            yi = ydata[idx]
            if yi > 0: pos += 1
            else: neg += 1
        assert pos+neg > 0
        self.label = float(pos)/(pos+neg)
        self.err = 2*len(self)*self.label*(1-self.label)





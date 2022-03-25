import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm, beta, bernoulli

#------------------------
#      Make data 
#-----------------------

def gen_data( n_trials=300):

    p_blue = np.zeros( n_trials)
    for i in range(n_trials):
        if i<120:
            p_blue[i] = 0.75
        elif i<160:
            p_blue[i] = 0.2
        elif i<200:
            p_blue[i] = 0.8
        elif i<230:
            p_blue[i] = 0.2
        elif i<260:
            p_blue[i] = 0.8
        else:
            p_blue[i] = 0.2
    blue_seen = np.zeros(n_trials)
    for i in range(n_trials):
        rand=np.random.rand(1)
        if rand < p_blue[i]:
            blue_seen[i] = 1

    return blue_seen

#-------------------------
#      Bayesian model
#-------------------------

def rbeta( r_space, r, v):
    '''Reparameterized beta

    Inputs:
        r_space: the discrete space for parameter r
        r: the value of r to infer the alpha parameter
        v: the 
    '''
    a = r/v
    b = (1/v)*(1-r)
    return beta.pdf( r_space, a, b)

class Bayes_model:

    def __init__( self, ):
        self._init_dist()

    def _ini_dist( self,):
        '''Init the disceted space
            r: reward probability
            v: the value of the space
            k: the volatility
        '''
        # get discerete space 
        self.n_split = 50 
        self.r_space = np.linspace( .01, .99, self.n_split)
        self.v_space = np.linspace( -11,  -2, self.n_split)
        self.k_space = np.linspace(  -2,   2, self.n_split)

    def _init_p_R1RV( self,):
        p_R1RV = np.zeros( [ self.n_split, self.n_split, self.n_split])
        for i, ri in enumerate( self.r_space):
            for j, vj in enumerate( self.v_space):
                p_R1RV[ :, i, j] = rbeta( self.r_space, ri, np.exp(vj))
        p_R1RV /= p_R1RV.sum(0,keepdims=True)
        self.p_R1RV = p_R1RV

    def _init_p_V1VK( self,):
        p_V1VK = np.zeros( [ self.n_split, self.n_split, self.n_split])
        for i, vi in enumerate( self.v_space):
            for j, kj in enumerate( self.k_space):
                p_V1VK[ :, i, j] = norm.pdf( self.v_space, loc=vi, scale=np.exp(kj))
        p_V1VK /= p_V1VK.sum(0,keepdims=True)
        self.p_V1VK = p_V1VK

    def _init_p_RVK( self,):
        f_RVK =  np.ones( [ self.n_split, self.n_split, self.n_split])
        self.p_RVK = f_RVK / f_RVK.sum()
        
    def udpate( self, x):
        '''
        '''

        #p(y|r)
        p_y1r = bernoulli.pdf( x, self.r_space)
        self.p_RVK *= p_y1r[ :, np.newaxis, np.newaxis]

        






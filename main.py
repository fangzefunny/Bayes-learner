import os
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from scipy.stats import norm, beta, bernoulli

# find the current path
path = os.path.dirname(os.path.abspath(__file__))
# create the folders for this folder
if not os.path.exists(f'{path}/figures'):
    os.mkdir(f'{path}/figures')
# define some color 
Blue    = .85 * np.array([   9, 132, 227]) / 255
Green   = .85 * np.array([   0, 184, 148]) / 255
Red     = .85 * np.array([ 255, 118, 117]) / 255
Yellow  = .85 * np.array([ 253, 203, 110]) / 255
Purple  = .85 * np.array([ 108,  92, 231]) / 255
colors    = [ Blue, Red, Green, Yellow, Purple]
sns.set_style("whitegrid", {'axes.grid' : False})

# image dpi
sfz, mfz, lfz = 11, 13, 16
dpi = 250

#------------------------
#      Make data 
#-----------------------

def get_data( n_trials=300):

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

    return blue_seen, p_blue

#-------------------------
#      Bayesian model
#-------------------------

def rbeta( r, v):
    '''Reparameterized beta

    r = a / (a+b)
    v = -log(a+b)

    Inputs:
        r_space: the discrete space for parameter r
        r: the value of r to infer the alpha parameter
        v: the 
    '''
    a = r*np.exp(-v)
    b = np.exp(-v)*(1-r)
    return beta( a, b)

class BayesLearner():

    def __init__( self):
        self._discretize()
        self._init_p_K()
        self._init_p_V1VK()
        self._init_p_R1VR()
        self._init_delta()

    def _discretize( self,):
        '''Implement all known conditions
            r: reward probability
            v: the value of the space
            k: the volatility

            Default dim convention:
            dim: yt, vt, rt, vt-1, rt-1, k
        '''
        # get discerete space 
        self.n_split = 50 
        self.r_space = np.linspace( .01, .99, self.n_split)
        self.v_space = np.linspace( -11,  -2, self.n_split)
        self.k_space = np.linspace(  -2,   2, self.n_split)

    def _init_p_K( self,):
        '''p(K) = Uniform 
        '''
        self.p_K = np.ones_like([self.k_space]) / self.n_split

    def _init_p_V1VK( self,):
        '''p(Vt|Vt-1=i,k) = N( i, exp(k))
        '''
        p_V1VK = np.zeros( [ self.n_split, self.n_split, self.n_split])
        for i, vi in enumerate( self.v_space):
            for k, kk in enumerate( self.k_space):
                p_V1VK[ :, i, k] = norm.pdf( self.v_space, loc=vi, scale=np.exp(kk))
        p_V1VK /= p_V1VK.sum(0,keepdims=True)
        self.p_V1VK = p_V1VK

    def _init_p_R1VR( self,):
        '''p(Rt|Vt-1=i,Rt-1=j) = rBeta(j,i)
        '''
        p_R1VR = np.zeros( [ self.n_split, self.n_split, self.n_split])
        for j, rj in enumerate( self.r_space):
            for i, vi in enumerate( self.v_space):
                p_R1VR[ :, i, j] = rbeta( rj, vi).pdf(self.r_space)
        p_R1VR /= p_R1VR.sum(0,keepdims=True)
        self.p_R1VR = p_R1VR

    def _init_delta( self,):
        '''δ0(vi,rj,kk)
            Init with Perks prior 
        '''
        f_VRK =  np.ones( [ self.n_split, self.n_split, self.n_split])
        self.delta = f_VRK / f_VRK.sum()
        
    def update( self, y):
        '''update δ with y 
        '''
        # get p(yt|rt): dim: rt
        p_y1r = bernoulli.pmf( y, self.r_space)
        # ∑i p(vt|vt-1=i, k) δ(i,rt-1,k)
        # dim: vt vt-1 @ vt-1, rt-1 --> vt, rt-1
        delta1 = np.zeros( [self.n_split]*3) 
        for k in range(self.n_split):
            delta1[:,:,k] = self.p_V1VK[:,:,k] @ self.delta[:,:,k]
        # ∑j p(rt|vt, rt-1=j) δ(vt,j,k)
        # dim:  rt rt-1 @ rt-1, k = rt k 
        delta2 = np.zeros( [self.n_split]*3) 
        for i in range(self.n_split):
            delta2[i,:,:] = self.p_R1VR[:,i,:]@delta1[i,:,:]
        # δ(vt,j,k) * p(y|rt=j)
        # get new delta: vt, rt, k
        delta = p_y1r[ np.newaxis, :, np.newaxis] * delta2
        self.delta = delta / delta.sum()
        # get some prediction
        self.p_V   = (self.delta.sum(axis=(1,2))*self.v_space).sum()
        self.p_R   = (self.delta.sum(axis=(0,2))*self.r_space).sum()
        
#-----------------------
#      Visualization  
#-----------------------

def fig1_volVar():
    '''Understand the reparameterized beta
    '''
    v_space = np.linspace( -11,  -2, 50)
    b_vars  = np.zeros_like( v_space)
    for i, v in enumerate(v_space):
        b_vars[i] = rbeta( r=1/2, v=v).var()
    fig, ax = plt.subplots( 1, 1, figsize=( 3, 3))
    sns.lineplot( x=v_space, y=b_vars, color=Blue, ax=ax)
    ax.set_xlabel( 'Value of V', fontsize=mfz)
    ax.set_ylabel( 'Var. of beta distribution', fontsize=mfz)
    ax.set_title( 'r=1/2')
    fig.tight_layout()
    plt.savefig( 'figures/fig1_volVar.png', dpi=dpi)

def fig2_sim():
    '''Simulate the experiment
    '''

    data, pTrue = get_data()
    model = BayesLearner()
    pVs, pRs = [], []
    # start simulte
    for y in data:
        model.update(y)
        pVs.append( model.p_V.copy())
        pRs.append( model.p_R.copy())
    
    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(6,4))
    ax1.plot( data,'go',label='noisy observations')
    ax1.plot( pTrue,color=Blue,label='true reward rate')
    ax1.plot( pRs,color=Red,label='estimated reward rate')
    ax1.legend()
    ax2.plot( pVs,'k',label='estimated volatility')
    ax2.legend()
    fig.tight_layout()
    plt.savefig( 'figures/fig2_sim.png', dpi=dpi)

if __name__ == '__main__':

    fig1_volVar()
    fig2_sim()



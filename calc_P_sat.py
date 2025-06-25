# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 23:48:31 2023

@author: tcava
"""
# from scipy.constants import k as kb  # constante de Boltzmann J /K
from scipy.constants import Avogadro as Navo
from jax import numpy as np
# import numpy as nnp

def calc_P_sat(T,guessP,eos,indxcomp=0):#,index):

    RES=1
    TOL=1e-5 
    MAX=100
    P=guessP*1
    i=0
    x = np.zeros((eos.ncomp))
    x = x.at[indxcomp].set(1)

    while(RES>TOL and i<MAX): 
        # densL = eos.dens_NR(x,T,P,phase='liq')
        # densV = eos.dens_NR(x,T,P,phase='vap')
        
        densL = eos.dens(T, P, x,phase='liq')
        densV = eos.dens(T, P, x,phase='vap')
        
        
        
        
        phiL = eos.phi(densL,x,T)
        phiV =  eos.phi(densV,x,T)

        if np.abs(densL-densV)/Navo<1e-3:
            # print('solução trivial')
            return np.nan
        
        phiL =phiL[indxcomp]
        phiV =phiV[indxcomp]
        
        # print(phiL,phiV)
        if np.isnan(phiV) or densV<0:
            phiV = phiL*10
            # print('phiL = nan')
            
        if np.isnan(phiL) or densL<0:
            phiL = phiV*10
            # print('phiV = nan')
        # print('P',P,end=' ')
        Pn=P*(phiL/phiV) 
        
        if i<=50:
            alpha=1
        elif 50<i<=75:
            alpha=.5
        elif 75<i<90:
            alpha=.25
        # elif 90<i<100:
        #     P_newt = calc_P_sat_newt(T,guessP,eos,indxcomp=0)
        #     # print('P_newt',P_newt)
        #     return P_newt
        # else:
        #     P_newt = calc_P_sat_newt2(T,guessP,eos,indxcomp=0)
        #     # print('P_newt',P_newt)
        #     return P_newt
        P = P*(1-alpha)+Pn*(alpha)
        
        RES=np.abs(phiL/phiV-1.) 
        i=i+1
        
        if i ==MAX:
            return np.nan

    return P
    

def calc_P_sat_newt(T,guessP,eos,indxcomp=0):#,index):
    x = np.zeros(len(eos.niki[0]))
    x = x.at[indxcomp].set(1)
    def RES_newt(v):
        P=10**v[0]
        RES=v*0
        
        densL = eos.dens(T, P, x,phase='liq')
        densV = eos.dens(T, P, x,phase='vap')
        
        phiL = eos.phi(densL,x,T)
        phiV =  eos.phi(densV,x,T)
        
        phiL =phiL[indxcomp]
        phiV =phiV[indxcomp]
        
        RES[0]=np.abs(phiL/phiV-1.) 
        return RES

    from scipy import optimize as opt
    v = opt.root(RES_newt,np.log10(guessP),tol=1e-6)
    # print(v.success)
    return 10**v.x[0]
    

def calc_P_sat_newt2(T,guessP,eos,indxcomp=0):#,index):
    x = np.zeros(len(eos.niki[0]))
    x = x.at[indxcomp].set(1)
    def RES_newt(v):
        P=np.exp(10**v[0])
        RES=v*0
        
        densL = eos.dens(T, P, x,phase='liq')
        densV = eos.dens(T, P, x,phase='vap')
        
        phiL = eos.phi(densL,x,T)
        phiV =  eos.phi(densV,x,T)
        
        phiL =phiL[indxcomp]
        phiV =phiV[indxcomp]
        
        RES[0]=np.abs(phiL/phiV-1.) 
        return RES

    from scipy import optimize as opt
    v = opt.root(RES_newt,np.log(np.log10(guessP)),tol=1e-7)
    # print(v.success)
    return np.exp(10**v.x[0])
    
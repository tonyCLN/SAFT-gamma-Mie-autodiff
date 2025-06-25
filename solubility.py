# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:41:37 2023

@author: tcava
"""
from scipy.constants import k as kb  # constante de Boltzmann J /K
from scipy.constants import Avogadro as Navo  # numero de avogadro mol^-1
from jax import numpy as np
import numpy as nnp
from scipy import optimize


def id_T(eos,T,P,indxcomp=0):
    ΔhSL = eos.deltaHSL[indxcomp]
    ΔcpSL = eos.deltacpSL[indxcomp]
    TSL = eos.TSL[indxcomp]
    
    R = Navo*kb
    kte = (ΔhSL/R*(1/TSL - 1/T)) - ΔcpSL/R * ( TSL*(1/T  - 1/TSL) - np.log(TSL/T) )
    
    x = np.exp(kte)
    
    return x
    

def sol_T(eos,T,P,propsol=np.array([0,1]),xsolguess=None,indxcomp=0,method = 'hybr',tol=None,nguesses=20):
    ΔhSL = eos.deltaHSL[indxcomp]
    ΔcpSL = eos.deltacpSL[indxcomp]
    TSL = eos.TSL[indxcomp]
    R = Navo*kb
    kte = (ΔhSL/R*(1/TSL - 1/T)) - ΔcpSL/R * ( TSL*(1/T  - 1/TSL) - np.log(TSL/T) )

    
    def residuo(xsol):
        # xsol = np.array([max(0,xsol[0])])
        # xsol = np.array([min(1,xsol[0])])
        x  = propsol * (1-xsol)
        x = x.at[indxcomp].set(xsol[0])
        gamma = eos.gamma(x,T,P,indxcomp)
                
        # print('x:',x)
        # print('gamma:',gamma)
        res = x[indxcomp]  - np.exp(kte)/gamma
        res = 1 - np.exp(kte)/gamma/x[indxcomp] 
        # res = x[indxcomp]*gamma  - np.exp(kte)
        
        f = nnp.asarray(res)
        
        # print(f,xsol,gamma)
        return f
    if xsolguess is None:
        guesses = np.logspace(-9,-0.3,nguesses)
        ress = nnp.ones(nguesses)
        for i in range (nguesses):
            ress[i] = abs(residuo(np.array([guesses[i]])))
            # print(ress[i])
            if ress[i] < 0.8:
                break 
            # print( ress[i] , guesses[i])
        xsolguess = np.array([guesses[np.argmin(ress)]])
        print('chute:',xsolguess,'res:',min(ress))
    ans1 = optimize.root(residuo, x0=xsolguess,method=method, tol=tol)['x']
    x  = propsol * (1-ans1)
    x = x.at[indxcomp].set(ans1[0])
    
    return x[indxcomp]


def solubilityT2(eos,T,P,propsol=np.array([1]),xsolguess=1e-2,indxcomp=0,method = 'hybr'):
    ΔhSL = eos.deltaHSL[indxcomp]
    ΔcpSL = eos.deltacpSL[indxcomp]
    TSL = eos.TSL[indxcomp]
    R = Navo*kb
    xsolguess = xsolguess *2
    # kte = (( -ΔhSL/(R*T)*(1-T/TSL) -  ΔcpSL/R*(np.log(TSL/T) - TSL/T + 1.0)))
    kte = (ΔhSL/R*(1/TSL - 1/T))
    def residuo(xsol):
        xsol = (np.sqrt(xsol))**2
        prop = propsol/nnp.sum(propsol)
        solventsx = prop * (1-xsol)
        x = np.hstack((xsol, solventsx))
        gamma = eos.gamma(x,T,P,indxcomp) #[indxcomp]

        res = 4*x[indxcomp]**2  - np.exp(kte)/(gamma**2)
        f = nnp.asarray(res)
        print(f,x)
        return f

    prop = propsol/np.sum(propsol)
    ans1 = optimize.root(residuo, x0=xsolguess,method=method)['x'] # root_scalar
    solventsx = prop * (1-ans1)
    x = np.hstack((ans1, solventsx))
    return x[indxcomp]

def solubilityT(eos,T,P,propsol=np.array([1]),xsolguess=1e-2,indxcomp=0,method = 'hybr'):
    ΔhSL = eos.deltaHSL[indxcomp]
    ΔcpSL = eos.deltacpSL[indxcomp]
    TSL = eos.TSL[indxcomp]
    R = Navo*kb
    xsolguess = xsolguess *2
    kte = 2*(( -ΔhSL/(R*T)*(1-T/TSL) -  ΔcpSL/R*(np.log(TSL/T) - TSL/T + 1.0)))
    # kte = 2*(ΔhSL/R*(1/TSL - 1/T))
    def residuo(xsol):
        xsol = (np.sqrt(xsol))**2
        prop = propsol/nnp.sum(propsol)
        solventsx = prop * (1-xsol)
        x = np.hstack((xsol, solventsx))
        gamma = eos.gamma(x,T,P,indxcomp) #[indxcomp]

        res = 4*x[indxcomp]**2  - np.exp(kte)/(gamma**2)
        f = nnp.asarray(res)
        print(f,x)
        return f

    prop = propsol/np.sum(propsol)
    ans1 = optimize.root(residuo, x0=xsolguess,method=method)['x'] # root_scalar
    solventsx = prop * (1-ans1)
    x = np.hstack((ans1, solventsx))
    return x[indxcomp]

        

def solubilityx(eos,x,P,Tguess,indxcomp=0):
    ΔhSL = eos.deltaHSL[indxcomp]
    ΔcpSL = eos.deltacpSL[indxcomp]
    TSL = eos.TSL[indxcomp]
    def residuo(T):
        # xi = x
        T0 = T[0]
        R = Navo*kb
        gamma = eos.gamma(x,T0,P,indxcomp,tol=1e-10)
        res = x[indxcomp] - np.exp(( -ΔhSL/(R*T0)*(1-T0/TSL) -  ΔcpSL/R*(np.log(TSL/T0) - TSL/T0 + 1.0)))/gamma - min(T,0)*1e+3
        f = np.array([res])
        return f
        
    ans1 = optimize.root(residuo, Tguess,tol=1e-15) # root_scalar

    return ans1["x"]

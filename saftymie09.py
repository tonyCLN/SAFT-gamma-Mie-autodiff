# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 00:38:57 2023

@author: tcava
"""
from jax import config
config.update('jax_enable_x64', True)
from functools import partial
from jax import jit as njit
from jax import lax
from scipy.constants import k as kb  # constante de Boltzmann J /K
from scipy.constants import Avogadro as Navo  # numero de avogadro mol^-1
from scipy.constants import pi
from scipy import optimize
import numpy as nnp
# import numpy as np
from jax import numpy as np
import numpy as npp
# from jax import numpy as nnp
from jax import jacfwd, lax
# from jax.experimental.host_callback import id_print

import pandas as pd
pd.options.mode.chained_assignment = None
data = pd.read_excel('saftgamma_database.xlsx')
datakl = pd.read_excel('saftgamma_database.xlsx','unlikemie_kl')
dataklab = pd.read_excel('saftgamma_database.xlsx','unlikeasso_kl') 


class saftymie():
    
    def __init__(self,groups, niki,M=None,deltaHSL=None,deltacpSL=None,TSL=None,
                 newgroups= None,newniki=None,newsig = None,neweps=None,newla=None,
                 newlr=None,newniast=None,newSk=None,newNst=None,newnH=None,newne1=None,
                 newne2=None,newepskl=None,newlrkl=None,newepsAB=None,newKAB=None,
                 newmu=None,newQ=None):
               
        
        orddata1 = pd.Index(groups, name='groups')
        data2 = data.set_index('groups').reindex(orddata1).reset_index()

        unlkdata2 = datakl[nnp.logical_and(datakl['group_k'].isin(groups), datakl['group_l'].isin(groups))]
    
        eps_kl_pivot = pd.pivot_table(unlkdata2, values='eps_kl', index='group_k', columns='group_l')
        eps_kl_arr = eps_kl_pivot.reindex(index=groups, columns=groups) # .fillna(0)
        
        
        if  neweps is not   None:
            ind = nnp.where(nnp.isin(groups, newgroups))[0]
            data2.sigma_kk.values[ind]= newsig
            data2.eps_kk.values[ind] = neweps
            data2.la_kk.values[ind] = newla
            data2.lr_kk.values[ind] = newlr
            data2.niast.values[ind] = newniast
            data2.Sk.values[ind] = newSk
            data2.Nst_kk.values[ind] = newNst
            data2.nH_kk.values[ind] = newnH
            data2.ne1_kk.values[ind] = newne1
            data2.ne2_kk.values[ind] = newne2
            

        if newepskl is not None:
            eps_kl_arr.loc[:,newgroups] = eps_kl_arr.loc[:,newgroups].replace({np.nan:newepskl})
            eps_kl_arr.loc[newgroups,newgroups] = 0
            # eps_kl_arr[:] = newepskl
            
            
        eps_kl_arr = eps_kl_arr.fillna(0)
        eps_kl_sym = eps_kl_arr + eps_kl_arr.T 
        
        
        eps_kl_sym.replace(0, np.nan, inplace=True)
        
        
        lr_kl_pivot = pd.pivot_table(unlkdata2, values='lr_kl', index='group_k', columns='group_l')
        lr_kl_arr = lr_kl_pivot.reindex(index=groups, columns=groups)
        
        if newlrkl is not None:
            lr_kl_arr.loc[:,newgroups] = lr_kl_arr.loc[:,newgroups].replace({np.nan:newlrkl})
            
        
        
        lr_kl_arr = lr_kl_arr.fillna(0)
        lr_kl_sym = lr_kl_arr + lr_kl_arr.T 
        lr_kl_sym.replace(0, np.nan, inplace=True)
        
        
        self.eps_kl = eps_kl_sym.values                    
        self.groups = groups
        self.niki = niki
        
        
        self.niast = data2.niast.values
        self.Sk = data2.Sk.values  #Sk
        
        self.ncomp = len(niki[0])
        self.ngroup = len(groups)
        self.M = M
        self.sigma =  data2.sigma_kk.values*1e-10 #sigma*1e-10
        
        self.epsilon = data2.eps_kk.values*kb  # epsilon*kb
        self.epsilonkl = ymie_epsilonkl(self.sigma,self.epsilon,self.ngroup,self.eps_kl)
        
        
        
        self.lambr = data2.lr_kk.values
        self.lamba = data2.la_kk.values
        self.lamb = np.array([self.lamba, self.lambr])
        lambr_kl = lr_kl_sym.values
        self.lambkl = ymie_lambkl(self.lamb,lambr_kl)
        
        self.phicoef = nnp.array([[ 7.5365557e+00, -3.7604630e+01,  7.1745953e+01, -4.6835520e+01,-2.4679820e+00, -5.0272000e-01,  8.0956883e+00],
               [-3.5944000e+02,  1.8256000e+03, -3.1680000e+03,  1.8842000e+03, -8.2376000e-01, -3.1935000e+00,  3.7090000e+00],
               [ 1.5509000e+03, -5.0701000e+03,  6.5346000e+03, -3.2887000e+03,-2.7171000e+00,  2.0883000e+00,  0.0000000e+00],
               [-1.1993200e+00,  9.0636320e+00, -1.7948200e+01,  1.1340270e+01, 2.0521420e+01, -5.6637700e+01,  4.0536830e+01],
               [-1.9112800e+03,  2.1390175e+04, -5.1320700e+04,  3.7064540e+04,   1.1037420e+03, -3.2646100e+03,  2.5561810e+03],
               [ 9.2369000e+03, -1.2943000e+05,  3.5723000e+05, -3.1553000e+05,  1.3902000e+03, -4.5182000e+03,  4.2416000e+03],
               [ 1.0000000e+01,  1.0000000e+01,  5.7000000e-01, -6.7000000e+00, -8.0000000e+00,   nnp.nan,  nnp.nan]])
        
        self.c = np.array([[ 8.1096e-01,  1.7888e+00, -3.7578e+01,  9.2284e+01],
               [ 1.0205e+00, -1.9341e+01,  1.5126e+02, -4.6350e+02],
               [-1.9057e+00,  2.2845e+01, -2.2814e+02,  9.7392e+02],
               [ 1.0885e+00, -6.1962e+00,  1.0698e+02, -6.7764e+02]])
        

        self.Nstk = data2.Nst_kk.values
        self.nhk = data2.nH_kk.values 
        self.ne1k = data2.ne1_kk.values 
        self.ne2k = data2.ne2_kk.values 
        
        
        self.nka = np.stack([self.nhk, self.ne1k,self.ne2k], axis=1)
        
        self.nsite = len(self.nka.T)

        self.idkasc = nnp.nonzero(self.Nstk)[0]
        
        
        self.idxasc = np.unique(np.nonzero(niki[self.idkasc])[1])


        groupsasc = groups[self.idkasc]
        dataklab2 = dataklab[nnp.logical_and(dataklab['group_k'].isin(groupsasc), dataklab['group_l'].isin(groupsasc))]
        
        
        indxasck = pd.Categorical(dataklab2.group_k, categories=groupsasc, ordered=True).codes #.reshape(lendataab,1)
        indxascl = pd.Categorical(dataklab2.group_l, categories=groupsasc, ordered=True).codes #.reshape(lendataab,1)
        indxascak = dataklab2.site_a_group_k.values
        indxascbl = dataklab2.site_b_group_l.values
        
        
        
        
        
        if newNst is not None and newNst != 0 :
            # dataklab2 = dataklab2.replace({np.nan:newepsABKAB})
            dataklab2.loc[:,'epsAB_kl'] = dataklab2.loc[:,'epsAB_kl'].replace({np.nan:newepsAB})
            dataklab2.loc[:,'KAB_kl'] = dataklab2.loc[:,'KAB_kl'].replace({np.nan:newKAB})
            
        
        self.dataklab2 = dataklab2
        
        self.ngroupasc = len(self.idkasc)
        self.ncompasc = len (self.idxasc)
        
        self.eklAB = nnp.zeros([self.ngroupasc,self.ngroupasc,3,3])
        self.kappaklAB = self.eklAB.copy()
        
        self.eklAB[indxasck,indxascl,indxascak,indxascbl] = self.eklAB[indxascl,indxasck,indxascbl,indxascak] = dataklab2.epsAB_kl.values*kb
        self.kappaklAB[indxasck,indxascl,indxascak,indxascbl] = self.kappaklAB[indxascl,indxasck,indxascbl,indxascak] = dataklab2.KAB_kl.values*1e-30
        
        
        cpq = np.array([
            [7.56425183020431E-02, -1.28667137050961E-01, 1.28350632316055E-01, -7.25321780970292E-02, 2.57782547511452E-02, -6.01170055221687E-03, 9.33363147191978E-04, -9.55607377143667E-05, 6.19576039900837E-06, -2.30466608213628E-07, 3.74605718435540E-09],
            [1.34228218276565E-01, -1.82682168504886E-01, 7.71662412959262E-02, -7.17458641164565E-04, -8.72427344283170E-03, 2.97971836051287E-03, -4.84863997651451E-04, 4.35262491516424E-05, -2.07789181640066E-06, 4.13749349344802E-08, 0.0],
            [-5.65116428942893E-01, 1.00930692226792E+00, -6.60166945915607E-01, 2.14492212294301E-01, -3.88462990166792E-02, 4.06016982985030E-03, -2.39515566373142E-04, 7.25488368831468E-06, -8.58904640281928E-08, 0.0, 0.0],
            [-3.87336382687019E-01, -2.11614570109503E-01, 4.50442894490509E-01, -1.76931752538907E-01, 3.17171522104923E-02, -2.91368915845693E-03, 1.30193710011706E-04, -2.14505500786531E-06, 0.0, 0.0, 0.0],
            [2.13713180911797E+00, -2.02798460133021E+00, 3.36709255682693E-01, 1.18106507393722E-03, -6.00058423301506E-03, 6.26343952584415E-04, -2.03636395699819E-05, 0.0, 0.0, 0.0, 0.0],
            [-3.00527494795524E-01, 2.89920714512243E+00, -5.67134839686498E-01, 5.18085125423494E-02, -2.39326776760414E-03, 4.15107362643844E-05, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-6.21028065719194E+00, -1.92883360342573E+00, 2.84109761066570E-01, -1.57606767372364E-02, 3.68599073256615E-04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.16083532818029E+01, 7.42215544511197E-01, -8.23976531246117E-02, 1.86167650098254E-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.02632535542427E+01, -1.25035689035085E-01, 1.14299144831867E-02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [4.65297446837297E+00, -1.92518067137033E-03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-8.67296219639940E-01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        
        self.cpq = np.array(cpq)
        
        self.deltaHSL = deltaHSL
        self.deltacpSL = deltacpSL
        self.TSL = TSL
        
        self.C1 = np.array([
                        [-0.488498, 0.863195, 0.761344, -0.750086, -0.218562, -0.538463],
                        [-0.611295, 1.10339, 0.921359, -0.838804, -0.197999, -0.940714],
                        [-0.69879, 1.278054, 1.020604, -0.880027, -0.140749, -1.222733],
                        [-0.735855, 1.351533, 1.058986, -0.894024, -0.104174, -1.338626],
                        [-0.800934, 1.479538, 1.121453, -0.914864, -0.02015, -1.53607],
                        [-0.856655, 1.587957, 1.169885, -0.928269, 0.074849, -1.698853]    ])
        
        self.C2 = np.array([
                        [-1.050534, 1.747476, 1.749366, -1.999227, -0.661046, -3.02872],
                        [-1.30955, 2.24912, 2.135877, -2.27853, -0.773166, -3.70469],
                        [-1.490116, 2.619997, 2.404319, -2.420706, -0.829466, -3.930928],
                        [-1.616385, 2.881007, 2.5776, -2.48499, -0.828596, -4.175589],
                        [-1.940503, 3.552034, 2.593808, -2.593808, -0.724353, -4.899975]    ])
        
        
        self.muk = data2.Mu.fillna(0).values
        self.Qk = data2.Q.fillna(0).values
        
        # self.idkmulti = (np.unique( np.array( list(np.where(self.Qk!=0)[0]) + list(np.where(self.muk!=0)[0]))))
        # self.idkmulti =  np.array( list(np.where(self.Qk!=0)[0]))
        # self.idkmulti = nnp.unique(nnp.array([nnp.nonzero(self.Qk)[0]]))#,np.nonzero(self.muk)[0]]))
        # self.idkmulti = np.array([6])
        self.idkmulti =  nnp.nonzero(self.Qk)[0]
        self.ngroupmulti = len(self.idkmulti)
        
        self.muk = self.muk[self.idkmulti]* (1e+4/1.3807) ** 0.5 #2.69122703924268e-06
        self.Qk = self.Qk[self.idkmulti]*(1e+4/1.3807) ** 0.5 
        
        
        return
    def xk(self,x):
        xk = ymie_xk(x,self.niast,self.niki,self.Sk)
        return xk
    
    def xsk(self,x):
        xsk = ymie_xsk(x,self.niast,self.niki,self.Sk)
        return xsk
    
    def dkk(self,x,T):
        
        dkk = ymie_dkk(x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup)
        return dkk
    
    def dkl(self,x,T):
        
        dkl = ymie_dkl(x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup)
        return dkl
    
    def rhos(self,rho,x):
        rhos = ymie_rhos(rho,x,self.niast,self.niki,self.Sk)
        return rhos
    
    def epsilonkl(self):
        epsilonkl = ymie_epsilonkl(self.sigma,self.epsilon,self.ngroup,self.eps_kl)
        return epsilonkl
    
    def epsilonmed(self):
        epsilonmed = ymie_epsilonmed(self.epsilon,self.epsilonkl,self.sigma,self.ngroup,self.niast,self.niki,self.Sk)
        return epsilonmed
    
    def csim(self,rho,x,T):
        
        csim = ymie_csim(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.niast,self.niki,self.Sk,self.ngroup)
        return csim
    
    def A_hs(self,rho,x,T):
        A_hs = ymie_A_hs(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.niast,self.niki,self.Sk,self.ngroup)
        return A_hs
    
    def a1(self,rho,x,T):
        a1 = ymie_a1(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return a1
    
    def A1(self,rho,x,T):
        A1 = ymie_A1(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return A1           
    
    # def csikleff(self,rho,x):
    #     csikleff = ymie_csikleff(rho,x,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
    #     return csikleff
    
    def csimedeff(self,rho,x,T):

        csikleff = ymie_csikleff(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return csikleff

    def csix(self,rho,x,T):
        csix = ymie_csix(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk)
        return csix
    
    def Bkl(self,rho,x,T):
        medlamb = np.array([(self.lamb[1]+self.lamb[0]), (self.lamb[1]+self.lamb[0])])
        Bkl = ymie_Bkl(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,medlamb,self.ngroup,self.niast,self.niki,self.Sk)
        return Bkl
    
    def x0kl(self,x,T):

        x0kl = ymie_x0kl(x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup)
        
        return x0kl
    
    def a1skl(self,rho,x,T):
        a1skl = ymie_a1skl(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return a1skl 
    
    # def a1kl(self,rho,x,T):

    #     csix = ymie_csix(rho,x,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk)
    #     a1kl = ymie_a1kl(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,csix)
    #     return a1kl
    
    def a1kl(self,rho,x,T):
        a1kl = ymie_a1kl(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return a1kl
    
    
    def Khs(self,rho,x,T):
        Khs = ymie_Khs(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk)
        return Khs
        
    def fm(self):
        
        fm = ymie_fm(self.phicoef,self.lambkl)
        return fm
    
    def alphakl(self):
        alphakl = ymie_alphakl(self.lambkl)
        return alphakl
    
    def csixast(self,rho,x):
        csixast = ymie_csixast(rho,x,self.sigma, self.ngroup,self.niast,self.niki,self.Sk)
        return csixast
    
    def chikl(self,rho,x):
        chikl = ymie_chikl(rho,x,self.sigma,self.lamb, self.ngroup,self.niast,self.niki,self.Sk,self.phicoef)
        return chikl

    def a2kl(self,rho,x,T):
       
        a2kl = ymie_a2kl(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
        return a2kl
    
    def a2(self,rho,x,T):
        a2 = ymie_a2(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
        return a2
    
    def a3(self,rho,x,T):
        a3 = ymie_a3(rho,x,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.phicoef)
        return a3
    
    def A2(self,rho,x,T):
        A2 = ymie_A2(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
                            
        return A2
    
    def a3kl(self,rho,x):
        
        a3kl = ymie_a3kl(rho,x,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.phicoef)
        return a3kl
    
    def A3(self,rho,x,T):
        A3 = ymie_A3(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.phicoef)
        return A3           
    
    def Zki(self):
        Zki = ymie_Zki(self.niast,self.niki,self.Sk)
        return Zki
    
    def sigmamed3(self):
        sigmamed3 = ymie_sigmamed3(self.sigma,self.ngroup, self.niast,self.niki,self.Sk)
        return sigmamed3
    
    def km(self,rho,x,T):
        km = ymie_km(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk)
        return km
    
    def ghs(self,rho,x,T):
        ghs = ymie_ghs(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk)
        return ghs
    
    def g1(self,rho,x,T):
        g1 = ymie_g1(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return g1
    
    def g2(self,rho,x,T):
        g2= ymie_g2(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
        return g2
    
    def lggmie(self,rho,x,T):
        lggmie = ymie_gmie(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
        return lggmie
    
    def ycii(self,rho,x,T):
        ycii = ymie_ycii(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.phicoef)
        return ycii
    
    def g2mca(self,rho,x,T):
        g2mca =  ymie_g2mca(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
        return g2mca
    
    def Achain(self,rho,x,T):
        Achain = ymie_Achain(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
        return Achain
    
    def AALL(self,rho,x,T):
        
        A_hs = ymie_A_hs(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.niast,self.niki,self.Sk,self.ngroup)
        A1 = ymie_A1(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        A2 = ymie_A2(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
        A3 = ymie_A3(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.phicoef)
    
        Achain = ymie_Achain(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef)
        A_asc = ymie_Aasc(rho,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.ngroup,self.niast,self.niki,self.Sk,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        Ares = A_hs+A1+A2+A3+Achain+A_asc
        
        return np.array([A_hs,A1,A2,A3,Achain,A_asc,Ares])

    
    def A_res(self,rho,x,T):
        A_res = ymie_A_res(rho,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return A_res
    
    def mu_res(self,rho,x,T):
        ans = ymie_mu_res(rho,x,T,self.ncomp, self.epsilon, self.sigma, self.lamb, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return ans
    
    def phi(self,rho,x,T):
        phi = ymie_phi(rho,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return phi
    
    def phi_xTP(self,x,T,P):
        phiL,phiV,rhoL,rhoV = ymie_phi_xTP(x,T,P,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return phiL,phiV,rhoL,rhoV
    
    def resi_sat_phis(self,T,P,indxcomp):
        
        resi_sat_phis = ymie_resi_sat_phis(T,P,indxcomp,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        dresi_phis_dP = ymie_dresi_sat_phis_dP(T,P,indxcomp,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        
        return resi_sat_phis,dresi_phis_dP
    

    def sat_dens(self,T,Pguess,indxcomp=0,method='hybr',tol=1e-9):
        
        x = np.zeros(len(self.niki[0]))
        x = x.at[indxcomp].set(1)
        
        def residuo(Pi):
            P,=Pi
            phiL,phiV = self.phi_TPx(T,P,x)
            res = phiL[indxcomp]/phiV[indxcomp] - 1 
            f = nnp.asarray([res])
            return f
        ans = optimize.root(residuo, [Pguess,], method=method,tol=tol)
        Psat = ans["x"][0]
        sat_densL,sat_densV = self.dens(T,Psat,x,method=method,xtol=tol,tol=tol)
        
        return sat_densL,sat_densV,Psat
    
    def sat_P(self,T,Pguess,indxcomp=0,method='hybr',tol=1e-9):
        
        x = np.zeros(len(self.niki[0]))
        x = x.at[indxcomp].set(1)
        
        def residuo(Pi):
            P,=Pi
            phiL,phiV = self.phi_TPx(T,P,x)
            phiL = phiL[indxcomp]
            phiV = phiV[indxcomp]
            res = phiL/phiV - 1 
            f = nnp.asarray([res])
            # print(res,phiL,phiV)
            return f
        ans = optimize.root(residuo, [Pguess,], method=method,tol=tol)
        Psat = ans["x"][0]
        
        return Psat
    
    
        
    def P(self,rho,x,T):
        P = ymie_P(rho,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return P
    
    def dPdrho(self,rho,x,T):
        
        dPdrho = ymie_dPdrho(rho,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        
        return dPdrho
    
    def SoS(self,T,P,x,phase):
        rho = self.dens(T,P,x,phase)
        SoS = ymie_SoS(rho,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc,self.M)
        return SoS
        
    def kappaT(self,T,P,x):
        rhov = self.dens(T,P,x,'liq')
        kappaT = ymie_kappaT(rhov,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return kappaT
    
    def alphaP(self,T,P,x):
        rhov = self.dens(T,P,x,'vap')
        alphaP = ymie_alphaP(rhov,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return alphaP
    
    def alphaP2(self,T,P,x):
        rhoV = self.dens(T,P,x,'vap')
        drho_dT = jacfwd(self.dens,argnums=0)(T, P, x, 'vap')
        print(drho_dT)
        alphaP = - drho_dT/rhoV
        alphaP = alphaP/(1e+4)
        return alphaP
    
    
    def x0iimed(self,x,T):
        x0iimed = ymie_x0iimed(x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk)
        return x0iimed
    
    def a1siimed(self,rho,x,T):
        a1siimed = ymie_a1siimed(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return a1siimed
    
    def aklbkl(self,rho,x,T):
        Bkl_2lamb,Bkl_medlamb =  ymie_Bkl_all(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk)
        a1skl_2lamb,a1skl_medlamb = ymie_a1skl_all(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return Bkl_2lamb,a1skl_2lamb,Bkl_medlamb,a1skl_medlamb
    
    def suma1sb(self,rho,x,T):
        Biimed_2lamb,Biimed_medlamb = ymie_Biimed_all(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk)
        a1siimed_2lamb,a1siimed_medlamb = ymie_a1siimed_all(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)

        return a1siimed_2lamb[0],Biimed_2lamb[0],a1siimed_2lamb[1],Biimed_2lamb[1],a1siimed_medlamb,Biimed_medlamb
    
    def lambs(self):
        _2lambkl,medlambkl  = ymie_lambkl_all(self.lamb)
        _2lambmed,medlambmed = ymie_lambmed_all(self.lamb,self.ngroup,self.niast,self.niki,self.Sk)
        return _2lambkl,medlambkl,_2lambmed,medlambmed
    
    def csikleff(self,rho,x,T):
        csikleff = ymie_csikleff(rho,x,T,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c)
        return csikleff
    
    def Iij(self,rho,x,T):
        Iij = ymie_Iij(rho,x,T,self.epsilon,self.epsilonkl,self.sigma, self.ngroup,self.niast,self.niki,self.Sk,self.cpq,self.ncomp,self.idxasc)
        return Iij
    
    def sigmax3(self,x):
        sigmax3 = ymie_sigmax3(x,self.sigma, self.ngroup,self.niast,self.niki,self.Sk)
        return sigmax3
    
    def sigmakl(self):
        sigmakl =ymie_sigmakl(self.sigma, self.ngroup)
        return sigmakl
    
    def Fklab(self,T):
        Fklab = ymie_Fklab(T,self.eklAB)
        return Fklab
    
    def delta(self,rho,x,T):
        delta = ymie_delta(rho,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.ngroup,self.niast,self.niki,self.Sk,self.eklAB, self.kappaklAB,self.cpq,self.ngroupasc,self.idxasc)
        return delta        
    
    # def eklAB(self):
    #     eklAB = ymie_eklAB(self.eklAB,self.ngroupasc)
    #     return eklAB
    
    
    def X_tan(self,rho,x,T):
        X_tan = ymie_X_tan(rho,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.ngroup,self.niast,self.niki,self.Sk,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return X_tan
    
    def A_asc(self,rho,x,T):
        A_asc = ymie_Aasc(rho,x,T,self.ncomp,self.sigma,self.ngroup,self.niast,self.niki,self.Sk,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return A_asc
    
    def Cvres(self,T,P,x,phase):
        rho = self.dens(T,P,x,phase)
        Cvres = ymie_Cvres(rho,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return Cvres
    
    def Cpres(self,T,P,x,phase):
        rho = self.dens(T,P,x,phase)
        Cpres = ymie_Cpres(rho,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        return Cpres
    
    def Cpexc(self,T,P,x,phase):
        rhom = self.dens(T,P,x,phase)
        Cpres = ymie_Cpres(rhom,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        
        xp = np.identity(self.ncomp)
        rhop = np.ones(self.ncomp)
        Cpresp = rhop.copy()
        
        for i in range(self.ncomp):
            rhop =  rhop.at[i].set(self.dens(T,P,xp[i],phase))
            Cpresp = Cpresp.at[i].set(ymie_Cpres(rhop[i],xp[i],T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc))
            
            
        Cpexc = Cpres - np.einsum('i,i->',x,Cpresp)
        return Cpexc
    
    def Hexc(self,T,P,x,phase):
        rhom = self.dens(T,P,x,phase)
        Hres = ymie_Hres(rhom,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        
        xp = np.identity(self.ncomp)
        rhop = np.ones(self.ncomp)
        Hresp = rhop.copy()
        
        for i in range(self.ncomp):
            rhop =  rhop.at[i].set(self.dens(T,P,xp[i],phase))
            Hresp = Hresp.at[i].set(ymie_Hres(rhop[i],xp[i],T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc))
            
            
        Hexc = Hres - np.einsum('i,i->',x,Hresp)
        return Hexc
    
    
    def xsk_multi(self,x):
        ans = ymie_xsk_multi(x,self.niast,self.niki,self.Sk,self.idkmulti)
        return ans
    
    def Qvsmu(self,x):
        ans = ymie_Qvsmu_multi(x,self.niast,self.niki,self.Sk,self.idkmulti,self.Qk,self.muk)
        return ans
    
    def sig_multi(self):
        ans = ymie_sig_multi(self.sigma,self.ngroup,self.idkmulti)
        return ans
    
    def A_multi(self,rho,x,T):
        A_multi = ymie_A_multi(rho,x,T,self.epsilonkl,self.sigma,self.ngroup, self.niast, self.niki, self.Sk,self.ngroupmulti,self.idkmulti,self.Qk,self.muk,self.C1,self.C2)
        return A_multi
    
    def dens(self, T, P, x, phase=None, method=None,xtol=None,tol=None):
        #[molecule/m³] -> *Navo -> [mol/m^3] -> *MW -> [g/m^3]

        etaguessL = 0.5
        etaguessV = 1e-10

        sigmaimed3 = ymie_sigmamed3(self.sigma, self.ngroup,self.niast,self.niki,self.Sk)
        Zi = np.sum(np.einsum('ki,k,k->ki ', self.niki,self.niast,self.Sk),axis=0)
        soma = np.einsum('i,i,i-> ',x,Zi,sigmaimed3)
            
        densL0 = 6/pi*etaguessL/soma
        densV0 = 6/pi*etaguessV/soma

        def residuo(dens):
            densi, = dens
            res0 = (((self.P(densi, x, T))/P) -1)
            f = nnp.asarray([res0])
            return f

        def residuo_log(dens_ad):  # escalar

            densi = dens_ad[0]
            pcalc = self.P(densi, x, T)
            res0 = np.log(pcalc / P)
            f = [res0]

            return f
        
        if method is None:
            method = 'hybr'

        if phase is None or phase == 'liq':
            ans = optimize.root(residuo, [densL0, ], method=method,tol=tol)
            densL_1 = ans["x"][0]

            if phase == 'liq':
                return densL_1

        if phase is None or phase == 'vap':
            ans = optimize.root(residuo_log, [densV0, ], method=method,tol=None)
            densV_1_ad = ans["x"][0]
            densV_1 = densV_1_ad
            if phase == 'vap':
                return densV_1
            
        return densL_1, densV_1
    
    def gamma(self,x,T,P,indxcomp=0,tol=None):
        
        rhox = self.dens(T,P,x,'liq',tol)
        
        phi = ymie_phi(rhox,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        
        xpuro = np.zeros(len(x))
        xpuro = x.at[indxcomp].set(1)

        rhopuro = self.dens(T,P,xpuro,'liq',tol)
        
        phip = ymie_phi(rhopuro,xpuro,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        gamma  = phi/phip

        return gamma[indxcomp] #,np.log(phip),np.log(phi)
    
    
    def densV(self,x,T,P):
        densV = ymie_densV(x,T,P,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        
        return densV
    
    def densL(self,x,T,P):
        densL = ymie_densL(x,T,P,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        
        return densL
    
    def dens_NR(self,x,T,P,phase=None):
        if phase is None:
            phase=0
            densL = ymie_dens(x,T,P,phase,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
            phase=1
            densV = ymie_dens(x,T,P,phase,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
            dens =  densL,densV
        elif phase == 'liq':
            phase=0
            dens = densL = ymie_dens(x,T,P,phase,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        elif phase == 'vap':
            phase=1
            dens = ymie_dens(x,T,P,phase,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
        else:
            print('ERROR: the phase values are \'vap\', \'liq\' or None')
            return
        return dens
    
    def densV2(self,x,T,P):
        
        etaguessV = 1e-9
        sigmaimed3 = ymie_sigmamed3(self.sigma, self.ngroup,self.niast,self.niki,self.Sk)
        Zi = np.sum(np.einsum('ki,k,k->ki ', self.niki,self.niast,self.Sk),axis=0)
        soma = np.einsum('i,i,i-> ',x,Zi,sigmaimed3)
            
        rho0 = 6/pi*etaguessV/soma
        
        tolmax = 1e-5
        itmax = 20
    
        
        tol = 1
        it = 0
        pontos = 20
        vrho = nnp.geomspace(1e+19,1.5e+28,pontos)
        vP = nnp.zeros(pontos)
        it_rho = []
        it_rho.append(rho0)
        for i in range (pontos):
            vP[i] = self.P(vrho[i],x,T)
            
        
        while tol>tolmax and it<itmax:
            
            Pcalc = ymie_P(rho0,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
            res =  ( Pcalc - P)
            dres = ymie_dPdrho(rho0,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
            grad = -res/dres
            rhon = rho0 + grad
            print(float(rhon),it,res,dres)
            tol = abs((rhon-rho0)/rhon)
            it = it+1
            
            rho0 =rhon*1
            it_rho.append(rho0)
        # return rhon
        return rhon,vP,vrho,it_rho
    

    def densL2(self,x,T,P):
        
        etaguessV = 0.5
        sigmaimed3 = ymie_sigmamed3(self.sigma, self.ngroup,self.niast,self.niki,self.Sk)
        Zi = np.sum(np.einsum('ki,k,k->ki ', self.niki,self.niast,self.Sk),axis=0)
        soma = np.einsum('i,i,i-> ',x,Zi,sigmaimed3)
            
        rho0 = 6/pi*etaguessV/soma
        
        tolmax = 1e-9
        itmax = 20
        
        tol = 1
        it = 0
        
        while tol>tolmax and it<itmax:
            
            Pcalc = ymie_P(rho0,x,T,self.ncomp,self.epsilon,self.epsilonkl,self.sigma,self.lamb,self.lambkl,self.ngroup,self.niast,self.niki,self.Sk,self.c,self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
            res =  Pcalc - P
            dres = ymie_dPdrho(rho0,x,T,self.ncomp, self.epsilon,self.epsilonkl, self.sigma, self.lamb,self.lambkl, self.ngroup, self.niast, self.niki, self.Sk, self.c, self.phicoef,self.eklAB,self.kappaklAB,self.cpq,self.nka,self.nsite,self.ngroupasc,self.ncompasc,self.idxasc,self.idkasc)
            rhon = rho0 - res/dres
            
            tol = abs((rhon-rho0)/rhon)
            it = it+1
            
            rho0 =rhon*1
        return rhon
    
    
    
    
# ---- par
# cnv
@partial(njit)
def ymie_lambkl(lamb,lambr_kl):

    lambakl = 3 + np.sqrt(np.einsum('k,l->kl',(lamb[0]-3),(lamb[0]-3)))
    lambrkl = 3 + np.sqrt(np.einsum('k,l->kl',(lamb[1]-3),(lamb[1]-3)))
    
    lambrkl = np.nan_to_num(lambr_kl, nan=lambrkl)
    
    lambkl = np.array([lambakl, lambrkl])
    
    return lambkl

@partial(njit)
def ymie_lambkl_all(lambkl):
    
    medlambkl = np.sum(lambkl,axis=0)
    # _2lamb = 2*lamb
    
    _2lambkl = lambkl*2
    
    # medlambkl = 3 + np.sqrt(np.einsum('k,l->kl',(medlamb-3),(medlamb-3)))
    
    return _2lambkl,medlambkl

#EQ2 cnv
@partial(njit)
def ymie_Ckl(lambkl):
    # lambkl = ymie_lambkl(lamb)
    Ckl =  lambkl[1]/ (lambkl[1] - lambkl[0]) * (lambkl[1]/lambkl[0])**(lambkl[0]/ (lambkl[1] - lambkl[0]))
    return Ckl

@partial(njit, static_argnames=['ngroup'] )
def ymie_Ciimed(lamb,lambkl,ngroup,niast,niki,Sk):
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    Ciimed =  lambmed[1]/ (lambmed[1] - lambmed[0]) * (lambmed[1]/lambmed[0])**(lambmed[0]/ (lambmed[1] - lambmed[0]))
    return Ciimed


#EQ33 cnv
@partial(njit)
def ymie_alphakl(lambkl):
    # lambkl = ymie_lambkl(lamb)
    Ckl = ymie_Ckl(lambkl)

    alphakl = Ckl*(1/(lambkl[0]-3) - 1/(lambkl[1]-3))
    
    return alphakl

#EQ61
@partial(njit, static_argnames=['ngroup'])
def ymie_alphaiimed(lamb,lambkl,ngroup,niast,niki,Sk):
    
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    Ciimed = ymie_Ciimed(lamb,lambkl,ngroup,niast,niki,Sk)

    alphakl = Ciimed*(1/(lambmed[0]-3) - 1/(lambmed[1]-3))
    
    return alphakl

# cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_sigmakl(sigma, ngroup):
    
    sigmakl = (sigma.reshape((ngroup,1))+sigma)/2
    
    return sigmakl

# cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_epsilonkl(sigma,epsilon,ngroup,eps_kl):
    
    sigmakl = ymie_sigmakl(sigma, ngroup)                                       
    
    epsilonkl1 = np.sqrt(np.outer(sigma**3,sigma**3))/(sigmakl**3)*np.sqrt(np.outer(epsilon,epsilon))

    eps_kl_arr = eps_kl*kb
    eps_kl_arr = np.nan_to_num(eps_kl_arr, nan=epsilonkl1)
    
    return eps_kl_arr

#EQ40
@partial(njit)
def ymie_Zki(niast,niki,Sk):
    
    Zki = np.einsum('ki,k,k->ki ', niki,niast,Sk)
    Zki = Zki / sum(Zki)
    return Zki

#EQ41
@partial(njit, static_argnames=['ngroup'] )
def ymie_sigmamed3(sigma, ngroup,niast,niki,Sk):
    Zki =ymie_Zki(niast,niki,Sk)
    sigmakl = ymie_sigmakl(sigma, ngroup)
    sigmamed3 =(np.einsum('ki,li,kl->i', Zki,Zki,sigmakl**3))
    return sigmamed3

#EQ27 article 2
@partial(njit, static_argnames=['ncomp','ngroup'] )
def ymie_sigmaijmed(sigma, ngroup,niast,niki,Sk,ncomp):
    sigmamed = ymie_sigmamed3(sigma, ngroup,niast,niki,Sk)**(1/3)
    sigmaijmed = (sigmamed.reshape((ncomp,1))+sigmamed)/2
    return sigmaijmed


#EQ42
@partial(njit, static_argnames=['ngroup'] )
def ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    Zki =ymie_Zki(niast,niki,Sk)
    dkl =ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    diimed3 =(np.einsum('ki,li,kl->i', Zki,Zki,dkl**3))
    return diimed3

#EQ44
@partial(njit, static_argnames=['ngroup'] )
def ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk):
    Zki =ymie_Zki(niast,niki,Sk)
    # epsilonkl = ymie_epsilonkl(sigma,epsilon,ngroup)
    epsilonmed =(np.einsum('ki,li,kl->i', Zki,Zki,epsilonkl))
    return epsilonmed

#EQ26 article 2
@partial(njit, static_argnames=['ncomp','ngroup'] )
def ymie_epsilonijmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk,ncomp):
    
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    sigmamed3 = ymie_sigmamed3(sigma, ngroup,niast,niki,Sk)
    sigmaijmed = ymie_sigmaijmed(sigma, ngroup,niast,niki,Sk,ncomp)
    
    epsilonijmed = np.sqrt(np.outer(sigmamed3,sigmamed3))/(sigmaijmed**3)*np.sqrt(np.outer(epsilonmed,epsilonmed))

    return epsilonijmed



#EQ45
@partial(njit, static_argnames=['ngroup'] )
def ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk):
    Zki =ymie_Zki(niast,niki,Sk)
    # lambkl = ymie_lambkl(lamb)
    lambmed =(np.einsum('ki,li,nkl->ni', Zki,Zki,lambkl))
    return lambmed

@partial(njit, static_argnames=['ngroup'] )
def ymie_lambmed_all(lamb,lambkl,ngroup,niast,niki,Sk):
    # Zki =ymie_Zki(niast,niki,Sk)
    # lambkl = ymie_lambkl(lamb)
    medlambmed = np.sum(ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk),axis=0)
    _2lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)*2
    
    # medlambmed = (np.einsum('ki,li,kl->i', Zki,Zki,medlambmed))
    return _2lambmed,medlambmed

# ---- par,x
#not in the article, just using to make the code faster
@partial(njit)
def ymie_xk(x,niast,niki,Sk):
    xk = np.einsum('i,ki,k,k->k ', x,niki,niast,Sk)
    
    return xk


#EQ15 ok!
@partial(njit)
def ymie_xsk(x,niast,niki,Sk):
    
    xk = ymie_xk(x,niast,niki,Sk)
    xsk = xk/sum(xk)
    
    return xsk

#EQ34 cnv The article does not use it.
@partial(njit, static_argnames=['ngroup'] )
def ymie_sigmax3(x,sigma, ngroup,niast,niki,Sk):
    
    sigmakl = ymie_sigmakl(sigma, ngroup)
    xsk = ymie_xsk(x,niast,niki,Sk)
    
    sigmax3 = np.einsum('k,l,kl-> ',xsk,xsk,sigmakl**3)
    
    return sigmax3



#EQ13 ok!
@partial(njit)
def ymie_rhos(rho,x,niast,niki,Sk):
    
    # rhos = np.einsum('i,ki,k,k-> ', x,niki,niast,Sk)*rho
    xk = ymie_xk(x,niast,niki,Sk)
    rhos = sum(xk)*rho
    
    return rhos



#EQ32 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_chikl(rho,x,sigma,lambkl, ngroup,niast,niki,Sk,phicoef):
    
    fm = ymie_fm(phicoef,lambkl)
    csixast = ymie_csixast(rho,x,sigma, ngroup,niast,niki,Sk)
    
    
    chikl = fm[0]*csixast +fm[1]*csixast**5 + fm[2]*csixast**8
    
    return chikl

#EQ 26-27 and EQ 57-58 cnv  
@partial(njit, static_argnames=['ngroup'] )
def ymie_csikleff(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)

    lambkl0 = np.ones_like(lambkl)
    invlamb = np.stack([lambkl0,1/lambkl,1/(lambkl**2),1/(lambkl**3)],axis=0)
    
    cmkl = np.einsum('mi,inkl->mnkl',c,invlamb)
    # cmkl2 = np.tensordot(c,invlamb, axes=(-1,0)) #5x slower, but works for more functions

    csikleff = cmkl[0]*csix +  cmkl[1]*csix**2 +  cmkl[2]*csix**3 + cmkl[3]*csix**4

    return csikleff

@partial(njit, static_argnames=['ngroup'] )
def ymie_csikleff_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    _2lambkl,medlambkl = ymie_lambkl_all(lambkl)
    
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    _2lambkl0 = np.ones_like(_2lambkl)
    medlambkl0 = np.ones_like(medlambkl)
    
    inv_2lamb = np.stack([_2lambkl0,1/_2lambkl,1/(_2lambkl**2),1/(_2lambkl**3)],axis=0)
    inv_medlamb = np.stack([medlambkl0,1/medlambkl,1/(medlambkl**2),1/(medlambkl**3)],axis=0)

    cmkl = np.tensordot(c,inv_2lamb, axes=(-1,0))
    cmkl_medlamb = np.tensordot(c,inv_medlamb, axes=(-1,0))

    
    csikleff_2lamb = cmkl[0]*csix +  cmkl[1]*csix**2 +  cmkl[2]*csix**3 + cmkl[3]*csix**4
    csikleff_medlamb = cmkl_medlamb[0]*csix +  cmkl_medlamb[1]*csix**2 +  cmkl_medlamb[2]*csix**3 + cmkl_medlamb[3]*csix**4
    return csikleff_2lamb, csikleff_medlamb



@partial(njit, static_argnames=['ngroup'] )
def ymie_csimedeff(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    lambmed0 = np.ones_like(lambmed)
    invlamb = np.stack([lambmed0,1/lambmed,1/(lambmed**2),1/(lambmed**3)],axis=0)
    
    cmkl = np.einsum('mi,ikl->mkl',c,invlamb)
    # cmkl = np.tensordot(c,invlamb, axes=(-1,0)) 
    
    csikleff = cmkl[0]*csix +  cmkl[1]*csix**2 +  cmkl[2]*csix**3 + cmkl[3]*csix**4

    return csikleff 

@partial(njit, static_argnames=['ngroup'] )
def ymie_csimedeff_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    _2lambmed,medlambmed = ymie_lambmed_all(lamb,lambkl,ngroup,niast,niki,Sk)
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    _2lambmed0 = np.ones_like(_2lambmed)
    medlambmed0 = np.ones_like(medlambmed)
    
    invlamb_2lamb = np.stack([_2lambmed0,1/_2lambmed,1/(_2lambmed**2),1/(_2lambmed**3)],axis=0)
    invlamb_medlambmed = np.stack([medlambmed0,1/medlambmed,1/(medlambmed**2),1/(medlambmed**3)],axis=0)

    cmkl_2lamb = np.einsum('mi,inj->mnj',c,invlamb_2lamb)
    cmkl_medlambmed = np.einsum('mi,ij->mj',c,invlamb_medlambmed)
        
    csikleff_2lamb = cmkl_2lamb[0]*csix +  cmkl_2lamb[1]*csix**2 +  cmkl_2lamb[2]*csix**3 + cmkl_2lamb[3]*csix**4
    csikleff_medlamb = cmkl_medlambmed[0]*csix +  cmkl_medlambmed[1]*csix**2 +  cmkl_medlambmed[2]*csix**3 + cmkl_medlambmed[3]*csix**4

    return csikleff_2lamb, csikleff_medlamb


#EQ35 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_csixast(rho,x,sigma,ngroup,niast,niki,Sk):
    
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    sigmakl = ymie_sigmakl(sigma, ngroup)
    xsk = ymie_xsk(x,niast,niki,Sk)
    sigmax3 = np.einsum('k,l,kl-> ',xsk,xsk,sigmakl**3)
    csixast = sigmax3*pi*rhos/6

    return csixast

# ---- par,x,T

#EQ 24 cnv e ok ??
@partial(njit, static_argnames=['ngroup'] )
def ymie_Ikl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup):
    
    # lambkl = ymie_lambkl(lamb)
    x0kl = ymie_x0kl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    
    I = (1-x0kl**(3-lambkl))/(lambkl-3)
    
    return I

@partial(njit, static_argnames=['ngroup'] )
def ymie_Ikl_all(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup):
    _2lambkl,medlambkl = ymie_lambkl_all(lambkl)
    x0kl = ymie_x0kl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    
    Ikl_2lamb = (1-x0kl**(3-_2lambkl))/(_2lambkl-3)
    Ikl_medlamb = (1-x0kl**(3-medlambkl))/(medlambkl-3)
    
    return Ikl_2lamb, Ikl_medlamb



@partial(njit, static_argnames=['ngroup'] )
def ymie_Iiimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    x0iimed = ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    I = (1-x0iimed**(3-lambmed))/(lambmed-3)
    
    return I

@partial(njit, static_argnames=['ngroup'] )
def ymie_Iiimed_all(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    _2lambmed,medlambmed = ymie_lambmed_all(lamb,lambkl,ngroup,niast,niki,Sk)
    x0iimed = ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    Iiimed_2lamb   = (1-x0iimed**(3-_2lambmed))/(_2lambmed-3)
    Iiimed_medlamb = (1-x0iimed**(3-medlambmed))/(medlambmed-3)
    
    return Iiimed_2lamb,Iiimed_medlamb



#EQ 23 cnv e ok ??
@partial(njit, static_argnames=['ngroup'] )
def ymie_Jkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup):
    
    x0kl = ymie_x0kl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    # lambkl = ymie_lambkl(lamb)
    J = (1-x0kl**(4-lambkl)*(lambkl-3) + x0kl**(3-lambkl)*(lambkl-4))/((lambkl-3)*(lambkl-4))

    return J

@partial(njit, static_argnames=['ngroup'] )
def ymie_Jkl_all(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup):
    _2lambkl,medlambkl = ymie_lambkl_all(lambkl)
    x0kl = ymie_x0kl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)

    Jkl_2lamb = (1-x0kl**(4-_2lambkl)*(_2lambkl-3) + x0kl**(3-_2lambkl)*(_2lambkl-4))/((_2lambkl-3)*(_2lambkl-4))
    Jkl_medlamb = (1-x0kl**(4-medlambkl)*(medlambkl-3) + x0kl**(3-medlambkl)*(medlambkl-4))/((medlambkl-3)*(medlambkl-4))
    
    return Jkl_2lamb,Jkl_medlamb



@partial(njit, static_argnames=['ngroup'] )
def ymie_Jiimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    x0iimed = ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    J = (1-x0iimed**(4-lambmed)*(lambmed-3) + x0iimed**(3-lambmed)*(lambmed-4))/((lambmed-3)*(lambmed-4))

    return J

@partial(njit, static_argnames=['ngroup'] )
def ymie_Jiimed_all(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    _2lambmed,medlambmed = ymie_lambmed_all(lamb,lambkl,ngroup,niast,niki,Sk)
    x0iimed = ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    Jiimed_2lamb    = (1-x0iimed**(4-_2lambmed)*(_2lambmed-3) + x0iimed**(3-_2lambmed)*(_2lambmed-4))/((_2lambmed-3)*(_2lambmed-4))
    Jiimed_medlamb  = (1-x0iimed**(4-medlambmed)*(medlambmed-3) + x0iimed**(3-medlambmed)*(medlambmed-4))/((medlambmed-3)*(medlambmed-4))

    return Jiimed_2lamb,Jiimed_medlamb


#EQ10 ok!
@partial(njit, static_argnames=['ngroup'] )
def ymie_dkk(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup):

    npontos = 201
    r = np.linspace(sigma/npontos*1e-5,sigma,npontos)
    Ckk = ymie_Ckl(lambkl).diagonal()
    f = 1 - np.exp(-Ckk*epsilon*((sigma/r)**lamb[1] - (sigma/r)**lamb[0])/(kb*T))
    
    def simps(y,x,npontos):
        h=(x[npontos-1]-x[0])/(npontos-1)
        mataux = nnp.ones([npontos,ngroup])
        odd = nnp.arange(1,npontos-1,2)
        even = nnp.arange(2,npontos-1,2)
        mataux[even,:] = 2
        mataux[odd,:] = 4
        v = sum(y*mataux)
        v = h/3*v
        return v
    
    dkk = simps(f,r,npontos) + sigma/npontos*1e-5
    
    return dkk

#EQ37 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_a3(rho,x,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,phicoef):
    
    a3kl = ymie_a3kl(rho,x,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,phicoef)
    xsk = ymie_xsk(x,niast,niki,Sk)
    
    a3 = np.einsum('k,l,kl-> ', xsk,xsk,a3kl)
    
    return a3

#EQ38 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_a3kl(rho,x,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,phicoef):
    
    fm = ymie_fm(phicoef,lambkl)
    csixast = ymie_csixast(rho,x,sigma, ngroup,niast,niki,Sk)
    
    a3kl = -epsilonkl**3*fm[3]*csixast*np.exp(fm[4]*csixast+fm[5]*csixast**2)
    
    return a3kl

#EQ72 ok!
@partial(njit, static_argnames=['ngroup'] )
def ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup):
    
    dkk = ymie_dkk(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)

    dkl = (dkk.reshape((ngroup,1))+dkk)/2
    
    return dkl

@partial(njit, static_argnames=['ngroup'] )
def ymie_x0kl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup):

    sigmakl = ymie_sigmakl(sigma, ngroup)
    dkl = ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    x0kl = sigmakl/dkl
    
    return x0kl

@partial(njit, static_argnames=['ngroup'] )
def ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    sigmamed = np.cbrt(ymie_sigmamed3(sigma,ngroup,niast,niki,Sk))
    diimed = np.cbrt(ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))

    x0iimed = sigmamed/diimed
    
    return x0iimed



# ---- par,rho,x,T

#EQ33 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_Khs(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    Khs = (1-csix)**4/(1+4*csix+4*csix**2-4*csix**3+csix**4)
    
    return Khs


#EQ 22 ok!
@partial(njit, static_argnames=['ngroup'] )
def ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    
    xsk = ymie_xsk(x,niast,niki,Sk)
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    dkl = ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    
    dx3 = np.einsum('k,l,kl-> ',xsk,xsk,dkl**3)
    csix = pi*rhos/6*dx3
    
    return csix

#EQ14 ok!
@partial(njit, static_argnames=['ngroup'] )
def ymie_csim(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,niast,niki,Sk,ngroup):
    
    xsk = ymie_xsk(x,niast,niki,Sk)
    dkk = ymie_dkk(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    dkk0 = np.ones_like(dkk)
    powd = np.stack([dkk0,dkk,dkk**2,dkk**3],axis=1)
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    
    csim = (xsk)@powd*pi*rhos/6
    
    return csim

# #EQ11 - not needed
# @partial(njit, static_argnames=['ncomp','ngroup'] )
# def ymie_A_hs_old(rho,T,x,xk,a_hs):
    
#     a_hs = ymie_a_hs(rho,T,x)
#     A_hs = xk*a_hs
    
#     return A_hs
        

# #EQ12 - not needed
# @partial(njit, static_argnames=['ncomp','ngroup'] )
# def ymie_a_hs(rho,ngroup,sigma,epsilon,dkk):
    
#     csim = ymie_csim(rho,ngroup,sigma,epsilon,dkk,ngroup)
#     rhos = ymie_rhos(rho)
#     a_hs =  6/(pi*rhos)*(((csim[2]**3)/(csim[3]**2) - csim[0])*np.log(1-csim[3]) + 
#                          3*csim[1]*csim[2]/(1-csim[3]) + csim[2]**3/(csim[3]*((1-csim[3])**2)) )
    
#     return a_hs
    
#EQ16 ok!
@partial(njit, static_argnames=['ngroup'] )
def ymie_A_hs(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,niast,niki,Sk,ngroup):
    
    csi0,csi1,csi2,csi3 = ymie_csim(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,niast,niki,Sk,ngroup)
    
    #gustavo usa outra , que nao da o mesmo resultado e é dif do artigo
    A_hs =  6/(pi*rho)*(((csi2**3)/(csi3**2) - csi0)*np.log(1-csi3) + 
            3*csi1*csi2/(1-csi3) + csi2**3/(csi3*((1-csi3)**2)) ) 
    
    # A_hs =  (((csi2**3)/(csi3**2) - csi0)*np.log(1-csi3) + 
    #         3*csi1*csi2/(1-csi3) + csi2**3/(csi3*((1-csi3)**2)) )/csi0

    return A_hs



#EQ17 cnv 
@partial(njit, static_argnames=['ngroup'] )
def ymie_A1(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    
    xk = ymie_xk(x,niast,niki,Sk)
    a1 = ymie_a1(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    A1 = 1/(kb*T)*sum(xk)*a1
    
    return A1

#EQ18 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_a1(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):

    a1kl = ymie_a1kl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    xsk = ymie_xsk(x,niast,niki,Sk)
    
    a1 = np.einsum('k,l,kl-> ', xsk,xsk,a1kl)

    return a1
    
#EQ30 cnv
# @partial(njit, static_argnames=['ncomp','ngroup'] )
# def ymie_a1kl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,csix):
    
#     dkl =ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
#     sigmakl = ymie_sigmakl(sigma, ngroup)
#     epsilonkl = ymie_epsilonkl(sigma,epsilon,ngroup)
#     lambkl = ymie_lambkl(lamb)
#     Ckl = ymie_Ckl(lambkl)
#     x0kl = ymie_x0kl(sigmakl,dkl)
#     Bkl = ymie_Bkl(rho,x,T,epsilonkl,sigmakl,dkl,lambkl,ngroup,niast,niki,Sk,csix)
#     a1skl = ymie_a1skl(rho,x,epsilonkl,sigmakl,dkl,lambkl,ngroup,niast,niki,Sk,c,csix)
    
#     a1kl =  Ckl*( x0kl**lamb[0]*(a1skl[0] +  Bkl[0]) - x0kl**lamb[1]*(a1skl[1] + Bkl[1]) )
    
#     return a1kl

#EQ19 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_a1kl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    # lambkl = ymie_lambkl(lamb)
    Ckl = ymie_Ckl(lambkl)
    x0kl = ymie_x0kl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    Bkl = ymie_Bkl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    a1skl = ymie_a1skl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    a1kl =  Ckl*( x0kl**lambkl[0]*(a1skl[0] +  Bkl[0]) - x0kl**lambkl[1]*(a1skl[1] + Bkl[1]) )
    
    return a1kl

#EQ55
@partial(njit, static_argnames=['ngroup'] )
def ymie_a1medii(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    Ciimed = ymie_Ciimed(lamb,lambkl,ngroup,niast,niki,Sk)
    x0iimed = ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    Biimed = ymie_Biimed(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    a1siimed = ymie_a1siimed(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    a1medii =  Ciimed*( x0iimed**lambmed[0]*(a1siimed[0] +  Biimed[0]) - x0iimed**lambmed[1]*(a1siimed[1] + Biimed[1]) )
    
    return a1medii


#EQ 20 cnv e ok ??
@partial(njit, static_argnames=['ngroup'] )
def ymie_Bkl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,):
    # epsilonkl = ymie_epsilonkl(sigma,epsilon,ngroup)
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    dkl = ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    Ikl = ymie_Ikl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    Jkl = ymie_Jkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    
    Bkl = 2*pi*rhos*dkl**3*epsilonkl*((1-csix/2)/((1-csix)**3)*Ikl - 9*csix*(1+csix)/(2*(1-csix)**3)*Jkl )
    
    return Bkl

#EQ 20 cnv e ok ??
@partial(njit, static_argnames=['ngroup'] )
def ymie_Bkl_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    # epsilonkl = ymie_epsilonkl(sigma,epsilon,ngroup)
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    dkl = ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    
    Ikl_2lamb,Ikl_medlamb = ymie_Ikl_all(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    Jkl_2lamb,Jkl_medlamb = ymie_Jkl_all(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    
    
    Bkl_2lamb = 2*pi*rhos*dkl**3*epsilonkl*((1-csix/2)/((1-csix)**3)*Ikl_2lamb - 9*csix*(1+csix)/(2*(1-csix)**3)*Jkl_2lamb )
    Bkl_medlamb = 2*pi*rhos*dkl**3*epsilonkl*((1-csix/2)/((1-csix)**3)*Ikl_medlamb - 9*csix*(1+csix)/(2*(1-csix)**3)*Jkl_medlamb )
    return Bkl_2lamb,Bkl_medlamb


@partial(njit, static_argnames=['ngroup'] )
def ymie_Biimed(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk ):
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    diimed3 = (ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    Iiimed = ymie_Iiimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    Jiimed = ymie_Jiimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    
    Biimed = 2*pi*rhos*diimed3*epsilonmed*((1-csix/2)/((1-csix)**3)*Iiimed - 9*csix*(1+csix)/(2*(1-csix)**3)*Jiimed )
    
    return Biimed

@partial(njit, static_argnames=['ngroup'] )
def ymie_Biimed_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    diimed3 = (ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    rhos = ymie_rhos(rho,x,niast,niki,Sk)

    Iiimed_2lamb,Iiimed_medlamb  = ymie_Iiimed_all(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    Jiimed_2lamb,Jiimed_medlamb  = ymie_Jiimed_all(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    
    Biimed_2lamb    = 2*pi*rhos*diimed3*epsilonmed*((1-csix/2)/((1-csix)**3)*Iiimed_2lamb - 9*csix*(1+csix)/(2*(1-csix)**3)*Jiimed_2lamb )
    Biimed_medlamb  = 2*pi*rhos*diimed3*epsilonmed*((1-csix/2)/((1-csix)**3)*Iiimed_medlamb - 9*csix*(1+csix)/(2*(1-csix)**3)*Jiimed_medlamb )
    
    return Biimed_2lamb,Biimed_medlamb

#EQ25 conv
@partial(njit, static_argnames=['ngroup'] )
def ymie_a1skl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c ):

    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    dkl = ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    csikleff = ymie_csikleff(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    a1skl =  -2*pi*rhos*(epsilonkl*dkl**3/(lambkl-3))*((1-csikleff/2)/((1-csikleff)**3))
    
    return a1skl

@partial(njit, static_argnames=['ngroup'] )
def ymie_a1skl_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    _2lambkl,medlambkl               = ymie_lambkl_all(lambkl)
    rhos                             = ymie_rhos(rho,x,niast,niki,Sk)
    dkl                              = ymie_dkl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    csikleff_2lamb, csikleff_medlamb = ymie_csikleff_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    
    a1skl_2lamb = -2*pi*rhos*(epsilonkl*dkl**3/(_2lambkl-3))*((1-csikleff_2lamb/2)/((1-csikleff_2lamb)**3))
    a1skl_medlamb = -2*pi*rhos*(epsilonkl*dkl**3/(medlambkl-3))*((1-csikleff_medlamb/2)/((1-csikleff_medlamb)**3))
    
    return a1skl_2lamb,a1skl_medlamb



@partial(njit, static_argnames=['ngroup'] )
def ymie_a1siimed(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    diimed3 = (ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))
    csimedeff = ymie_csimedeff(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    
    a1siimed = -2*pi*rhos*(epsilonmed*diimed3/(lambmed  -3))*((1-csimedeff/2)/((1-csimedeff)**3))
    
    return a1siimed

@partial(njit, static_argnames=['ngroup'] )
def ymie_a1siimed_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    _2lambmed,medlambmed = ymie_lambmed_all(lamb,lambkl,ngroup,niast,niki,Sk)
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    diimed3 = (ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))
    
    csimedeff_2lamb,csimedeff_medlamb = ymie_csimedeff_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    
    a1siimed_2lamb = -2*pi*rhos*(epsilonmed*diimed3/(_2lambmed-3))*((1-csimedeff_2lamb/2)/((1-csimedeff_2lamb)**3))
    a1siimed_medlamb = -2*pi*rhos*(epsilonmed*diimed3/(medlambmed-3))*((1-csimedeff_medlamb/2)/((1-csimedeff_medlamb)**3))
    
    return a1siimed_2lamb,a1siimed_medlamb

#EQ28 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_A2(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef):
    
    a2 = ymie_a2(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef)
    
    A2 = 1/((kb*T)**2)*np.einsum('i,ki,k,k-> ', x,niki,niast,Sk)*a2
    
    return A2

#EQ29 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_a2(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef):
    
    # csix = ymie_csix(rho,x,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    a2kl = ymie_a2kl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef)
    xsk = ymie_xsk(x,niast,niki,Sk)
    
    a2 = np.einsum('k,l,kl-> ', xsk,xsk,a2kl)
    
    return a2

#EQ30 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_a2kl(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef):

    x0kl = ymie_x0kl(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup)
    Ckl = ymie_Ckl(lambkl)
    Khs = ymie_Khs(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    Bkl_2lamb,Bkl_medlamb =  ymie_Bkl_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    a1skl_2lamb,a1skl_medlamb = ymie_a1skl_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    chikl = ymie_chikl(rho,x,sigma,lambkl, ngroup,niast,niki,Sk,phicoef)
    
    mataux1 = sum( x0kl**(2*lambkl)*(a1skl_2lamb + Bkl_2lamb))
    mataux2 = x0kl**(lambkl[1]+lambkl[0])*(a1skl_medlamb + Bkl_medlamb)
    a2kl = Khs*(1+chikl)*epsilonkl*Ckl**2*(mataux1-2*mataux2)/2

    return  a2kl

@partial(njit, static_argnames=['ngroup'] )
def ymie_a2iimed(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef):
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    # diimed = np.cbrt(ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))
    x0iimed =  ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    Ciimed = ymie_Ciimed(lamb,lambkl,ngroup,niast,niki,Sk)
    Khs = ymie_Khs(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    Biimed_2lamb,Biimed_medlamb = ymie_Biimed_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    a1siimed_2lamb,a1siimed_medlamb = ymie_a1siimed_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    
    mataux1 = sum( x0iimed**(2*lambmed)*(a1siimed_2lamb + Biimed_2lamb))
    mataux2 = x0iimed**(lambmed[1]+lambmed[0])*(a1siimed_medlamb + Biimed_medlamb)
    a2kl1 = 0.5*Khs*epsilonmed*Ciimed**2*(mataux1-2*mataux2)

    return  a2kl1


#EQ36 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_A3(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,phicoef):
    
    a3 = ymie_a3(rho,x,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,phicoef)
    
    A3 = 1/((kb*T)**3)*np.einsum('i,ki,k,k-> ', x,niki,niast,Sk)*a3
    
    return A3

#EQ39 cnv
@partial(njit )
def ymie_fm(phicoef,lambkl):
    
    alphakl = ymie_alphakl(lambkl)
    alphakl0 = np.ones_like(alphakl)
    powalphakl = np.stack([alphakl0,alphakl,alphakl**2,alphakl**3],axis=0)
    
    num = np.einsum('mn,nkl->mkl',phicoef[:,0:4],powalphakl)
    den = np.einsum('mn,nkl->mkl',phicoef[:,4:7],powalphakl[1:])
    # fm = (phicoef[:,0:4]@powalphakl)#/ (1 +  (phicoef[:,4:7]@powalphakl[1:]))
    fm = num/(1+den)
    return fm[:-1]


#EQ46
@partial(njit, static_argnames=['ngroup'] )
def ymie_Achain(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef):
    
    lggmie = ymie_gmie(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef)
    
    Achain = - (  np.einsum('i,ki,k,k,i->', x,niki,niast,Sk,lggmie) - np.einsum('i,i->', x,lggmie))
    
    return Achain

#EQ47
@partial(njit, static_argnames=['ngroup'] )
def ymie_gmie(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef):
    
    ghs = ymie_ghs(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    g1 = ymie_g1(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    g2 = ymie_g2(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef)
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    beta = 1/(kb*T)
    
    lggmie = np.log(ghs)+((beta*epsilonmed*g1 + (beta*epsilonmed)**2*g2)/ghs)

    return lggmie#,ghs,g1,g2

#EQ48 ok!
@partial(njit, static_argnames=['ngroup'] )
def ymie_ghs(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    
    x0iimed =  ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    k0,k1,k2,k3 = ymie_km(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    ghs = np.exp(k0 + k1*x0iimed + k2*x0iimed**2 + k3*x0iimed**3)
    
    return ghs

#EQ49-52 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_km(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk):
    
    csix = ymie_csix(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    k0 = -np.log(1-csix) + (42*csix-39*csix**2+9*csix**3 - 2*csix**4)/(6*(1-csix)**3)
    k1 = (csix**4 + 6*csix**2-12*csix)/(2*(1-csix)**3)
    k2 = -3*csix**2/(8*(1-csix)**2)
    k3 = (-csix**4 + 3*csix**2 + 3*csix)/(6*(1-csix)**3)
    
    return k0,k1,k2,k3

#EQ53 ODD
@partial(njit, static_argnames=['ngroup'] )
def ymie_g1(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    diimed = np.cbrt(ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))
    x0iimed =  ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    Ciimed = ymie_Ciimed(lamb,lambkl,ngroup,niast,niki,Sk)
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)

    Biimed = ymie_Biimed(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    a1siimed = ymie_a1siimed(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    da1iimed_drhos =  jacfwd( ymie_a1medii,argnums=(0))(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)/(rhos/rho)
    
    Mat1 = Ciimed*lambmed*x0iimed**lambmed*(a1siimed + Biimed)/rhos
    g1 = (3*da1iimed_drhos + (Mat1[1]-Mat1[0]))/(2*pi*epsilonmed*diimed**3)
    
    return g1

#EQ59
@partial(njit, static_argnames=['ngroup'] )
def ymie_g2(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef):
    
    ycii = ymie_ycii(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,phicoef)
    g2mca = ymie_g2mca(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef)
    g2 = (1+ycii)*g2mca
    
    return g2

#EQ60 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_ycii(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,phicoef):
    
    alphaiimed = ymie_alphaiimed(lamb,lambkl,ngroup,niast,niki,Sk)
    csixast = ymie_csixast(rho,x,sigma, ngroup,niast,niki,Sk)
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    beta = 1/(kb*T)
    
    theta = np.exp(beta*epsilonmed)-1
    ycii = phicoef[6,0]*(-np.tanh( phicoef[6,1]*( phicoef[6,2] - alphaiimed)) + 1)*csixast*theta*np.exp(phicoef[6,3]*csixast + phicoef[6,4]*csixast**2  )

    return ycii 

#EQ62 cnv
@partial(njit, static_argnames=['ngroup'] )
def ymie_g2mca(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef):
    
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    # sigmamed = (ymie_sigmamed3(sigma, ngroup,niast,niki,Sk))**(1/3)
    diimed3 = (ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))
    x0iimed =  ymie_x0iimed(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
    Ciimed = ymie_Ciimed(lamb,lambkl,ngroup,niast,niki,Sk)
    epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
    # csix = ymie_csix(rho,x,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    Khs = ymie_Khs(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    
    Biimed_2lamb,Biimed_medlamb = ymie_Biimed_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
    a1siimed_2lamb,a1siimed_medlamb = ymie_a1siimed_all(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
 
    Mat1 = epsilonmed*Khs*Ciimed**2*lambmed*x0iimed**(2*lambmed)*(a1siimed_2lamb + Biimed_2lamb)/rhos
    Mat2 = epsilonmed*Khs*Ciimed**2*(lambmed[1]+lambmed[0])*x0iimed**(lambmed[1]+lambmed[0])*(a1siimed_medlamb + Biimed_medlamb)/rhos
    
    diffa2ii = jacfwd( ymie_a2iimed,argnums=0)(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c,phicoef)/(rhos/rho)

    g2mca = (3*diffa2ii + Mat2 -(Mat1[0]+Mat1[1]))/(2*pi*epsilonmed**2*diimed3)
    
    return  g2mca

@partial(njit, static_argnames=['ngroup'] )
def ymie_diffa2ii(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
    
    def a2ii_chii(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c):
        csix = ymie_csix(rho,x,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
        sigmamed = np.cbrt(ymie_sigmamed3(sigma, ngroup,niast,niki,Sk))
        diimed = np.cbrt(ymie_diimed3(x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk))
        x0iimed =  (ymie_x0kl(sigmamed,diimed))
        lambmed = ymie_lambmed(lamb,lambkl,ngroup,niast,niki,Sk)
        epsilonmed = ymie_epsilonmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk)
        Ciimed = ymie_Ckl(lambmed)
        Khs = ymie_Khs(rho,x,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk)
        Bkl_2lamb = ymie_Bkl(rho,x,T,epsilonmed,sigmamed,diimed,lambmed*2,ngroup,niast,niki,Sk,csix)
        Bkl_medlamb = ymie_Bkl(rho,x,T,epsilonmed,sigmamed,diimed,(lambmed[1]+lambmed[0]),ngroup,niast,niki,Sk,csix)
        a1skl_2lamb = ymie_a1skl(rho,x,epsilonmed,sigmamed,diimed,lambmed*2,ngroup,niast,niki,Sk,c,csix)
        a1skl_medlamb = ymie_a1skl(rho,x,epsilonmed,sigmamed,diimed,(lambmed[1]+lambmed[0]),ngroup,niast,niki,Sk,c,csix)
        
        mataux1 = sum( x0iimed**(2*lambmed)*(a1skl_2lamb + Bkl_2lamb))
        mataux2 = x0iimed**(lambmed[1]+lambmed[0])*(a1skl_medlamb + Bkl_medlamb)
        a2ii_chii = 0.5*Khs*epsilonmed*Ciimed**2*(mataux1-2*mataux2)
        return a2ii_chii
    
    diffa2ii = jacfwd(a2ii_chii,argnums=(0))(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    return diffa2ii

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_A_res(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    A_hs = ymie_A_hs(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,niast,niki,Sk,ngroup)
    A1 = ymie_A1(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,c)
    A2 = ymie_A2(rho,x,T, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef)
    A3 = ymie_A3(rho,x,T,epsilon,epsilonkl,sigma,lamb,lambkl,ngroup,niast,niki,Sk,phicoef)
    Achain = ymie_Achain(rho,x,T, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef)
    Aasc = ymie_Aasc(rho,x,T,ncomp,epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    A_res = A_hs + A1 + A2 + A3 +Achain + Aasc
    
    return A_res
    
@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_mu_res(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    Ares = ymie_A_res(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    jaca_res = jacfwd(ymie_A_res,argnums=[0,1])(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    soma = np.einsum('j,j->', x,jaca_res[1])
    Z_res = rho*jaca_res[0]
    mu = jaca_res[1] -soma + Z_res + Ares
    return  mu,Z_res

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_phi(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    mu,Z_res = ymie_mu_res(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)

    lnphi = mu - np.log(Z_res+1)
    
    phi = np.exp(lnphi)
    return phi

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_phi_xTP(x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    rhoL = ymie_densL(x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    phiL = ymie_phi(rhoL,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    rhoV = ymie_densV(x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    phiV = ymie_phi(rhoV,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    return phiL,phiV,rhoL,rhoV

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_resi_sat_phis(T,P,indxcomp,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    x = np.zeros(ncomp)
    x = x.at[indxcomp].set(1.0)
    phiL,phiV,_,_ = ymie_phi_xTP(x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    phiL = phiL[indxcomp]
    phiV = phiV[indxcomp]
    return (phiL - phiV)

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_resi_phi(T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc,indxcomp):
    x = np.zeros(ncomp)
    x = x.at[indxcomp].set(1.0)
    phiL,phiV = ymie_phi_xTP(x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    phiL = phiL[indxcomp]
    phiV = phiV[indxcomp]
    return (phiL - phiV)

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_dresi_sat_phis_dP(T,P,indxcomp,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    x = np.zeros(ncomp)
    x = x.at[indxcomp].set(1.0)
    dresi_phiL_dP = jacfwd(ymie_resi_sat_phis,argnums=1)(T,P,indxcomp,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    # dresi_phiV_dP = jacfwd(ymie_phi_xTP,argnums=1)(x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc,indxcomp,phase=1)
    
    return dresi_phiL_dP


@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_Z(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):

    da_drho = jacfwd(ymie_A_res,argnums=0)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)

    Z = 1+rho*da_drho

    return Z

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
# EQ A.23 ok! -> there is no avogadro number in the paper, only volume convertion in intead
def ymie_P(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    Z = ymie_Z(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    P = Z*kb*T*rho
    
    return P

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
# EQ A.23 ok! -> there is no avogadro number in the paper, only volume convertion in intead
def ymie_logP(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    P = ymie_P(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    
    return (np.log(P))

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
# EQ A.23 ok! -> there is no avogadro number in the paper, only volume convertion in intead
def ymie_dlogPdrho(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    dpdrho = jacfwd(ymie_logP,argnums=0)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    return dpdrho

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
# EQ A.23 ok! -> there is no avogadro number in the paper, only volume convertion in intead
def ymie_dPdrho(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    dpdrho = jacfwd(ymie_P,argnums=0)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    return dpdrho

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
# EQ A.23 ok! -> there is no avogadro number in the paper, only volume convertion in intead
def ymie_dP2d2rho(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    dP2d2rho = jacfwd(ymie_dPdrho,argnums=0)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    return dP2d2rho

# ---- Asc

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_Aasc(rho,x,T,ncomp,epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    XAsc = ymie_X_tan(rho,x,T,ncomp,epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    x = np.take(x, idxasc, axis=0)
    niki  =  np.take(niki, idkasc, axis=0)
    niki  =  np.take(niki, idxasc, axis=1)
    nka = np.take(nka, idkasc, axis=0)
    
    Aasc = np.einsum('i,ki,ka,ika->', x,niki,nka,(np.log(XAsc) + (1-XAsc)/2) )
    return Aasc
 

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_X_tan(rho,x,T,ncomp,epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    delta = ymie_delta(rho,x,T,ncomp,epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk,eklAB, kappaklAB,cpq,ngroupasc,idxasc)
    
    X_A = np.ones([ncompasc,ngroupasc,nsite])*0.8
    
    x = np.take(x, idxasc, axis=0)
    niki  =  np.take(niki, idkasc, axis=0)
    niki  =  np.take(niki, idxasc, axis=1)
    nka = np.take(nka, idkasc, axis=0)

    
    X_A_old = X_A*(0.2)
    fabricio = 0
    
    
    def itX_A(args): #body fun
        X_A = args[1]
        fabricio=args[2]
        X_A_old = X_A*1
        sum1 = np.einsum('j,lj,lb,jlb,ijklab->ika',x,niki,nka,X_A_old,delta)
        X_A = 1./(1. + rho*sum1)
        fabricio = fabricio +1
        return [X_A_old,X_A,fabricio]
    
    def cond_fun(args):
        X_A_old = args[0]
        X_A = args[1]
        it=args[2]
        resmin = 3.e-10
        itMAX = 1e+5
        dif = ((X_A-X_A_old)/X_A)**2
        res = nnp.max(dif)*1
        cond1 = res>resmin
        cond2 = it<itMAX
        return np.logical_and(cond1,cond2) 
    
    _,X_A,fabricio = lax.while_loop(cond_fun, itX_A, [X_A_old,X_A,fabricio])
    return X_A#,fabricio

@partial(njit, static_argnames=['ncomp','ngroup','ngroupasc'])
def ymie_delta(rho,x,T,ncomp,epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk,eklAB, kappaklAB,cpq,ngroupasc,idxasc):

    Fklab = ymie_Fklab(T,eklAB)
    Iij = ymie_Iij(rho,x,T,epsilon,epsilonkl,sigma, ngroup,niast,niki,Sk,cpq,ncomp,idxasc)
    delta = np.einsum('klab,klab,ij->ijklab',Fklab,kappaklAB,Iij)

    return delta

@partial(njit)
def ymie_Fklab(T,eklAB):
    # eklAB =  ymie_eklAB(eklAB,ngroupasc)
    Fklab = np.exp(eklAB/(kb*T))-1
    
    return Fklab
    


@partial(njit, static_argnames=['ncomp','ngroup'])
def ymie_Iij(rho,x,T,epsilon,epsilonkl,sigma, ngroup,niast,niki,Sk,cpq,ncomp,idxasc):
    
    sigmax3 = ymie_sigmax3(x,sigma, ngroup,niast,niki,Sk)
    rhos = ymie_rhos(rho,x,niast,niki,Sk)
    rhossig = np.ones(11)*rhos*sigmax3
    powp = np.arange(11)
    powq = powp[:,np.newaxis,np.newaxis]
    
    powrhossig = rhossig**powp
    epsilonijmed = ymie_epsilonijmed(epsilon,epsilonkl,sigma,ngroup,niast,niki,Sk,ncomp)/kb
    kbT_eps = (T/epsilonijmed)
    powkbT_eps = np.repeat(kbT_eps[np.newaxis],11,axis=0)
    powkbT_eps = powkbT_eps**powq
    
    powkbT_eps =  np.take(powkbT_eps, idxasc, axis=1)
    powkbT_eps =  np.take(powkbT_eps, idxasc, axis=2)
    
    Iij = np.einsum('pq,p,qij->ij',cpq,powrhossig,powkbT_eps)

    return Iij

    
@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_kappaT(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    kappaT = rho* ( jacfwd(ymie_P,argnums=0)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc))
    
    return kappaT

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_alphaP(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    kappaT = ymie_kappaT(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    alphaP = - kappaT * ( jacfwd(ymie_P,argnums=2)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc))
    
    return alphaP

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_dAres_dT(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    dAres_dT = jacfwd(ymie_A_res,argnums=2)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    return  dAres_dT

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_d2Ares_dT2(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    d2Ares_dT2 = jacfwd(ymie_dAres_dT,argnums=2)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    return  d2Ares_dT2

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_Hres(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    da_dt = ymie_dAres_dT(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    Z = ymie_Z(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    Hres = Navo*kb*(-T*da_dt + Z - 1 )*T
    
    return Hres

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_Cvres(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    da_dt = ymie_dAres_dT(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    dda_ddT = ymie_d2Ares_dT2(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    # Cvres =  -T*(dda_ddT+ 2*Navo*kb*da_dt)
    Cvres = Navo*kb*(-T**2*dda_ddT - 2*T*da_dt)
    
    return Cvres

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_Cpres(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    Cvres =  ymie_Cvres(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    dP_drho = jacfwd(ymie_P,argnums=0)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    dP_dT = jacfwd(ymie_P,argnums=2)(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    Cpres = Cvres + T*Navo/(rho**2)*dP_dT**2/dP_drho  - kb*Navo
    
    return Cpres

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
# EQ A.23 ok! -> there is no avogadro number in the paper, only volume convertion in intead
def ymie_SoS(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc,M):
    #[m/s]
    dP_drho = ymie_dPdrho(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    
    MM = M*x*1e-3/Navo # kg/molecule
    SoS = np.sqrt(dP_drho/MM)
    
    return SoS

# @partial(njit, static_argnames=['ngroupmulti','idkmulti'])
# def ymie_xskl_multi(x,niast,niki,Sk, ngroupmulti,idkmulti):
    
#     xsk = ymie_xsk(x,niast,niki,Sk)
    
#     xsk = np.take(xsk, idkmulti, axis=0)
    
#     xskl = np.einsum('k,l->kl',xsk,xsk)
    
#     return xskl

# @partial(njit, static_argnames=['ngroupmulti','idkmulti'])
# def ymie_Qvsmu_multi(x,niast,niki,Sk, ngroupmulti,idkmulti,Qk,muk):
    
    
#     mumu = np.einsum('k,l->kl',muk,muk)**2
#     muQ = np.einsum('k,l->kl',Qk,muk)**2
#     QQ = np.einsum('k,l->kl',Qk,Qk)**2
    
#     return mumu,muQ,QQ


@partial(njit, static_argnames=[])
def ymie_xsk_multi(x,niast,niki,Sk,idkmulti):
    
    xsk = ymie_xsk(x,niast,niki,Sk)
    
    xsk = np.take(xsk, idkmulti, axis=0)
    
    xskl = np.einsum('k,l->kl',xsk,xsk)
    xsklm = np.einsum('k,l,m->klm',xsk,xsk,xsk)
    
    return xskl,xsklm
    
@partial(njit, static_argnames=[])
def ymie_Qvsmu_multi(x,niast,niki,Sk,idkmulti,Qk,muk):
    mumu = np.einsum('k,l->kl',muk,muk)**2
    muQ = np.einsum('k,l->kl',Qk,muk)**2
    QQ = np.einsum('k,l->kl',Qk,Qk)**2
    
    mumumu = np.einsum('k,l,m->klm',muk,muk,muk)**2
    mumuQ = np.einsum('k,l,m->klm',muk,muk,Qk)**2
    muQQ = np.einsum('k,l,m->klm',muk,Qk,Qk)**2
    QQQ = np.einsum('k,l,m->klm',Qk,Qk,Qk)**2

    mu2mu2QQ = np.einsum('k,l,k,l->kl',muk**2,muk**2,Qk,Qk)
    mu2QQ3 = np.einsum('k,k,l->kl',muk**2,Qk,Qk**3)

    
    return mumu,muQ,QQ,mumumu,mumuQ,muQQ,QQQ ,mu2mu2QQ ,mu2QQ3

@partial(njit, static_argnames=['ngroup'])
def ymie_sig_multi(sigma,ngroup,idkmulti):
    
    sigkl = ymie_sigmakl(sigma, ngroup)
    sigkl = np.take(sigkl, idkmulti, axis=0)
    sigkl = np.take(sigkl, idkmulti, axis=1)
    
    sigkl_lm_km  = np.einsum('kl,lm,km->klm',sigkl,sigkl,sigkl)*1e+30
    sigkl_lm2_km2  = np.einsum('kl,lm,km->klm',sigkl,sigkl**2,sigkl**2)*1e+50
    sigkl2_lm2_km3  = np.einsum('kl,lm,km->klm',sigkl**2,sigkl**2,sigkl**3)*1e+70
    
    return sigkl,sigkl_lm_km,sigkl_lm2_km2,sigkl2_lm2_km3


@partial(njit, static_argnames=['ngroup','ngroupmulti'])
def ymie_A_multi(rho,x,T,epsilonkl,sigma,ngroup, niast, niki, Sk,ngroupmulti,idkmulti,Qk,muk,C1,C2):
    
    xskl,xsklm                                         = ymie_xsk_multi(x,niast,niki,Sk,idkmulti) 
    mumu,muQ,QQ,mumumu,mumuQ,muQQ,QQQ,mu2mu2QQ ,mu2QQ3 = ymie_Qvsmu_multi(x,niast,niki,Sk,idkmulti,Qk,muk)
    sigkl,sigkl_lm_km,sigkl_lm2_km2,sigkl2_lm2_km3     = ymie_sig_multi(sigma,ngroup,idkmulti)
    
    epsilonkl = np.take(epsilonkl, idkmulti, axis=0)
    epsilonkl = np.take(epsilonkl, idkmulti, axis=1)
    
    rhoad = rho*sigkl**3
    Tad = kb*T/epsilonkl
     
    mataux = np.array([rhoad**2*np.log(Tad),rhoad**2,rhoad*np.log(Tad),rhoad,np.log(Tad),Tad**0])
    
    J = np.einsum('ij,jkl->ikl',C1,mataux)
    K = np.einsum('ij,jkl->ikl',C2,mataux)
    
    dens = rho/1e30
    sigkl = sigkl *1e+10
    
    A2_112 = -2/3*pi*dens*xskl*mumu*J[0]/(sigkl**3)/(T**2)

    A2_123 = -pi*dens*xskl*muQ*J[1]/(sigkl**5)/(T**2)
    A2_224 = -14/5*pi*dens*xskl*QQ*J[2]/(sigkl**7)/(T**2)
    
    A2 = np.sum( A2_112 + 2*A2_123 + A2_224)
    
    A3A_112_112_224  = 8/25*pi*dens/(T**2)*xskl/(sigkl**8)*mu2mu2QQ*J[3]
    A3A_112_123_213  = 8/75*pi*dens/(T**2)*xskl/(sigkl**8)*mu2mu2QQ*J[3]
    A3A_112_123_224  = 8/35*pi*dens/(T**2)*xskl/(sigkl**10)*mu2QQ3*J[4]
    A3A_224_224_224  = 144/245*pi*dens/(T**2)*xskl/(sigkl**12)*(QQ**(3/2))*J[5]
    
    A3B_112_112_112  = 32/135*pi**3*(14*pi/5)**(1/2)*dens**2/(T**3)*xsklm/(sigkl_lm_km)*mumumu*K[0]
    A3B_112_123_123  = 64/315*pi**3*(3*pi)**(1/2)*dens**2/(T**3)*xsklm/(sigkl_lm2_km2)*mumuQ*K[1]
    A3B_123_123_224  = -32/45*pi**3*(22*pi/63)**(1/2)*dens**2/(T**3)*xsklm/(sigkl2_lm2_km3)*muQQ*K[2]
    A3B_224_224_224  = 32/2.025*pi**3*(2.002*pi)**(1/2)*dens**2/(T**3)*xsklm/(sigkl_lm_km**3)*QQQ*K[3]
    
    
    A3A = np.sum(3*A3A_112_112_224 + 6*A3A_112_123_213 +6*A3A_112_123_224 + A3A_224_224_224)
    A3B = np.sum( A3B_112_112_112  + 3*A3B_112_123_123 + 3*A3B_123_123_224 + A3B_224_224_224)
    
    A0 = 0
    
    A_multi = A0 + A2*(1/(1-(A3A+A3B)/A2))
    
    return A_multi















@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def residuoL(rhoi,x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    Pcalc = ymie_P(rhoi,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    res = Pcalc - P 
    return res

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def dresdrhoL(rhoi,x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    dresdrhoL =  jacfwd(residuoL,argnums=0)(rhoi,x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    return dresdrhoL


@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def residuo(rhoi,x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    Pcalc = ymie_P(rhoi,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    res = (Pcalc / P - 1 )
    return res

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def dresdrho(rhoi,x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    dresdrho =  jacfwd(residuo,argnums=0)(rhoi,x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
    return dresdrho

    
    
@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_densV(x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    rhoV00 = 1e-11
    etaguessV = 1e-10
    sigmaimed3 = ymie_sigmamed3(sigma,ngroup,niast,niki,Sk)
    Zi = np.sum(np.einsum('ki,k,k->ki ', niki,niast,Sk),axis=0)
    soma = np.einsum('i,i,i-> ',x,Zi,sigmaimed3)
        
    rhoV0 = 6/pi*etaguessV/soma
    
    fabricio = 0
    
    def residuo(rho):
        Pcalc = ymie_P(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
        res0 = (Pcalc -P )
        return res0
    
    def dresdrho(rho):
        dresdrho =  jacfwd(residuo,argnums=0)(rho)
        return dresdrho
    
    def itrho(args): #body fun
        rho = args[1]
        fabricio=args[2]
        rho0 = rho*1
        
        res = residuo(rho0)
        dres = dresdrho(rho0)
       
        rho = (rho0 - res/dres)
        # id_print(rho)
        # id_print(rho0)
        # id_print(res)
        # id_print(dres)
        # id_print(fabricio)
        fabricio = fabricio +1
        return [rho0,rho,fabricio]
    
    def cond_fun(args):
        rho0 = args[0]
        rho = args[1]
        fabricio=args[2]
        tolmin = 1e-4
        itMAX = 20
        tol = abs((rho0-rho)/rho)
        cond1 = tol>tolmin
        cond2 = fabricio<itMAX

        return np.logical_and(cond1,cond2) 
    
    _,rho,fabricio = lax.while_loop(cond_fun, itrho, [rhoV00,rhoV0,fabricio])
    
    # L = itrho([rhoV00,rhoV0,fabricio])
    
    # L = itrho(L)
    
    return rho

   
@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_densL(x,T,P,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    rhoV00 = 1e-11
    etaguessV =  0.5
    sigmaimed3 = ymie_sigmamed3(sigma,ngroup,niast,niki,Sk)
    Zi = np.sum(np.einsum('ki,k,k->ki ', niki,niast,Sk),axis=0)
    soma = np.einsum('i,i,i-> ',x,Zi,sigmaimed3)
        
    rhoV0 = 6/pi*etaguessV/soma
    
    fabricio = 0
    
    def residuo(rho):
        Pcalc = ymie_P(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
        res0 = Pcalc -P 
        return res0
    
    def dresdrho(rho):
        dresdrho =  jacfwd(residuo,argnums=0)(rho)
        return dresdrho
    
    def itrho(args): #body fun
        rho = args[1]
        fabricio=args[2]
        rho0 = rho*1
        
        res = residuo(rho0)
        dres = dresdrho(rho0)
       
        rho = (rho0 - res/dres)
        # id_print(rho)
        # id_print(rho0)
        # id_print(res)
        # id_print(dres)
        # id_print(fabricio)
        fabricio = fabricio +1
        return [rho0,rho,fabricio]
    
    def cond_fun(args):
        rho0 = args[0]
        rho = args[1]
        fabricio=args[2]
        tolmin = 1e-4
        itMAX = 20
        tol = abs((rho0-rho)/rho)
        cond1 = tol>tolmin
        cond2 = fabricio<itMAX

        return np.logical_and(cond1,cond2) 
    
    _,rho,fabricio = lax.while_loop(cond_fun, itrho, [rhoV00,rhoV0,fabricio])
    
    # L = itrho([rhoV00,rhoV0,fabricio])
    
    # L = itrho(L)
    
    return rho

@partial(njit, static_argnames=['ncomp','ngroup','nsite','ngroupasc','ncompasc'])
def ymie_dens(x,T,P,phase,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc):
    
    
    def rhoV0guess(etaguess):
        # etaguess = 1e-10
        sigmaimed3 = ymie_sigmamed3(sigma,ngroup,niast,niki,Sk)
        Zi = np.sum(np.einsum('ki,k,k->ki ', niki,niast,Sk),axis=0)
        soma = np.einsum('i,i,i-> ',x,Zi,sigmaimed3)
            
        rhoV0 = 6/pi*etaguess/soma
        return rhoV0
    
    rhoV00 = 1e-11
    fabricio = 0
    
    def residuo(rho):
        Pcalc = ymie_P(rho,x,T,ncomp, epsilon,epsilonkl, sigma, lamb,lambkl, ngroup, niast, niki, Sk, c, phicoef,eklAB,kappaklAB,cpq,nka,nsite,ngroupasc,ncompasc,idxasc,idkasc)
        res0 = (Pcalc -P )
        return res0
    
    def dresdrho(rho):
        dresdrho =  jacfwd(residuo,argnums=0)(rho)
        return dresdrho
    
    def itrho(args): #body fun
        rho = args[1]
        fabricio=args[2]
        rho0 = rho*1
        
        res = residuo(rho0)
        dres = dresdrho(rho0)
       
        rho = (rho0 - res/dres)
        
        fabricio = fabricio +1
        return [rho0,rho,fabricio]
    
    def cond_fun(args):
        rho0 = args[0]
        rho = args[1]
        fabricio=args[2]
        tolmin = 1e-4
        itMAX = 150
        tol = abs((rho0-rho)/rho)
        cond1 = tol>tolmin
        cond2 = fabricio<itMAX

        return np.logical_and(cond1,cond2) 
    
    rhoV0 = lax.select(phase ==0,rhoV0guess(0.5),rhoV0guess(1e-10))
    
    _,rho,fabricio = lax.while_loop(cond_fun, itrho, [rhoV00,rhoV0,fabricio])
    
    return rho
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 12:46:11 2023

@author: antonio-pc
"""

#%%     IMPORTS
from saftymie10 import saftymie
from solubility import sol_T
from matplotlib import pyplot as plt
from input_eos import input_eos
import numpy as nnp
import numpy as np
from jax import numpy as np
import pandas as pd


#%% COMPONENTS 


CBD =   {    'CH3': 3  , 'CH2': 3  , 'aCH': 2  , 'aCCH2': 1  , 'aCCH': 1 , 
             'CH2=': 1 , 'cCH2': 2  , 'C': 2  , 'aCOH': 2    , 'cCH': 2  }
pentane = {  'CH3': 2  , 'CH2': 3  }
hexane = {  'CH3': 2  , 'CH2': 4  }
heptane = {  'CH3': 2  , 'CH2': 5  }
ciclopentane = {  'cCH2': 5  }
Benzene = {  'cCH2': 6  }
fenol = {  'aCH': 5, 'aCOH':1  }
difenol = {  'aCH': 4, 'aCOH':2  }
octane ={  'CH3': 2  , 'CH2': 6  }
isooctane ={  'CH3': 5  , 'CH2': 1,'CH':1,'C':1  }

comps = [CBD,pentane,hexane,heptane]#,isooctane]#,ciclopentane,Benzene,fenol,difenol,octane]
ncomp = len(comps)
names = ['CBD','pentane','hexane','heptane']#,'isooctane']#,'ciclopentane','Benzene','fenol','difenol','octane']

groups , niki = input_eos(comps)

deltaHSL = nnp.zeros(ncomp)
deltaHSL[0] = 28.4*1e+3
deltacpSL = nnp.zeros(ncomp)
deltacpSL[0]=100
TSL =  nnp.zeros(ncomp)
TSL[0] =341.4
eosmie = saftymie(groups,niki,deltaHSL=deltaHSL,deltacpSL=deltacpSL,TSL=TSL)


Tpentane = np.array([253.1, 258.1, 265.1, 269.1, 274.1, 280.1, 284.1, 289.1, 293.1, 298.1, 301.1, 296.1, 298.1, 300.1, 303.1, 304.6, 306.6, 308.1, 309.1])
xpentane = np.array([0.0024, 0.003, 0.0049, 0.006, 0.0075, 0.0124, 0.0159, 0.0357, 0.041, 0.0883, 0.2151, 0.0505, 0.085, 0.0901, 0.1318, 0.1974, 0.2349, 0.3093, 0.3751])

Thexane = np.array([253.1, 258.1, 262.1, 264.1, 269.1, 274.1, 279.1, 282.1, 285.1, 290.6, 294.1, 292.1, 293.1, 294.1, 298.1, 301.1, 302.1, 307.1, 310.1])
xhexane = np.array([0.0022, 0.0026, 0.0027, 0.0032, 0.0034, 0.0048, 0.007, 0.0105, 0.0121, 0.0222, 0.044, 0.0251, 0.0287, 0.0489, 0.0617, 0.0923, 0.1215, 0.2754, 0.3725])

Theptane = np.array([254.1, 255.1, 259.1, 264.1, 265.1, 269.1, 274.1, 279.1, 283.6, 288.1, 290.1, 292.1, 293.1, 294.1, 298.1, 301.1, 302.1, 307.1, 310.1])
xheptane = np.array([0.0024, 0.0026, 0.0032, 0.0038, 0.0041, 0.0053, 0.0074, 0.0084, 0.0142, 0.0178, 0.0222, 0.0251, 0.0287, 0.0489, 0.0617, 0.0923, 0.1215, 0.2754, 0.3725])

Tisooctane = np.array([253.1, 258.1, 267.1, 269.1, 270.6, 276.1, 279.1, 282.1, 285.1, 289.1, 292.1, 293.6, 298.6, 299.6, 302.6, 305.1, 306.1, 307.1, 309.6])
xisooctane = np.array([0.0014, 0.0015, 0.0017, 0.0028, 0.0034, 0.0047, 0.0057, 0.0076, 0.0090, 0.0151, 0.0190, 0.0262, 0.0406, 0.0508, 0.0637, 0.1247, 0.1411, 0.1775, 0.2614])


xpentanec = []
xhexanec = []
xheptanec = []
xcpentc = []
P = 70000
npontos = 100
vT = np.linspace(250, 312,npontos)

compn = np.identity(ncomp)

xscalc = np.zeros([ncomp,npontos])

for j in range(ncomp-1):
    xscalcji = None
    for i in range(npontos):
        xscalcji = None
        xscalcji =  sol_T(eosmie,vT[i],P,propsol=compn[j+1],xsolguess=xscalcji,nguesses=100)
        xscalc = xscalc.at[j,i].set( xscalcji)
        print(names[j+1],i,xscalc[j,i],vT[i])

#%%
line_styles = ['-',      # Solid line
               '--',     # Dashed line
               '-.',     # Dash-dot line
               ':',      # Dotted line
               (0, (3, 1, 1, 1)),  # Custom dash pattern
               (0, (5, 10)),       # Custom dash pattern
               (0, (3, 5, 1, 5)),  # Custom dash pattern
               (0, (1, 1)),        # Custom dash pattern
               (0, (3, 10, 1, 10)),  # Custom dash pattern
               (0, (3, 1, 1, 1, 1, 1))]  # Custom dash pattern

comp_list = ['pentane','hexane', 'heptane']

fig2=plt.figure(figsize=[4,3])
ax2=fig2.add_subplot(111)

plt.scatter(Tpentane,xpentane  )
plt.scatter(Thexane,xhexane )
plt.scatter(Theptane,xheptane)
# plt.scatter(xisooctane,Tisooctane, label = 'isooctane')


for j in range(ncomp-1):
    plt.plot(vT,xscalc[j,:],color = f'C{j}',linestyle=line_styles[j],label =comp_list[j])

plt.ylabel('Temperature [K]')
plt.xlabel('Solubility')
# plt.plot(-0,260, label=f'SAFT-gamma-mie ',color='k')
# plt.yscale('log')
# plt.xlim(273,313.5)
# plt.xscale('log')

plt.title('Solubility of CBD in organic solvents')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))


plt.savefig(f'CBD_SOL_in_varius_6_2.png',dpi=500,bbox_inches='tight')

xxsss
#%%
from calc_P_sat import calc_P_sat
comps = [CBD]

groups , niki = input_eos(comps)

eosmie = saftymie(groups,niki)
npontos= 100
vT  =  nnp.logspace(np.log10(320), np.log10(430),npontos) [::-1]
vT_exp = np.array([61.1,81.1,101.1,121.1,141.1]) +273.15
vP_exp = [5.18e-03,5.65e-3,8.95e-2,6e-1,2.24]
vP  =vT*np.nan
guessP = 14
for i in range(50,npontos):
    vP[i] = calc_P_sat(vT[i],guessP,eosmie,indxcomp=0)
    print(i, vT[i], vP[i])
    guessP = vP[i]


plt.scatter(vT,np.log(vP),label='SAFT-gamma-mie')
plt.scatter(vT_exp,np.log(np.array(vP_exp)),label='exp.')

plt.xlabel('Temperature [K]')
plt.ylabel('Pressure np.log([Pa])')
# plt.yscale('log')

plt.title('Psat of CBD')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))



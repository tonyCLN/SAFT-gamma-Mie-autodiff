# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 11:41:37 2023

@author: tcava
"""
  # numero de avogadro mol^-1
from jax import numpy as np
import numpy as nnp

def BubblePy(T,x,guessP,guessy,eos,Comp):
    C1, C2 = Comp
    #estimativa inicial de Pbolha
    P = guessP*1.
    y = guessy*1
    j=0
    res_loopP=1
    soma_y=np.sum(y)
    y=y/soma_y #guess
    soma_y=1.
    tol_loopP=1e-10
    jmax=400
    while (res_loopP>tol_loopP and j<jmax):
        densL1 = eos.dens(T,P,x,phase='liq')
        # densL1 = eos.dens_NR(x,T,P,phase='liq')
        # rhoL1 = densL1*Navo
        phi_L1=eos.phi(densL1,x,T)

        # print(densL1,phi_L1)
        f_L=phi_L1*x*P #vetorial
        res_loopY=1 #qq coisa maior q 1e-6
        tol_loopy=1e-10
        i=0
        imax=400
        while (res_loopY>tol_loopy and i<imax):
            #fugacidade do vapor
            densV = eos.dens(T,P,y,phase='vap')
            # densV = eos.dens_NR(y,T,P,phase='vap')
            # rhoV = densV*Navo
            phi_V=eos.phi(densV,y,T)
            # print(densV)
            # print(phi_V)
            f_V=phi_V*y*P #vetorial
            #calculando yi'
            y_novo = nnp.zeros(len(x))
            y_novo[C1]=y[C1]*f_L[C1]/f_V[C1]
            y_novo[C2]=y[C2]*f_L[C2]/f_V[C2]
            res=y_novo-y #vetorial
            res_loopY=np.linalg.norm(res)
            soma_y=np.sum(y_novo) #normalizacao dos y p/ entrar na eq de estado como uma fracao
            y=y_novo*1 #cópia do array essencial para o método numérico #salvando o novo como o 'y velho' da iteracao anterior
            i=i+1
            # if (P == np.nan):
            #     return P,y,i,j
        res_loopP=abs(soma_y-1)
        P=P*soma_y #atualiza P
        j=j+1
    if np.linalg.norm(y/soma_y - x/np.sum(x) ) < 1e-4:
        print ('solução trivial')
        P=np.nan
        # y= y.at[:].set(np.nan)
        y = np.array([np.nan, np.nan])

    return  P, y, i, j

def curva_BubblePy(T,vx,guessP0,guessy0,eos,Comp=np.array([0, 1])):
    nx=len(vx)
    vP=np.zeros(nx)+np.nan
    vy=np.zeros(nx)+np.nan

    yguess=np.array([guessy0,1-guessy0])
    Pguess=guessP0

    for i in range(nx):
        x=np.array([vx[i],1-vx[i]])
        ans= BubblePy(T,x,Pguess,yguess,eos,Comp)

        if ( (~np.isnan(ans[0])) & (ans[2]<200) & (ans[3]<200) ):
            vP= vP.at[i].set(ans[0])
            vy= vy.at[i].set(ans[1][0])
            yguess=np.array([vy[i],1-vy[i]])
            Pguess=vP[i]
                        
            print(i,"...",ans)            
        else:
            vP= vP.at[i].set(np.nan)#discard guess
            vy= vy.at[i].set(np.nan)#discard guess
            print(i,"!!!",ans)
            #stop
        
    return vP,vy


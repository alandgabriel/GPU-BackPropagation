#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 21:10:54 2020

@author: alan
"""
import numpy as np
import matplotlib.pyplot as plt
from Aleatorio import Aleatorio


class ActivFunc:
    
    def __init__(self, xin):
        self.xin = xin
        
        
    def HL (self):
        xin = self.xin.copy()
        ix = [i for i,val in enumerate(xin) if val < 0]
        xout = np.ones(len(xin))
        xout[ix] = 0
        return xout

    def SHL (self):
        xin = self.xin.copy()
        ix = [i for i,val in enumerate(xin) if val < 0]
        xout = np.ones(len(xin))
        xout[ix] = -1
        return xout
    
    def Linear (self):
        xout = self.xin.copy() 
        return xout
    
    def sLinear(self):
        xin = self.xin.copy()
        xout = xin
        xout[xin<0] = 0
        xout[xin>1] = 1
        return xout
    
    def ssLinear(self):
        xin = self.xin.copy()
        xout = xin
        xout[xin<-1] = -1
        xout[xin>1] = 1
        return xout
    
    def logsig (self):
        xin = np.float128(self.xin.copy())
        xout = 1/(1+np.exp(-xin))
        return xout
    
    def tansig (self):
        xin = self.xin.copy()
        xout = (np.exp(xin)-np.exp(-xin))/(np.exp(xin)+np.exp(-xin))
        return xout

    def posLinear(self):
        xin = self.xin.copy()
        xout = xin
        xout[xin<0] = 0
        xout[xin>=0] = xin[xin>=0]
        return xout
    
    def compet(self, xin):
        xout = np.zeros(len(xin))
        xout[xin == np.max(xin)] = 1
        return xout 


"""
N = 20
rnd = Aleatorio()
xr = rnd.gen(N,-4, 5)
x = np.arange(-5,5,.1)
activate = ActivFunc(x)
masc_method = [activate.HL(), activate.SHL(), activate.Linear(), activate.sLinear(), activate.ssLinear(), activate.logsig(), activate.tansig(), activate.posLinear()]
masc_etiq = ["HL", "SHL","Linear", "sLinear", "ssLinear", "logsig","tansig","posLinear"]
y_compet = activate.compet(xr)
plt.subplot(2,1,1)
plt.title("input and output, competitive TF")
plt.stem(np.arange(N),xr)
plt.subplot(2,1,2)
plt.stem(np.arange(N),y_compet)
plt.figure()
for i in range(len(masc_method)):
    plt.plot(x,masc_method[i],label=masc_etiq[i])
    legend = plt.legend(loc='lower right', shadow=True, fontsize='small')
    legend.get_frame().set_facecolor('C0') 
    plt.title('Funciones de transferencia de activación')
    plt.grid()
    
"""

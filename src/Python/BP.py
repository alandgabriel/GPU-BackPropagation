#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 09:57:12 2020

@author: alan
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from ActivFunc import ActivFunc

class BP:
    def __init__(self, iN, hN, oN, lR):
        self.n_inputs      = iN
        self.n_hidden      = hN
        self.n_outputs     = oN
        self.learning_rate = lR
    
        
    # Initialize a network
    def initializeNetwork(self):
    	self.network = {'w1':np.array([[np.random.random_sample() for i in range(self.n_inputs + 1)] for i in range(self.n_hidden)]),'w2':np.array([[np.random.random_sample() for i in range(self.n_hidden + 1)] for i in range(self.n_outputs)])}
    
    #training network
    def trainNetwork(self, x, y):    
        MSE = []
        train = np.random.choice(len(x),130, replace=False)
        a0 = x[train]
        t = y[train]
        j = 0
        mseix = 1
        while j < 10000 and mseix > .0008:
            self.mse = []
            for i in range (len(train)):     
                z = np.array([a0[i]] + [1])
                self.network['a1'] = ActivFunc(np.dot(self.network['w1'],z)).logsig()
                self.network['a2'] = ActivFunc(np.dot(self.network['w2'], np.append(self.network['a1'],1))).Linear()
                e = t[i] - self.network['a2']
                self.mse.append(e[0]    **2)
                self.network['s2'] = -2 * 1 * e
                Fn1 = np.diag([a * (1-a) for a in self.network['a1']])
                self.network['s1'] = np.dot(np.dot(Fn1, self.network['w2'][:,:-1].T), self.network['s2'])
                self.network['w2'] = self.network['w2'] - (self.learning_rate * self.network['s2'] * np.append(self.network['a1'],1).T)
                self.network['w1'] = self.network['w1'] - (self.learning_rate * np.dot(self.network['s1'].reshape((self.n_hidden,1)), np.array([a0[i], 1]).reshape((1,2))))
            mseix = np.mean(self.mse)
            MSE.append(mseix)
            j+=1
            
        plt.xlabel('Épocas')
        plt.ylabel('MSE')
        plt.grid()
        plt.plot(MSE)
        
    #testing network
    def testNetwork(self,x,y):
        fit = []
        for i in range (len(x)):
            z = np.array([x[i]] + [1])
            a1 = ActivFunc(np.dot(self.network['w1'],z)).logsig()
            fit.append( ActivFunc(np.dot(self.network['w2'], np.append(a1,1))).Linear())
        
        plt.figure()
        plt.plot(x, y, label= 'sinc(\u03A0x)')
        plt.plot(x, fit, label= 'BP fit')
        legend = plt.legend(loc='upper right', shadow=True, fontsize='medium')
        legend.get_frame().set_facecolor('C0') 
        plt.xlabel('x')
        plt.grid()
        

#%% MAIN()


x = np.linspace(-15, 15, 150)
y = np.sin(np.pi * x) / (np.pi * x)
bp = BP(1,10,1,.02)	
bp.initializeNetwork()
bp.trainNetwork(x,y)
bp.testNetwork (x,y)



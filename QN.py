# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:00:00 2017

@author: pyh
"""

class fit_QN(object):
    import numpy as np
    def __init__(self,a_init,X,Y,lr,func,dfunc_a,limit=1e-8):
        self.a=np.array(a_init)
        self.input=np.array(X)
        self.Y=np.array(Y)
        self.lr=np.array(lr)
        self.func=func
        self.dfunc=dfunc_a
        self.D=np.mat(np.eye(len(self.a)))
        self.B=np.mat(np.eye(len(self.a)))
        self.limit=limit
        
    def cost(self):
        return np.mean(np.square(np.array(self.func(self.a,self.input))-self.Y))
    
    def dcost(self,da):
        dc=[]
        df=np.array(self.func(da,self.input))-self.Y
        for i in range(len(da)):
            dc.append(2*np.mean(df*
                                np.array(self.dfunc(da,self.input)[i])))

        return np.array(dc)
    
    
    def DFP(self):
        Gra_L2=np.sum(np.square(self.dcost(self.a)))
        while Gra_L2 > self.limit:
            a_old=self.a.copy()
            ds=-self.lr*np.array(np.matmul(self.D,self.dcost(self.a).T))[0]
            self.a=self.a+ds
            dy=self.dcost(self.a)-self.dcost(a_old)
            self.D=self.D+\
            np.matmul(np.mat(ds).T,np.mat(ds))/np.matmul(np.mat(ds),np.mat(dy).T)-\
            np.matmul(np.matmul(np.matmul(self.D,np.mat(dy).T),np.mat(dy)),self.D)/np.matmul(np.matmul(np.mat(dy),self.D),np.mat(dy).T)
            Gra_L2=np.sum(np.square(self.dcost(self.a)))


if __name__=='__main__':
    import numpy as np

    def target(x):
        return 3*x**2+2*x+1+np.random.uniform()

    def function(a,x):
        return a[0]*x**2+a[1]*x+a[2]

    def dfun_a(a,x):
        return [x**2,x,1]

    x=np.array([np.random.uniform()*10 for i in range(10000)])
    y=target(x)

    fit=fit_QN([1,1,1],x,y,[1e-1,1e-1,1e-1],function,dfun_a,limit=1e-16)

    fit.DFP()
    print(fit.a)
    
    
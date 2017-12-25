# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 15:00:00 2017

@author: pyh
"""

class fit_QN(object):
    import numpy as np
    def __init__(self,a_init,X,Y,lr,func,dfunc_a,limit=1e-8,step=1e3):
        self.a=np.array(a_init)
        self.input=np.array(X)
        self.Y=np.array(Y)
        self.lr=np.array(lr)
        self.func=func
        self.dfunc=dfunc_a
        self.D=np.mat(np.eye(len(self.a)))
        self.B=np.mat(np.eye(len(self.a)))
        self.limit=limit
        self.show=step
        
    def cost(self):
        return np.mean(np.square(np.array(self.func(self.a,self.input))-self.Y))
    
    def __dcost(self,da):
        dc=[]
        df=np.array(self.func(da,self.input))-self.Y
        for i in range(len(da)):
            dc.append(2*np.mean(df*
                                np.array(self.dfunc(da,self.input)[i])))

        return np.array(dc)


    def ddcost(self,da):
        dc=[]
        df=np.array(self.func(da,self.input))-self.Y
        for i in range(len(da)):
            dc.append(2*np.mean(df*
                                np.array(self.dfunc(da,self.input)[i])))

        return np.array(dc)    
    
    def DFP(self):
        Gra_L2=np.sum(np.square(self.__dcost(self.a)))
        k=0
        while Gra_L2 > self.limit:
            if not k%self.show:
                print('Step:%d'%k)
                print(self.a)
            a_old=self.a.copy()
            ds=-self.lr*np.array(np.matmul(self.D,self.__dcost(self.a).T))[0]
            self.a=self.a+ds
            dy=self.__dcost(self.a)-self.__dcost(a_old)
            self.D=self.D+\
            np.matmul(np.mat(ds).T,np.mat(ds))/np.matmul(np.mat(ds),np.mat(dy).T)-\
            np.matmul(np.matmul(np.matmul(self.D,np.mat(dy).T),np.mat(dy)),self.D)/np.matmul(np.matmul(np.mat(dy),self.D),np.mat(dy).T)
            Gra_L2=np.sum(np.square(self.__dcost(self.a)))
            k+=1
            
    def BFGS(self):
        Gra_L2=np.sum(np.square(self.__dcost(self.a)))
        k=0
        while Gra_L2>self.limit:
            if not k%self.show:
                print('Step:%d'%k)
                print(self.a)
            a_old=self.a.copy()
            ds=-self.lr*np.array(np.matmul(self.B,self.__dcost(self.a).T))[0]
            self.a=self.a+ds
            dy=self.__dcost(self.a)-self.__dcost(a_old)
            II=np.mat(np.eye(len(self.a)))
            sy_mat=np.matmul(np.mat(ds).T,np.mat(dy))
            ys=np.matmul(np.mat(dy),np.mat(ds).T)
            ys_mat=np.matmul(np.mat(dy).T,np.mat(ds))
            ss_mat=np.matmul(np.mat(ds).T,np.mat(ds))
            self.B=np.matmul(np.matmul((II-sy_mat/ys),self.B),(II-ys_mat/ys))+ss_mat/ys
            Gra_L2=np.sum(np.square(self.__dcost(self.a)))
            k+=1
            


if __name__=='__main__':
    import numpy as np

    def target(x):
        return 3*x[:,0]*np.exp(2*x[:,1])

    def function(a,x):
        return a[0]*x[:,0]*np.exp(a[1]*x[:,1])

    def dfun_a(a,x):
        return [x[:,0]*np.exp(a[1]*x[:,1]),a[0]*x[:,0]*x[:,1]*np.exp(a[1]*x[:,1])]

    x=np.array([[np.random.uniform()*10,np.random.uniform()] for i in range(100)])
    y=target(x)

    fit=fit_QN([2,2],x,y,[1e-3,1e-3],function,dfun_a,limit=1e-8)


#    fit.BFGS()
#    print(fit.a)
    fit.DFP()
    print(fit.a)

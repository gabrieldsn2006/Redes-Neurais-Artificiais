import numpy as np
import matplotlib.pyplot as plt


class MultilayerPerceptronClassifier:
    def __init__(self,X:np.ndarray,D:np.ndarray,learning_rate:float,precision:float, max_epochs:int, topology:list,ax):
        #X (p x N)
        
        self.ax = ax
        
        self.X = X
        self.p, self.N = X.shape
        self.D = D
        self.m = D.shape[0]
        
        self.eta = learning_rate
        self.pr = precision
        self.me = max_epochs
        
        self.topology = topology
        self.topology.append(self.m)
        self.W = []
        self.y = [None]*len(self.topology)
        self.u = [None]*len(self.topology)
        self.delta = [None]*len(self.topology)
        for i,q in enumerate(self.topology):
            if i == 0:
                W = np.random.uniform(-.5, .5, (q, self.p+1))
            else:
                W = np.random.uniform(-.5,.5, (q, self.topology[i-1]+1))
            self.W.append(W)
        bp=1
        
        self.X = np.vstack((
            -np.ones((1,self.N)), self.X
        ))
        t = 250
        x1 = np.linspace(-2,2,t)
        self.X1, self.X2 = np.meshgrid(x1,x1)
        self.X_teste = np.vstack((
            -np.ones((1,t*t)),
            np.ravel(self.X1),
            np.ravel(self.X2),
        ))
        self.ax.set_xlim(-1.1,1.1)
        self.ax.set_ylim(-1.1,1.1)
        self.predict(self.X_teste)
        self.cont = self.ax.contourf(self.X1,self.X2, self.y[-1].reshape(self.X1.shape),alpha=.3)
     
        bp=1
        
    def predict(self,X):
        self.forward(X)
        self.y[-1][self.y[-1]>= 0 ] = 1
        self.y[-1][self.y[-1]< 0 ] = -1
    
    def forward(self, X):
        for i,W in enumerate(self.W):
            if i == 0:
                self.u[i] = W @ X
            else:
                yb = np.vstack((
                    -np.ones((1,self.y[i-1].shape[1])), self.y[i-1]
                ))
                self.u[i] = W @ yb
            self.y[i] = self.g(self.u[i])
            
    def eqm(self):
        self.forward(self.X)
        return np.mean((self.D - self.y[-1])**2)/2
        
    def g(self,u):
        aux = np.exp(-u)
        return (1-aux)/(1+aux)
    
    def g_d(self,y):
        return 0.5*(1-y**2)
    
    def backward(self, e, x):
        for i in range(len(self.W)-1, -1, -1):
            if i == len(self.W)-1:
                self.delta[i] = self.g_d(self.y[i]) * (e)
                yb = np.vstack((
                    -np.ones((1,self.y[i-1].shape[1])), self.y[i-1]
                )) 
                self.W[i] = self.W[i] + self.eta*(self.delta[i] @ yb.T)              
            elif i == 0:
                Wnb = (self.W[i+1][:,1:]).T
                self.delta[i] = self.g_d(self.y[i]) * (Wnb @ self.delta[i+1])                
                self.W[i] = self.W[i] + self.eta*(self.delta[i] @ x.T)             
            else:
                Wnb = (self.W[i+1][:,1:]).T
                self.delta[i] = self.g_d(self.y[i]) * (Wnb @ self.delta[i+1])
                yb = np.vstack((
                    -np.ones((1,self.y[i-1].shape[1])), self.y[i-1]
                )) 
                self.W[i] = self.W[i] + self.eta*(self.delta[i] @ yb.T)
                
        
        
    def fit(self):
        epoch = 0
        EQM = self.eqm()
        
        while epoch < self.me and EQM > self.pr:
            idx = np.random.permutation(self.N)
            self.X = np.copy(self.X)[:,idx]
            self.D = np.copy(self.D)[:,idx]
            for k in range(self.N):
                x_k = self.X[:,k].reshape(self.p+1,1)
                self.forward(x_k)
                d_k = self.D[:,k]
                e_k = d_k - self.y[-1]
                self.backward(e_k, x_k)

            epoch+=1
            self.predict(self.X_teste)
            self.cont.remove()
            self.cont = self.ax.contourf(self.X1,self.X2, self.y[-1].reshape(self.X1.shape),alpha=.3)
            EQM = self.eqm()
            ax.set_title(f"Época: {epoch} EQM: {EQM:.8f}")
            plt.pause(.1)
        plt.show()
data = np.loadtxt("./data/spiral_d.csv",delimiter=',')
classes = np.unique(data[:,-1])

X = data[:,:-1]
# 1400 x 2
min = np.min(X,axis=0)
max = np.max(X,axis=0)
#Min-Max
X = 2*(X - min)/(max-min)-1
d = data[:,-1:]
fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.scatter(X[d[:,0]==1,0], X[d[:,0]==1,1],
           edgecolors='k',c='g')
ax.scatter(X[d[:,0]==-1,0], X[d[:,0]==-1,1],
           edgecolors='k',c='b')
mlp = MultilayerPerceptronClassifier(X.T, d.T, 1e-1,1e-8, 1000, [100],ax)
mlp.fit()
plt.show()

bp = 1
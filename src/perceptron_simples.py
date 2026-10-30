import numpy as np
import matplotlib.pyplot as plt


def degrau_bipolar(u_k):
    return 1 if u_k>=0 else -1


bp = 1

X = np.array([
[1,1],
[0,1],
[0,2],
[1,0],
[2,2],
[4,1.5],
[1.5,6],
[3,5],
[3,3],
[6,4    ],
])


d = np.array([
[1],
[-1],
[1],
[1],
[1],
[-1],
[-1],
[-1],
[-1],
[-1   ],
])


fig = plt.figure(1)
ax = fig.add_subplot(1,1,1)
ax.scatter(X[:5,0],X[:5,1],)
ax.scatter(X[5:,0],X[5:,1],)
ax.set_xlim(-1,7)
ax.set_ylim(-1,7)

N,p = X.shape

X = X.T
X = np.vstack((
    -np.ones((1,N)), X
))

eta = 0.0001 #Learning Rate
w = np.zeros((p+1,1))
w = np.random.random_sample((p+1,1))-.5
x1 = np.linspace(-4,10)
x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
plt.plot(x1,x2, c = 'k')

linha = None
erro = True
epoca = 0
while erro:
    erro = False
    
    for k in range(N):
        x_k = X[:,k].reshape(p+1,1)
        d_k = d[k,0]
        u_k = (w.T @ x_k)[0,0]
        y_k = degrau_bipolar(u_k)
        e_k = d_k - y_k
        w = w + eta*e_k*x_k
        if e_k != 0: 
            erro = True
    epoca+=1
    x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    if linha is not None:
        linha[0].remove()
    linha = ax.plot(x1,x2, c = 'k')
    plt.pause(.1)
    
    
x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
linha = ax.plot(x1,x2, c = 'b', lw=3)


bp=1



plt.title(f"Épocas:{epoca}")

plt.show()
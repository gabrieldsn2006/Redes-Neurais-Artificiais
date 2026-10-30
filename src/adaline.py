import numpy as np
import matplotlib.pyplot as plt

def erro_quadratico_medio(X):
    s = 0
    for k in range(N):
        x_k = X[:,k].reshape(p+1,1)
        d_k = d[k,0]
        u_k = (w.T @ x_k)[0,0]        
        s += (d_k - u_k)**2
    return s/(2*N)

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
[1],
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

eta = 0.01 #Learning Rate
max_epocas = 200
precisao = 1e-1
w = np.zeros((p+1,1))
w = np.random.random_sample((p+1,1))-.5
x1 = np.linspace(-4,10)
x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
plt.plot(x1,x2, c = 'k')

linha = None

epoca = 0
EQM = erro_quadratico_medio(X)
historico_EQM = [EQM]

while epoca < max_epocas and EQM > precisao:
    
    
    for k in range(N):
        x_k = X[:,k].reshape(p+1,1)
        d_k = d[k,0]
        u_k = (w.T @ x_k)[0,0]        
        e_k = d_k - u_k
        w = w + eta*e_k*x_k
    EQM = erro_quadratico_medio(X)
    historico_EQM.append(EQM)
    epoca+=1
    x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
    x2 = np.nan_to_num(x2)
    if linha is not None:
        linha[0].remove()
    linha = ax.plot(x1,x2, c = 'k')
    plt.title(f"Época {epoca}, EQM: {EQM:.12f}")
    # plt.pause(.1)
    
    
x2 = -w[1,0]/w[2,0]*x1 + w[0,0]/w[2,0]
x2 = np.nan_to_num(x2)
linha = ax.plot(x1,x2, c = 'b', lw=3)


bp=1



plt.title(f"Épocas:{epoca}")
plt.figure(2)
plt.plot(historico_EQM)
plt.xlabel("Épocas")
plt.ylabel("EQM")
plt.title("Curva do Aprendizado")

plt.show()
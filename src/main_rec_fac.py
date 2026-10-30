import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


CAMINHO = "./data/RecFac"
DIMENSAO = 30   # imagens redimensionadas para 30x30 (opções: 30, 40, 50 ou 80)
R = 10          # rodadas de Monte Carlo
np.random.seed(42)


def degrau_bipolar(u):
    return np.where(u >= 0, 1, -1)


def sigmoide_bipolar(u):
    aux = np.exp(-u)
    return (1-aux)/(1+aux)


def add_bias(X):
    return np.vstack((-np.ones((1, X.shape[1])), X))


def matriz_confusao(y_real, y_pred, C):
    mc = np.zeros((C, C), dtype=int)
    for i, j in zip(y_real, y_pred):
        mc[i, j] += 1
    return mc


# Em todos os modelos X já vem com o bias, (p+1) x N, e D é C x N.
# A classe predita é o neurônio de saída com maior valor.

class PerceptronSimples:
    def __init__(self, eta, max_epocas):
        self.eta = eta
        self.me = max_epocas

    def eqm(self, X, D):
        return np.mean((D - degrau_bipolar(self.W @ X))**2)/2

    def fit(self, X, D):
        N = X.shape[1]
        self.W = np.random.uniform(-.5, .5, (D.shape[0], X.shape[0]))
        self.historico_EQM = [self.eqm(X, D)]
        erro = True
        epoca = 0
        while erro and epoca < self.me:
            erro = False
            for k in np.random.permutation(N):
                x_k = X[:, k:k+1]
                e_k = D[:, k:k+1] - degrau_bipolar(self.W @ x_k)
                self.W = self.W + self.eta*(e_k @ x_k.T)
                if np.any(e_k != 0):
                    erro = True
            epoca += 1
            self.historico_EQM.append(self.eqm(X, D))

    def predict(self, X):
        return np.argmax(self.W @ X, axis=0)


class Adaline:
    def __init__(self, eta, precisao, max_epocas):
        self.eta = eta
        self.pr = precisao
        self.me = max_epocas

    def eqm(self, X, D):
        return np.mean((D - self.W @ X)**2)/2

    def fit(self, X, D):
        N = X.shape[1]
        self.W = np.random.uniform(-.5, .5, (D.shape[0], X.shape[0]))
        EQM = self.eqm(X, D)
        self.historico_EQM = [EQM]
        epoca = 0
        while epoca < self.me and EQM > self.pr:
            for k in np.random.permutation(N):
                x_k = X[:, k:k+1]
                e_k = D[:, k:k+1] - self.W @ x_k
                self.W = self.W + self.eta*(e_k @ x_k.T)
            EQM = self.eqm(X, D)
            self.historico_EQM.append(EQM)
            epoca += 1

    def predict(self, X):
        return np.argmax(self.W @ X, axis=0)


class MLP:
    def __init__(self, topologia, eta, precisao, max_epocas):
        self.topologia = topologia
        self.eta = eta
        self.pr = precisao
        self.me = max_epocas

    def forward(self, X):
        self.y = []
        entrada = X
        for W in self.W:
            self.y.append(sigmoide_bipolar(W @ entrada))
            entrada = add_bias(self.y[-1])
        return self.y[-1]

    def backward(self, x, e):
        for i in range(len(self.W)-1, -1, -1):
            g_d = 0.5*(1 - self.y[i]**2)
            if i == len(self.W)-1:
                delta = g_d*e
            else:
                delta = g_d*(self.W[i+1][:, 1:].T @ delta)
            entrada = x if i == 0 else add_bias(self.y[i-1])
            self.W[i] = self.W[i] + self.eta*(delta @ entrada.T)

    def eqm(self, X, D):
        return np.mean((D - self.forward(X))**2)/2

    def fit(self, X, D):
        N = X.shape[1]
        self.W = []
        n_entradas = X.shape[0]
        for q in self.topologia + [D.shape[0]]:
            self.W.append(np.random.uniform(-.5, .5, (q, n_entradas)))
            n_entradas = q + 1
        EQM = self.eqm(X, D)
        self.historico_EQM = [EQM]
        epoca = 0
        while epoca < self.me and EQM > self.pr:
            for k in np.random.permutation(N):
                x_k = X[:, k:k+1]
                e_k = D[:, k:k+1] - self.forward(x_k)
                self.backward(x_k, e_k)
            EQM = self.eqm(X, D)
            self.historico_EQM.append(EQM)
            epoca += 1

    def predict(self, X):
        return np.argmax(self.forward(X), axis=0)


# Organização dos dados
pessoas = sorted(os.listdir(CAMINHO))
X = []
rotulos = []
for c, pessoa in enumerate(pessoas):
    for arquivo in sorted(os.listdir(os.path.join(CAMINHO, pessoa))):
        img = cv2.imread(os.path.join(CAMINHO, pessoa, arquivo), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (DIMENSAO, DIMENSAO), interpolation=cv2.INTER_AREA)
        X.append(img.flatten())
        rotulos.append(c)

X = np.array(X, dtype=float).T   # p x N
rotulos = np.array(rotulos)
p, N = X.shape
C = len(pessoas)

# Min-Max para [-1, 1]
X = 2*(X - X.min())/(X.max() - X.min()) - 1
X = add_bias(X)                  # (p+1) x N

# One-hot encoding com +1/-1
Y = -np.ones((C, N))
Y[rotulos, np.arange(N)] = 1

print(f"X: {X.shape}  Y: {Y.shape}")

modelos = {
    "Perceptron Simples": PerceptronSimples(eta=1e-2, max_epocas=100),
    "ADALINE": Adaline(eta=1e-3, precisao=1e-3, max_epocas=100),
    "MLP": MLP(topologia=[50], eta=1e-2, precisao=1e-3, max_epocas=200),
}

# Validação por Monte Carlo (80% treino / 20% teste)
acuracias = {nome: [] for nome in modelos}
matrizes = {nome: [] for nome in modelos}
curvas = {nome: [] for nome in modelos}

for r in range(R):
    idx = np.random.permutation(N)
    n_treino = int(.8*N)
    treino, teste = idx[:n_treino], idx[n_treino:]
    for nome, modelo in modelos.items():
        modelo.fit(X[:, treino], Y[:, treino])
        mc = matriz_confusao(rotulos[teste], modelo.predict(X[:, teste]), C)
        acc = np.trace(mc)/np.sum(mc)
        acuracias[nome].append(acc)
        matrizes[nome].append(mc)
        curvas[nome].append(modelo.historico_EQM)
        print(f"Rodada {r+1}/{R} - {nome}: acurácia = {acc:.4f} ({len(modelo.historico_EQM)-1} épocas)")

# Tabela da acurácia
print(f"\n{'Modelos':<20}{'Média':>10}{'Desvio-Padrão':>15}{'Maior Valor':>13}{'Menor Valor':>13}")
for nome, acc in acuracias.items():
    print(f"{nome:<20}{np.mean(acc):>10.4f}{np.std(acc):>15.4f}{np.max(acc):>13.4f}{np.min(acc):>13.4f}")

# Boxplot e violin plot das acurácias
fig, axs = plt.subplots(1, 2, figsize=(14, 5))
sns.boxplot(data=list(acuracias.values()), ax=axs[0])
sns.violinplot(data=list(acuracias.values()), ax=axs[1])
for ax in axs:
    ax.set_xticks(range(len(modelos)), modelos.keys())
    ax.set_ylabel("Acurácia")
fig.suptitle(f"Acurácia em {R} rodadas")

# Matrizes de confusão e curvas de aprendizado da maior e da menor acurácia
fig_curvas, axs_curvas = plt.subplots(1, len(modelos), figsize=(16, 5))
for ax_curva, nome in zip(axs_curvas, modelos):
    casos = [("Maior", np.argmax(acuracias[nome])), ("Menor", np.argmin(acuracias[nome]))]

    fig, axs = plt.subplots(1, 2, figsize=(18, 8))
    for ax, (caso, i) in zip(axs, casos):
        sns.heatmap(matrizes[nome][i], annot=True, cbar=False, cmap="Blues",
                    xticklabels=pessoas, yticklabels=pessoas, ax=ax)
        ax.set_title(f"{caso} acurácia: {acuracias[nome][i]:.4f} (rodada {i+1})")
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
    fig.suptitle(nome)

    for caso, i in casos:
        ax_curva.plot(curvas[nome][i], label=f"{caso} acurácia (rodada {i+1})")
    ax_curva.set_title(nome)
    ax_curva.set_xlabel("Épocas")
    ax_curva.set_ylabel("EQM")
    ax_curva.legend()
fig_curvas.suptitle("Curvas de aprendizado")
fig_curvas.tight_layout()

plt.show()

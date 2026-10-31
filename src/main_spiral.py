import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


R = 500         # rodadas de Monte Carlo
np.random.seed(42)
METRICAS = ["Acurácia", "Sensibilidade", "Especificidade", "Precisão", "F1-score"]


def degrau_bipolar(u):
    return np.where(u >= 0, 1, -1)


def sigmoide_bipolar(u):
    aux = np.exp(-u)
    return (1-aux)/(1+aux)


def add_bias(X):
    return np.vstack((-np.ones((1, X.shape[1])), X))


def matriz_confusao(d, y):
    # linhas: classe real, colunas: classe predita, na ordem [+1, -1]
    mc = np.zeros((2, 2), dtype=int)
    for i, j in zip(d, y):
        mc[int(i == -1), int(j == -1)] += 1
    return mc


def metricas(mc):
    # a classe +1 é a positiva
    VP, FN, FP, VN = mc[0, 0], mc[0, 1], mc[1, 0], mc[1, 1]
    acc = (VP + VN)/np.sum(mc)
    sens = VP/(VP + FN) if VP + FN > 0 else 0
    esp = VN/(VN + FP) if VN + FP > 0 else 0
    prec = VP/(VP + FP) if VP + FP > 0 else 0
    f1 = 2*prec*sens/(prec + sens) if prec + sens > 0 else 0
    return [acc, sens, esp, prec, f1]


# Em todos os modelos X já vem com o bias, (p+1) x N, e D é 1 x N.
# Com um único neurônio de saída, a classe predita é o sinal da saída.

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
        return degrau_bipolar(self.W @ X)[0]


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
        return degrau_bipolar(self.W @ X)[0]


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
        return degrau_bipolar(self.forward(X))[0]


# Organização dos dados
dados = np.loadtxt("./data/spiral_d.csv", delimiter=",")
X = dados[:, :2]                 # N x p
d = dados[:, 2]
N = X.shape[0]

# Visualização inicial
plt.figure(figsize=(6, 6))
plt.scatter(X[d == 1, 0], X[d == 1, 1], s=12, edgecolors="k", linewidths=.3, label="Classe +1")
plt.scatter(X[d == -1, 0], X[d == -1, 1], s=12, edgecolors="k", linewidths=.3, label="Classe -1")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.title("Conjunto spiral_d")

# Min-Max para [-1, 1] em cada atributo
X = 2*(X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0)) - 1
X = add_bias(X.T)                # (p+1) x N
D = d.reshape(1, N)              # 1 x N
n_treino = int(.8*N)

print(f"X: {X.shape}  D: {D.shape}")

# Underfitting e overfitting da MLP em uma partição 80/20
topologias = {
    "Subdimensionada": [2],
    "Adequada": [10],
    "Superdimensionada": [100, 100],
}
idx = np.random.permutation(N)
treino, teste = idx[:n_treino], idx[n_treino:]
x1 = np.linspace(-1.1, 1.1, 200)
X1, X2 = np.meshgrid(x1, x1)
X_grade = add_bias(np.vstack((X1.ravel(), X2.ravel())))

fig, axs = plt.subplots(3, len(topologias), figsize=(16, 15))
for j, (nome, topologia) in enumerate(topologias.items()):
    mlp = MLP(topologia, eta=1e-1, precisao=1e-3, max_epocas=1000)
    mlp.fit(X[:, treino], D[:, treino])
    acc_treino = metricas(matriz_confusao(d[treino], mlp.predict(X[:, treino])))[0]
    mc = matriz_confusao(d[teste], mlp.predict(X[:, teste]))
    m = metricas(mc)
    print(f"{nome} {topologia}: acurácia treino = {acc_treino:.4f}, teste = {m[0]:.4f}, "
          f"sensibilidade = {m[1]:.4f}, especificidade = {m[2]:.4f} ({len(mlp.historico_EQM)-1} épocas)")

    axs[0, j].contourf(X1, X2, mlp.predict(X_grade).reshape(X1.shape), levels=[-2, 0, 2],
                       colors=["tab:orange", "tab:blue"], alpha=.3)
    axs[0, j].scatter(X[1, teste][d[teste] == 1], X[2, teste][d[teste] == 1], s=12, c="tab:blue", edgecolors="k", linewidths=.3)
    axs[0, j].scatter(X[1, teste][d[teste] == -1], X[2, teste][d[teste] == -1], s=12, c="tab:orange", edgecolors="k", linewidths=.3)
    axs[0, j].set_title(f"{nome} {topologia}\ntreino: {acc_treino:.4f}  teste: {m[0]:.4f}")

    axs[1, j].plot(mlp.historico_EQM)
    axs[1, j].set_xlabel("Épocas")
    axs[1, j].set_ylabel("EQM")

    sns.heatmap(mc, annot=True, fmt="d", cbar=False, cmap="Blues",
                xticklabels=["+1", "-1"], yticklabels=["+1", "-1"], ax=axs[2, j])
    axs[2, j].set_xlabel("Predito")
    axs[2, j].set_ylabel("Real")
fig.suptitle("MLP: underfitting e overfitting")
fig.tight_layout()

modelos = {
    "Perceptron Simples": PerceptronSimples(eta=1e-2, max_epocas=50),
    "ADALINE": Adaline(eta=1e-3, precisao=1e-3, max_epocas=50),
    "MLP": MLP(topologia=[10], eta=1e-1, precisao=1e-3, max_epocas=100),
}

# Validação por amostragem aleatória (80% treino / 20% teste)
resultados = {nome: np.zeros((R, len(METRICAS))) for nome in modelos}
matrizes = {nome: [] for nome in modelos}
curvas = {nome: [] for nome in modelos}

for r in range(R):
    idx = np.random.permutation(N)
    treino, teste = idx[:n_treino], idx[n_treino:]
    for nome, modelo in modelos.items():
        modelo.fit(X[:, treino], D[:, treino])
        mc = matriz_confusao(d[teste], modelo.predict(X[:, teste]))
        resultados[nome][r] = metricas(mc)
        matrizes[nome].append(mc)
        curvas[nome].append(modelo.historico_EQM)
    print(f"Rodada {r+1}/{R} - " + ", ".join(f"{nome}: {resultados[nome][r, 0]:.4f}" for nome in modelos))

# Uma tabela por métrica
for k, metrica in enumerate(METRICAS):
    print(f"\n{metrica}")
    print(f"{'Modelos':<20}{'Média':>10}{'Desvio-Padrão':>15}{'Maior Valor':>13}{'Menor Valor':>13}")
    for nome in modelos:
        v = resultados[nome][:, k]
        print(f"{nome:<20}{np.mean(v):>10.4f}{np.std(v):>15.4f}{np.max(v):>13.4f}{np.min(v):>13.4f}")

# Boxplot de cada métrica
fig, axs = plt.subplots(1, len(METRICAS), figsize=(22, 5))
for k, metrica in enumerate(METRICAS):
    sns.boxplot(data=[resultados[nome][:, k] for nome in modelos], ax=axs[k])
    axs[k].set_xticks(range(len(modelos)), ["Perceptron", "ADALINE", "MLP"])
    axs[k].set_title(metrica)
fig.suptitle(f"Métricas em {R} rodadas")
fig.tight_layout()

# Matrizes de confusão e curvas de aprendizado das rodadas de maior e menor valor de cada métrica
for nome in modelos:
    fig, axs = plt.subplots(3, len(METRICAS), figsize=(24, 13))
    for k, metrica in enumerate(METRICAS):
        casos = [("Maior", np.argmax(resultados[nome][:, k])), ("Menor", np.argmin(resultados[nome][:, k]))]
        for linha, (caso, i) in enumerate(casos):
            sns.heatmap(matrizes[nome][i], annot=True, fmt="d", cbar=False, cmap="Blues",
                        xticklabels=["+1", "-1"], yticklabels=["+1", "-1"], ax=axs[linha, k])
            axs[linha, k].set_title(f"{caso} {metrica.lower()}: {resultados[nome][i, k]:.4f}\n(rodada {i+1})")
            axs[linha, k].set_xlabel("Predito")
            axs[linha, k].set_ylabel("Real")
            axs[2, k].plot(curvas[nome][i], label=f"{caso} (rodada {i+1})")
        axs[2, k].set_xlabel("Épocas")
        axs[2, k].set_ylabel("EQM")
        axs[2, k].legend()
    fig.suptitle(nome)
    fig.tight_layout()

plt.show()

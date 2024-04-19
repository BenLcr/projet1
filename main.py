#%%
import numpy as np
import scipy.io 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#%%
data = scipy.io.loadmat('C:/Users/lacou/Desktop/traitement donnée/projet1/data.mat') 

X1 = pd.DataFrame(data["data1"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X2 = pd.DataFrame(data["data2"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X3 = pd.DataFrame(data["data3"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])

# Convertir les données en excel
# merged_df = pd.concat([X1, X2, X3], ignore_index=True)
# merged_df.to_excel('C:/Users/lacou/Desktop/traitement donnée/projet1/data.xlsx', index=False)

#%% Description des données
df = [X1, X2, X3]
i = 1

for elt in df:
    print(f"Description data{i}")
    print(f"Nombre de pulses : {len(elt)}")
    print(f"Nombre d'artéfacts: {len(elt[elt['Class'] == 1])}")
    print(f"Nombre de forme correcte : {len(elt[elt['Class'] == 2])}")
    # print(elt.describe())
    i += 1

#%% ACP
X1_clean = X1.iloc[:,2:]
y = X1['Class']

acp = PCA(n_components=2)
X1_acp = acp.fit_transform(X1_clean)
acp_df = pd.DataFrame(data=X1_acp, columns=['PC1', 'PC2'])
acp_df['Class'] = y.values

#%% Affichage des données selon les différentes composantes
plt.figure(figsize=(8, 6))
colors = {1: 'red', 2: 'blue'}
for cls, color in colors.items():
    plt.scatter(acp_df.loc[acp_df['Class'] == cls, 'PC1'], acp_df.loc[acp_df['Class'] == cls, 'PC2'], c=color, label=f'Classe {cls}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('ACP - Visualisation des données')
plt.legend()

#%% ADL
adl = LinearDiscriminantAnalysis()
X1_adl = adl.fit_transform(X1_clean, y)
rand = np.random.randn(int(np.shape(X1_adl)[0]))

plt.figure()
scatter = plt.scatter(X1_adl[:, 0], rand, c=y)
plt.xlabel('Composante discriminante 1')
plt.ylabel('Composante discriminante 2')
plt.title('LDA')
plt.legend(*scatter.legend_elements())

corvar = np.zeros((X1_clean.shape[1], 2))

for i in range(X1_clean.shape[1]):
    corvar[i, 0] = np.corrcoef(X1_clean.iloc[:, i], X1_adl[:, 0])[0, 1]

# Affichage des étiquettes (noms des variables)
for j in range(X1_clean.shape[1] - 1):
    plt.annotate(X1_clean.columns[j], (corvar[j, 0], corvar[j, 1]))

corvar = np.zeros((X1_clean.shape[1] - 2, 2))

for i in range(2, X1_clean.shape[1] - 2):
    corvar[i-2, 0] = np.corrcoef(X1_clean.iloc[:, i], X1_adl[:, 0])[0, 1]

fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)

rand2 = np.random.rand(X1_clean.shape[1] - 2)
plt.scatter(corvar[:, 0], rand2)

# Affichage des étiquettes (noms des variables)
for j in range(2, X1_clean.shape[1] - 2):
    plt.annotate(X1_clean.columns[j], (corvar[j, 0], rand2[j-2]))

# Ajout des axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)
plt.plot("Cercle de corrélation")

# Ajout d'un cercle
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axes.add_artist(cercle)

# %%
#Projection et comparasion pour trier les inconnus
#Inconnu = X2_clean
X2_clean = X2.iloc[:,2:]
X2_adl = adl.transform(X2_clean.iloc[:,2:])
plt.figure()
plt.scatter(X2_adl[:, 0], rand, c=data['C'])
plt.scatter(X2_adl, X2_clean['Class'], marker='*')
plt.xlabel('Composante discriminante 1')
plt.ylabel('rand')
plt.show()

Xinc_pred = adl.predict(X2_clean.iloc[:,2:]) #Predict class labels for samples in X
print("Prediction :", Xinc_pred)
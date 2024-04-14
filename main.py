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

corvar = np.zeros((X1_adl.shape[1], 2))

for i in range(X1_adl.shape[1]):
    corvar[i, 0] = np.corrcoef(X1_clean.iloc[:, i], X1_adl[:, 0])[0, 1]

plt.show()

# %%

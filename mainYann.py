# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:57:52 2024

@author: picody
"""

#%%
import numpy as np
import scipy.io 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

#%%
data = scipy.io.loadmat('data.mat') 

X1 = pd.DataFrame(data["data1"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X2 = pd.DataFrame(data["data2"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X3 = pd.DataFrame(data["data3"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X1["Duree"] = X1["Fin"] - X1["Debut"]
X2["Duree"] = X2["Fin"] - X2["Debut"]
X3["Duree"] = X3["Fin"] - X3["Debut"]

#print("Le nombre d'artéfacts dans le jeu de données 1 est : ", Nb_pulse)
print (f"Nombre de pulses dans le jeu de données 1 : {len(X1[X1['Class'] == 1])}")
print (f"Nombre de pulses dans le jeu de données 2 : {len(X2[X2['Class'] == 1])}")
print (f"Nombre de pulses dans le jeu de données 3 : {len(X3[X3['Class'] == 1])}")

X1_clean = X1.iloc[:,4:]
X2_clean = X2.iloc[:,4:]
X3_clean = X3.iloc[:,4:]
y1 = X1['Class']
y2 = X2['Class']
y3 = X3['Class']

# Corrélation
correlation_matrix = X1_clean.corr()

# Heatmap avec seaborn
plt.figure()
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.show()

# instanciation
acp = PCA()
X1acp = acp.fit_transform(X1_clean)

# Affichage des variances expliquées par chaque axe (Pareto)
plt.figure()
plt.bar(np.arange(1, len(acp.explained_variance_ratio_)+1), acp.explained_variance_ratio_)
plt.plot(np.arange(1, len(acp.explained_variance_ratio_)+1), np.cumsum(acp.explained_variance_ratio_), color='red')
plt.ylabel("Variance expliquée en ratio et cumul")
plt.xlabel("Nombre de composantes principales")
plt.title("Pareto des variances")
plt.show()

# Les individus projetés dans l'espace des 2 axes principaux
acp_df = pd.DataFrame(data=X1acp, columns=['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6'])
acp_df['Class'] = y1.values

#%% Affichage des données selon les différentes composantes
colors = {1: 'red', 2: 'blue'}
for cls, color in colors.items():
    plt.scatter(acp_df.loc[acp_df['Class'] == cls, 'PC1'], acp_df.loc[acp_df['Class'] == cls, 'PC2'], c=color, label=f'Classe {cls}')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('ACP - Visualisation des données')
plt.legend()

#%%
corvar = np.zeros((X1_clean.shape[1], 2))

for j in range(X1_clean.shape[1]):
    corvar[j, 0] = np.corrcoef(X1_clean.iloc[:, j], X1acp[:, 0])[0, 1]
    corvar[j, 1] = np.corrcoef(X1_clean.iloc[:, j], X1acp[:, 1])[0, 1]

# Cercle des corrélations
fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)
plt.scatter(corvar[:, 0], corvar[:, 1])

# Affichage des étiquettes (noms des variables)
for j in range(X1_clean.shape[1]):
    plt.annotate(X1_clean.columns[j], (corvar[j, 0], corvar[j, 1]))

# Ajout des axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

# Ajout du cercle
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axes.add_artist(cercle)

plt.title("Cercle des corrélations")
plt.show()

#%% Fusion des enregistrements pour les classifiers
x_frames = [X1_clean, X2_clean, X3_clean]
x = pd.concat(x_frames)

y_frames = [y1, y2, y3]
y = pd.concat(y_frames)

#%% ADL
adl = LinearDiscriminantAnalysis()
x_adl = adl.fit_transform(x, y)
rand = np.random.randn(int(np.shape(x_adl)[0]))

plt.figure()
scatter = plt.scatter(x_adl[:, 0], rand, c=y)
plt.xlabel('Composante discriminante 1')
plt.ylabel('Composante discriminante 2')
plt.title('LDA')
plt.legend(*scatter.legend_elements())


# Utiliser tous les classifiers sur la meme base de test (train_test_split)

# si il y a un gros ecart entre les classes (en nb), effectuer une matrisse de perf normalisée (balenced accuracy)
# il vaut mieux ajouter l'intervalle de confiance (intervalle de confiance à 95%, f-1.96*...)
# ou faire la moyenne et variablilité sur plusieurs split


# pour les classifiers de Bayes, on apprend le centre de gravité (mu) et matrisse de variances covariances
#%% Gaussian Naive Bayes

#fractionner dataset (train-test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)
# corss_val => utilise plusieurs paquets pour faire plusieurs matrices de confusion puis score
#instanciation
model_Gaussian = GaussianNB()
#training
model_Gaussian.fit(x_train, y_train)
#prédiction
prediction = model_Gaussian.predict(x_test)
print(prediction)
#evaluation du modèle
precision = accuracy_score(y_test, prediction)*100
print(precision)

#%% Discriminant linéaire

# X contient les données avec en ligne les individus et en colonne les caractéristiques et y les # classes des individus 
#X_r2 = lda.fit(X, y).transform(X) 
#print(adl.explained_variance_ratio_) 

#%% Discriminant quadratique



# Pour PPV
# en mémoire, on doit stocker la base d'apprentissage -> très lourd pour la machine

# la validation de doit pas venur du testing
# dans validation, tracer accuracy par k. Choisir k entre 1 et 30 (pas plus)
# afficher la l'IC pour l'accuracy en validation
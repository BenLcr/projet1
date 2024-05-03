#%%
import numpy as np
import scipy.io 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay

#%%
data = scipy.io.loadmat('data.mat') 

X1 = pd.DataFrame(data["data1"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X2 = pd.DataFrame(data["data2"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X3 = pd.DataFrame(data["data3"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X1['Duree'] = X1['Fin'] - X1['Debut']
X2['Duree'] = X2['Fin'] - X2['Debut']
X3['Duree'] = X3['Fin'] - X3['Debut']

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


pd.plotting.scatter_matrix(X1.iloc[:, 4:], c=X1['Class'])
plt.title("scatter matrix")

#%% ACP
X1_clean = X1.iloc[:,4:]
y = X1['Class']

acp = PCA()
X1_acp = acp.fit_transform(X1_clean)
acp_df = pd.DataFrame(data=X1_acp[:,:2], columns=['PC1', 'PC2'])
acp_df['Class'] = y.values

plt.figure()
plt.bar(np.arange(1, len(acp.explained_variance_ratio_)+1), acp.explained_variance_ratio_)
plt.plot(np.arange(1, len(acp.explained_variance_ratio_)+1), np.cumsum(acp.explained_variance_ratio_), color='r')
plt.ylabel("Variance expliquée en ratio et cumul")
plt.xlabel("Nombre de facteurs")
plt.show()

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

#%% Cercle de correlation
corvar = np.zeros((X1_clean.shape[1], 2))

for i in range(X1_clean.shape[1]):
    corvar[i, 0] = np.corrcoef(X1_clean.iloc[:, i], X1_acp[:, 0])[0, 1]
    corvar[i, 1] = np.corrcoef(X1_clean.iloc[:, i], X1_acp[:, 1])[0, 1]

# Affichage des étiquettes (noms des variables)
for j in range(X1_clean.shape[1] - 1):
    plt.annotate(X1_clean.columns[j], (corvar[j, 0], corvar[j, 1]))

fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)

#rand2 = np.random.rand(X1_clean.shape[1] - 2)
plt.scatter(corvar[:, 0], corvar[:, 1])

# Ajout des axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

# Ajout d'un cercle
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axes.add_artist(cercle)

# Affichage du cercle de corrélation
plt.title('Cercle de corrélation')
plt.axis('equal')  # Pour que les axes aient la même échelle
plt.figure()

# %%
#Projection et comparasion pour trier les inconnus
#Inconnu = X2_clean

X2_clean = X2.iloc[:,4:]
X2_adl = adl.transform(X2_clean)
rand = np.random.randn(int(np.shape(X2_adl)[0]))

X2_pred = adl.predict(X2_clean) #Predict class labels for samples in X
print(X2_pred)

plt.figure()
plt.scatter(X2_adl[:, 0], X2.iloc[:,1], label='Classe réelle')
plt.scatter(X2_adl[:, 0], X2.iloc[:,1:]['Class'], marker='*', label='Classe prédite')
plt.xlabel('Composante discriminante 1')
plt.ylabel('rand')
plt.legend()
plt.title('Projection des données inconnues')

# Matrice de confusion
cm = confusion_matrix(X2["Class"], X2_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='hot')

#%% Courbes ROC
'''
plt.figure()
for i in range(0, X2_clean.shape[1]-1):
    fpr, tpr, thresholds = roc_curve(X2_clean, X2_clean[:,i], pos_label=1)
    #Redressement de la courbe
    auc = roc_auc_score(X2_clean, X2_clean[:,i])
    if auc < 0.5 :
        plt.plot(fpr, tpr, label=X2_clean.columns[i])
    else :
        fpr, tpr, thresholds = roc_curve(X2_clean, -X2_clean[:,i], pos_label=1)
        plt.plot(fpr, tpr, label=data.columns[i])

# Plot Xadl et ROC
plt.xlabel('Taux de Faux Negatif')
plt.ylabel('Taux de Vrai Positif')
plt.title('Courbe ROC')
plt.plot([0, 1], [0, 1], color='b', lw=2, linestyle='--', label='Pire cas')
plt.legend()

'''
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

X = pd.concat([X1, X2, X3])
x = X.iloc[:, 4:]
y = X['Class']
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

#instanciation et définition du k
knn = KNeighborsClassifier(n_neighbors = 3)
#training
knn.fit(x_train,y_train)
#Precision
precision = knn.score(x_test, y_test)
print("Précision du classificateur kvoisins :", precision)

# Instanciation et définition du classificateur linéaire
# linear_svc = LinearSVC()

# # Entraînement du modèle
# linear_svc.fit(x_train, y_train)

# # Précision du modèle sur l'ensemble de test
# precision = linear_svc.score(x_test, y_test)
# print("Précision du classificateur linéaire :", precision)

# Instanciation et définition du classificateur quadratique
quadratic_classifier = QuadraticDiscriminantAnalysis()

#training
quadratic_classifier.fit(x_train, y_train)

#Precision
precision_quadratic = quadratic_classifier.score(x_test, y_test)
print("Précision du classificateur quadratique :", precision_quadratic)

#Instanciation et définition du classificateur bayésien naïf
naive_bayes_classifier = GaussianNB()

#training
naive_bayes_classifier.fit(x_train, y_train)

#Precision
precision_naive_bayes = naive_bayes_classifier.score(x_test, y_test)
print("Précision du classificateur bayésien naïf :", precision_naive_bayes)


plt.show() 

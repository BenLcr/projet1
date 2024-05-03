#%%
import numpy as np
import scipy.io 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

def accuracy_cm(confusion_matrix):
    correct_predictions = np.diag(confusion_matrix).sum()
    total_predictions = confusion_matrix.sum()
    accuracy = correct_predictions / total_predictions
    return accuracy

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

plt.figure()

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


#Predict avec ADL
X2_clean = X2.iloc[:,4:]
X2_adl = adl.transform(X2_clean)
rand = np.random.randn(int(np.shape(X2_adl)[0]))

X2_pred = adl.predict(X2_clean) #Predict class labels for samples in X

cm = confusion_matrix(X2["Class"], X2_pred)
disp = ConfusionMatrixDisplay(cm)
print("accuracy adl", accuracy_cm(cm))
disp.plot(cmap='hot')

#Predict avec methode knvoisins
x = X1.iloc[:, 4:]
y = X1['Class']
#x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

x_train = X1.iloc[:, 4:]
x_test = X2.iloc[:, 4:]
y_train = X1['Class']
y_test = X2['Class']

tab_accuracy = []
confidence_intervals = []

for i in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_cm(cm)
    tab_accuracy.append(accuracy)
    n = len(y_test)
    p = accuracy

plt.figure(figsize=(8, 6))
#plt.errorbar([i for i in range(1, 16)], tab_accuracy, yerr=np.transpose(confidence_intervals), fmt='-o')
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("Accuracy du modèle en fonction de la valeur de k avec intervalle de confiance")
plt.grid(True)
plt.show()

#disp = ConfusionMatrixDisplay(cm)
#disp.plot(cmap='hot')

#Precision
#precision = knn.score(x_test, y_test)
#print("Précision du classificateur kvoisins :", precision)

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


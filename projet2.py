from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# Chargement et préparation des données
Xdf = pd.read_csv('dataset.csv')
x = Xdf.iloc[:, :12]
y = Xdf['target']
Xdf = Xdf.dropna()

# Visualisation de la répartition des classes
sns.countplot(x='target', data=Xdf)
plt.title("Répartition des classes dans le dataset")
plt.show()

print("Nombre de personnes malades :", len(Xdf[Xdf.target == 0]))
print("Nombre de personnes non malades :", len(Xdf[Xdf.target == 1]))

# Affichage de la matrice de corrélation
correlation_matrix = Xdf.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
plt.tight_layout()
plt.title("Matrice de corrélation")
plt.show()

# LDA
adl = LinearDiscriminantAnalysis()
Xadl = adl.fit_transform(x, y)

# Séparation des données
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Prédiction avec LDA
yadl_pred = adl.predict(x_test)

# Affichage de la matrice de confusion pour LDA
CM_ADL = confusion_matrix(y_test, yadl_pred)
disp = ConfusionMatrixDisplay(CM_ADL)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - Analyse Discriminante Linéaire")
plt.show()

# Entraînement d'un arbre de décision sur le dataset
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)

# Prédiction avec l'arbre de décision
y_pred = clf.predict(x_test)

# Affichage de la matrice de confusion pour l'arbre de décision
CM_CLF = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(CM_CLF)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - Arbre de Décision")
plt.show()

# Affichage de l'arbre de décision
plt.figure(figsize=(12, 10))
tree.plot_tree(clf, filled=True, feature_names=x.columns, class_names=['Classe 0', 'Classe 1'])
plt.title("Arbre de Décision")
plt.show()

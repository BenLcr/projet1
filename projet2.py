from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def accuracy_cm(confusion_matrix):
    correct_predictions = np.diag(confusion_matrix).sum()
    total_predictions = confusion_matrix.sum()
    accuracy = correct_predictions / total_predictions
    return accuracy

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

rand = np.random.randn(int(np.shape(Xadl)[0]))
plt.scatter(Xadl, rand, c=y)
plt.title("ADL")
plt.show()

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
print("accuracy CM_ADL", accuracy_cm(CM_ADL))

# Entraînement d'un arbre de décision sur le dataset
clf = tree.DecisionTreeClassifier()

# Test des meilleurs paramètres 
param_grid = {
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': np.logspace(-4, 0, 50)
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=5, scoring='accuracy')

# Entraînement de GridSearchCV
grid_search.fit(x_train, y_train)

# Affichage des meilleurs paramètres
print(f"Meilleurs paramètres: {grid_search.best_params_}")

# Liste des valeurs de ccp_alpha à tester
ccp_alphas = np.logspace(-3, 0, 100)

# Stockage des résultats
train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha, criterion='gini', random_state=0)
    clf.fit(x_train, y_train)
    train_scores.append(clf.score(x_train, y_train))
    test_scores.append(clf.score(x_test, y_test))

# Visualisation des scores
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, label='Train Accuracy', marker='o')
plt.plot(ccp_alphas, test_scores, label='Test Accuracy', marker='o')
plt.xscale('log')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs ccp_alpha, criterion=gini')
plt.legend()
plt.grid(True)
plt.show()

train_scores = []
test_scores = []

for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(ccp_alpha=ccp_alpha, criterion='entropy', random_state=0)
    clf.fit(x_train, y_train)
    train_scores.append(clf.score(x_train, y_train))
    test_scores.append(clf.score(x_test, y_test))

# Visualisation des scores
plt.figure(figsize=(10, 6))
plt.plot(ccp_alphas, train_scores, label='Train Accuracy', marker='o')
plt.plot(ccp_alphas, test_scores, label='Test Accuracy', marker='o')
plt.xscale('log')
plt.xlabel('ccp_alpha')
plt.ylabel('Accuracy')
plt.title('Accuracy vs ccp_alpha, criterion=entropy')
plt.legend()
plt.grid(True)
plt.show()

# Utilisation des meilleurs paramètres
clf = tree.DecisionTreeClassifier(ccp_alpha=0.00, criterion='gini')
clf = clf.fit(x, y)

importances = clf.feature_importances_
print("Importances des caractéristiques :", importances)

feature_names = x.columns
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = plt.subplots()
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances")
ax.set_ylabel("Diminution moyenne des impuretés")
fig.tight_layout()

tree_depth = clf.get_depth()
print("Profondeur de l'arbre :", tree_depth)

# Prédiction avec l'arbre de décision
y_pred = clf.predict(x_test)

# Affichage de la matrice de confusion pour l'arbre de décision
CM_CLF = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(CM_CLF)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - Arbre de Décision")
plt.show()
print("accuracy CM_CLF", accuracy_cm(CM_CLF))

# Affichage de l'arbre de décision
plt.figure(figsize=(12, 10))
tree.plot_tree(clf, filled=True, feature_names=x.columns, class_names=['Classe 0', 'Classe 1'])
plt.title("Arbre de Décision")
plt.show()

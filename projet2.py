from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
"""iris = load_iris()
X, Y = iris.data, iris.target

clf = tree.DecisionTreeClassifier(max_depth=4) #, min_samples_leaf=40
clf = clf.fit(X, Y)

plt.figure(figsize=(12, 10))
tree.plot_tree(clf)"""
  

Xdf = pd.read_csv('dataset.csv')
y = Xdf['target']

Xdf = Xdf.dropna()

Xdf.target.value_counts()
sns.countplot(x='target', data=Xdf)

print("Nombre de personnes malades :", len(Xdf[Xdf.target == 0]))
print("Nombre de personnes non malades :", len(Xdf[Xdf.target == 1]))

#%% Analyse univariée des variables
# Xdf.boxplot(by='target')
# Xdf.groupby('target').mean()

#%% Analyse bivariée des variables
correlation_matrix = Xdf.corr()

# Heatmap avec seaborn
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
plt.tight_layout()
plt.title("Matrice de corrélation")
plt.show()

#%% Analyse multivariée 
adl = LinearDiscriminantAnalysis()
Xadl = adl.fit_transform(Xdf.iloc[:,:12], Xdf['target'])

#%%  
plt.bar(np.arange(1, len(adl.explained_variance_ratio_)+1), adl.explained_variance_ratio_)
plt.plot(np.arange(1, len(adl.explained_variance_ratio_)+1), np.cumsum(adl.explained_variance_ratio_), color='r')
plt.ylabel("Variance expliquée en ratio et cumul")
plt.xlabel("Nombre de facteurs")
plt.show()

#%% 
rand = np.random.randn(int(np.shape(Xadl)[0]))
scatter = plt.scatter(Xadl[:, 0], rand, c=Xdf['target'])
plt.xlabel('Composante discriminante 1')
plt.ylabel('Composante discriminante 2')
plt.title('LDA of Hearth diseases dataset')
plt.legend(*scatter.legend_elements())
plt.show()

corvar = np.zeros((Xdf.shape[1] - 1, 2))
for i in range(Xdf.shape[1] - 1):
    corvar[i, 0] = np.corrcoef(Xadl[:, 0], Xdf.iloc[:, i])[0, 1]

fig, axes = plt.subplots(figsize=(8, 8))
axes.set_xlim(-1, 1)
axes.set_ylim(-1, 1)
plt.scatter(corvar[:, 0], corvar[:, 1])

# Affichage des étiquettes (noms des variables)
for j in range(Xdf.shape[1] - 1):
    plt.annotate(Xdf.columns[j], (corvar[j, 0], corvar[j, 1]))

# Ajout des axes
plt.plot([-1, 1], [0, 0], color='silver', linestyle='-', linewidth=1)
plt.plot([0, 0], [-1, 1], color='silver', linestyle='-', linewidth=1)

# Ajout d'un cercle
cercle = plt.Circle((0, 0), 1, color='blue', fill=False)
axes.add_artist(cercle)

x_train, x_test, y_train, y_test = train_test_split(Xdf.iloc[:, :12], Xdf['target'], test_size=0.2, random_state=0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(Xdf, Xdf['target'])

y_pred = clf.predict(x_test)

CM_CLF = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(CM_CLF)
disp.plot(cmap='hot')

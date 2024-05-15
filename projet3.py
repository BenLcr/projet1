import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def accuracy_cm(confusion_matrix):
    correct_predictions = np.diag(confusion_matrix).sum()
    total_predictions = confusion_matrix.sum()
    accuracy = correct_predictions / total_predictions
    return accuracy

# Chargement et préparation des données
Xdf = pd.read_csv('database.csv')
missing_values = Xdf.isnull().sum()
print(f"Nombre de données manquantes : {missing_values.sum()}")

Xdf = Xdf.dropna()

x = Xdf.iloc[:, 3:]
y = Xdf['Classe']

# Conversion des données en 0 naturel 1 artificiel
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = 1 - y

# Analyse données
# Visualisation de la répartition des classes
sns.countplot(x=y, data=Xdf)
plt.title("Répartition des classes dans le dataset")
plt.show()

print("Nombre de paysages naturels :", (y == 0).sum())
print("Nombre de paysages artificiels :", (y == 1).sum())

# Affichage de la matrice de corrélation
correlation_matrix = Xdf.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, vmin=-1, vmax=1, center=0, cmap='coolwarm')
plt.title("Matrice de corrélation")
plt.tight_layout()
plt.show()

# LDA
adl = LinearDiscriminantAnalysis()
X_adl = adl.fit_transform(x, y)

rand = np.random.randn(X_adl.shape[0])
plt.scatter(X_adl, rand, c=y, cmap='viridis', edgecolor='k')
plt.title("ADL")
plt.xlabel("Composante discriminante")
plt.show()

# Séparation des données
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0, stratify=y)
x_train2, x_validation, y_train2, y_validation = train_test_split(x_train, y_train, test_size=0.33, random_state=0, stratify=y_train)

# Prédiction avec LDA
yadl_pred = adl.predict(x_test)

# Affichage de la matrice de confusion pour LDA
CM_ADL = confusion_matrix(y_test, yadl_pred)
disp = ConfusionMatrixDisplay(CM_ADL)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - Analyse Discriminante Linéaire")
plt.show()
print("accuracy CM_ADL", accuracy_cm(CM_ADL))

# Training MLP
tab_accuracy =  []
tab_conf = []
nb_neurones = np.linspace(1, 21, 1)
alpha = np.linspace(0.00001, 0.001, 0.00001)

for i in range(0, max(alpha, len(nb_neurones))):
    clf_MLP = MLPClassifier(random_state=0, hidden_layer_sizes=(nb_neurones,), alpha=alpha, max_iter=300).fit(x_train2, y_train2)
    y_pred = clf_MLP.predict(x_validation)
    tab_accuracy.append(clf_MLP.score(x_test, y_test))
    tab_conf.append()

clf_MLP = MLPClassifier(random_state=0, hidden_layer_sizes=(2,), alpha=0.0001, max_iter=300).fit(x_train, y_train)

# Prédiction avec l'arbre de décision
y_pred = clf_MLP.predict(x_test)

# Affichage de la matrice de confusion pour MLP
CM_MLP = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(CM_MLP)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - MLP")
plt.show()
print("Score MLP", clf_MLP.score(x_test, y_test))
print("accuracy CM_CLF", accuracy_cm(CM_MLP))

# Training SVM
C = 1.0
kernel = 'linear' #{'linear', 'poly', 'rbf', 'sigmoid'}
clf_SVM = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel))
clf_SVM.fit(x_train, y_train)

# Prédiction avec l'arbre de décision
y_pred = clf_SVM.predict(x_test)

# Affichage de la matrice de confusion pour SVM
CM_SVM = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(CM_SVM)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - SVM")
plt.show()
print("Score SVM", clf_SVM.score(x_test, y_test))
print("accuracy CM_SVM", accuracy_cm(CM_SVM))

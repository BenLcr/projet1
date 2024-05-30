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
from sklearn.exceptions import ConvergenceWarning
import warnings as w
w.simplefilter("ignore", category=ConvergenceWarning)

# Fonction pour tracer les résultats de la recherche en grille avec des barres d'erreur
def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1, 1)

    # Param1 is the X-axis, Param 2 is represented as a different scatter plot (color points)
    for idx, val in enumerate(grid_param_2):
        ax.errorbar(grid_param_1, scores_mean[idx, :], yerr=scores_sd[idx, :], fmt='o', label=name_param_2 + ': ' + str(val), capsize=5)

    plt.tight_layout()
    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=5)
    ax.grid(True)
    plt.show()

def accuracy_cm(confusion_matrix):
    correct_predictions = np.diag(confusion_matrix).sum()
    total_predictions = confusion_matrix.sum()
    accuracy = correct_predictions / total_predictions
    return accuracy

# Chargement et prÃ©paration des donnÃ©es
Xdf = pd.read_csv('database.csv')
missing_values = Xdf.isnull().sum()
print(f"Nombre de donnÃ©es manquantes : {missing_values.sum()}")

Xdf = Xdf.dropna()

x = Xdf.iloc[:, 3:]
y = Xdf['Classe']

# Conversion des donnÃ©es en 0 naturel 1 artificiel
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = 1 - y

# Analyse donnÃ©es
# Visualisation de la rÃ©partition des classes
sns.countplot(x=y, data=Xdf)
plt.title("RÃ©partition des classes dans le dataset")
plt.show()

print("Nombre de paysages naturels :", (y == 0).sum())
print("Nombre de paysages artificiels :", (y == 1).sum())

# LDA
adl = LinearDiscriminantAnalysis()
X_adl = adl.fit_transform(x, y)

rand = np.random.randn(X_adl.shape[0])
plt.scatter(X_adl, rand, c=y, cmap='viridis', edgecolor='k')
plt.title("ADL")
plt.xlabel("Composante discriminante")
plt.show()

# SÃ©paration des donnÃ©es
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0, stratify=y)
x_train2, x_validation, y_train2, y_validation = train_test_split(x_train, y_train, test_size=0.33, random_state=0, stratify=y_train)

# PrÃ©diction avec LDA
yadl_pred = adl.predict(x_test)

# Affichage de la matrice de confusion pour LDA
CM_ADL = confusion_matrix(y_test, yadl_pred, normalize='all')
disp = ConfusionMatrixDisplay(CM_ADL)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - Analyse Discriminante LinÃ©aire")
plt.show()
print("accuracy CM_ADL", accuracy_cm(CM_ADL))

# Training MLP
# nb_neurones = [n for n in range(1, 18, 4)]
# print(nb_neurones)
# alpha = np.linspace(0.00001, 0.01, 10)
# tab_accuracy = np.zeros((len(alpha), len(nb_neurones)))
# print(tab_accuracy)

#for i in range(0, len(alpha) - 1):
#    for j in range(0, len(nb_neurones) - 1):
#        clf_MLP = MLPClassifier(random_state=0, hidden_layer_sizes=(int(nb_neurones[i]),), alpha=alpha[j], max_iter=300).fit(x_train2, y_train2)
#        tab_accuracy[i][j] = clf_MLP.score(x_validation, y_validation)

# On va faire intervalle de confiance = moyenne des accuracy - val max/val min 

clf_MLP = MLPClassifier(random_state=0, max_iter=300)

# DÃ©finir la grille des hyperparamÃ¨tres Ã  tester
param_grid = {
    'hidden_layer_sizes': [(n,) for n in range(1, 20, 2)],  # De 1 a 2 neurones dans une seule couche
    'alpha': np.linspace(1e-5, 0.1, 30)  # 10 valeurs de alpha entre 0.00001 et 0.001
}

grid_search = GridSearchCV(estimator=clf_MLP, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train2, y_train2)

# Afficher les meilleurs paramÃ¨tres
print(f"Meilleurs parametres: {grid_search.best_params_}")
plot_grid_search(grid_search.cv_results_, param_grid['alpha'], param_grid["hidden_layer_sizes"], 'alpha', 'hidden_layer_sizes')

clf_MLP = MLPClassifier(random_state=0, hidden_layer_sizes=grid_search.best_params_["hidden_layer_sizes"], alpha=grid_search.best_params_["alpha"], max_iter=300).fit(x_train, y_train)

# PrÃ©diction avec MLP
y_pred = clf_MLP.predict(x_test)

# Affichage de la matrice de confusion pour MLP
CM_MLP = confusion_matrix(y_test, y_pred, normalize='all')
disp = ConfusionMatrixDisplay(CM_MLP)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - MLP")
plt.show()
print("Score MLP", clf_MLP.score(x_test, y_test))
print("accuracy CM_CLF", accuracy_cm(CM_MLP))

# Training SVM
clf_SVM = make_pipeline(StandardScaler(), SVC())

# DÃ©finir la grille des hyperparamÃ¨tres Ã  tester
param_grid = {
    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # DiffÃ©rents noyaux Ã  tester
    'svc__C': np.linspace(0.001, 100.0, 30)  # DiffÃ©rentes valeurs de C Ã  tester
}

grid_search = GridSearchCV(estimator=clf_SVM, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(x_train2, y_train2)

# Afficher les meilleurs paramÃ¨tres
print(f"Meilleurs paramÃ¨tres: {grid_search.best_params_}")
plot_grid_search(grid_search.cv_results_, param_grid['svc__C'], param_grid["svc__kernel"], 'C', 'Kernel')

C = grid_search.best_params_['svc__C']
kernel = grid_search.best_params_["svc__kernel"] #{'linear', 'poly', 'rbf', 'sigmoid'}
clf_SVM = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel))
clf_SVM.fit(x_train, y_train)

# PrÃ©diction avec l'arbre de dÃ©cision
y_pred = clf_SVM.predict(x_test)

# Affichage de la matrice de confusion pour SVM
CM_SVM = confusion_matrix(y_test, y_pred, normalize='all')
disp = ConfusionMatrixDisplay(CM_SVM)
disp.plot(cmap='hot')
plt.title("Matrice de confusion - SVM")
plt.show()
print("Score SVM", clf_SVM.score(x_test, y_test))
print("accuracy CM_SVM", accuracy_cm(CM_SVM))

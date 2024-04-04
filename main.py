import scipy.io 
import pandas as pd

data = scipy.io.loadmat('data.mat') 

X1 = pd.DataFrame(data["data1"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X2 = pd.DataFrame(data["data2"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])
X3 = pd.DataFrame(data["data3"], columns=['NumPulse', 'Class', 'Debut', 'Fin', 'MaxInter', 'DiffMax', 'DiffMin', 'DiffPress', 'DiffLarg'])

#print("Le nombre d'artéfacts dans le jeu de données 1 est : ", Nb_pulse)
print (f"Nombre de pulses dans le jeu de données 1 : {len(X1[X1['Class'] == 1])}")
print (f"Nombre de pulses dans le jeu de données 2 : {len(X2[X2['Class'] == 1])}")
print (f"Nombre de pulses dans le jeu de données 3 : {len(X3[X3['Class'] == 1])}")


from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from boosting import Boosting
import numpy as np
import matplotlib.pyplot as plt

# Générer des données aléatoires pour la classification binaire
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialiser l'algorithme de Boosting avec 50 itérations
boosting = Boosting(T=50)

# Entraîner l'algorithme de Boosting sur l'ensemble d'entraînement
boosting.fit(X_train, y_train)
y_predict=boosting.predict(X_test)

# Calculer la précision en fonction du nombre d'itérations
accuracies = []
for t in range(1, boosting.T+1):
    y_pred = boosting.predict(X_test, t)
    accuracy = 100* accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

print('Accuracy: %.2f'%max(accuracies),'%')


if __name__ == '__main__':
    pc = Boosting(T=50)
    pc.fit(X_train, y_train)
    pc.predict(X_test)
    #pc.plot_performance(X_test,y_predict)


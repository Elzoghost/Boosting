from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

class Boosting():
    def __init__(self, T):
        self.T = T
        self.classifiers = []
        self.alphas = []
       
    def fit(self, X, y):
        n_samples, n_features = X.shape
       
        # Initialiser les poids des exemples
        sample_weights = np.ones(n_samples) / n_samples
       
        for t in range(self.T):
            # Appliquer les poids aux exemples
            X_weighted = X * sample_weights[:, np.newaxis]
           
            # Trouver le classifieur faible avec la plus faible erreur pondérée
            classifier = DecisionTreeClassifier(max_depth=1)
            classifier.fit(X_weighted, y, sample_weight=sample_weights)
            predictions = classifier.predict(X)
            weighted_error = np.sum(sample_weights * (predictions != y))
            error = weighted_error / np.sum(sample_weights)
            alpha = np.log((1 - error) / error)
           
            # Mettre à jour les poids des exemples
            sample_weights = sample_weights * np.exp(-alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)
           
            # Ajouter le classifieur faible et son poids associé
            self.classifiers.append(classifier)
            self.alphas.append(alpha)
       
    def predict(self, X, T=None):
        if T is None:
            T = self.T
        predictions = np.zeros(X.shape[0])
        for t in range(T):
            classifier = self.classifiers[t]
            alpha = self.alphas[t]
            predictions += alpha * classifier.predict(X)
        return np.sign(predictions)
      
    def plot_performance(self, X, y):
        # Plot performance
        fig, ax = plt.subplots()
        ax.plot(X, y, marker='o')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Number of updates')
        ax.set_title('Perceptron')
        plt.show()

     

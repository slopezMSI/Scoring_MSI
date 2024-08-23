# Scoring_MSI - Niveau Medium

## Introduction
Le niveau "medium" du projet Scoring_MSI approfondit l'analyse et l'optimisation des modèles de prédiction pour les données non quantitatives, en se concentrant sur l'exemple des molécules chimiques.
Ce niveau introduit des techniques plus avancées d'analyse de données et d'optimisation des modèles.

## Contenu
Ce dossier contient XX notebooks Jupyter:
1. `data_cleansing_fingerprint.ipynb`: optimisation des ensembles de données, avec un focus sur les classes sous-représentées, pour la représentation "fingerprints"
2. `data_cleansing_fingerprint_family.ipynb`: idem pour la représentation fingerprints combinée à la famille chimique.
3. `gridsearch_fingerprint.ipynb`: optimisation des paramètres des algorithmes via la technique de grid search, à partir de la représentation "fingerprints".
4. `gridsearch_fingerprint_family.ipynb`: idem pour la représentation fingerprints combinée à la famille chimique.

## Objectifs
- Améliorer les bases de données pour l'entrainement et la validatinon, notamment pour traiter les problèmes potentiels comme les classes sous-représentées.
- Optimiser les prformances des modèles en utilisant des techniques de recherche systématique des meilleurs paramètres.

## Prérequis
- Python 3.7+
- Jupyter Notebook
- Bibliothèques: rdkit, deepchem, sklearn, pandas, numpy, matplotlib, seaborn

## Utilisation
- Assurez-vous d'avoir complété le niveau "basic" du projet.
- Installez toutes les dépendances nécessaires.
- Ouvrez les notebooks dans Jupyter et exécutez les cellules dans l'ordre.

## Description des Notebooks
### `data_cleansing_fingerprint.ipynb`
Ce notebook se concentre sur :
- L'identification et le traitement des classes sous-représentées.
La comparaison des performances avant et après optimisation peut se faire en se référant aux résultats obtenus dans les notebooks "basic".
### `gridsearch_fingerprint.ipynb`
Ce notebook couvre :
- L'introduction à la technique de grid search pour l'optimisation des hyperparamètres.
- L'application du grid search sur différents algorithmes (par exemple, Random Forest, régression logistique, SVR etc.).
- L'analyse de l'importance des différents paramètres sur les performances des modèles.

## Données
Les notebooks utilisent le fichier sweetnersDB.xlsx qui doit être placé dans le dossier data/ à la racine du projet. Des ensembles de données prétraités issus du niveau "basic" peuvent également être utilisés.

## Résultats Attendus
À la fin de chaque notebook, vous devriez obtenir :
- Une compréhension approfondie de la structure et des défis de votre jeu de données.
- Des modèles optimisés avec des performances améliorées.

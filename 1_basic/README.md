# Scoring-MSI - Niveau Basic

## Introduction
Ce niveau "basic" du projet Scoring_MSI propose une approche initiale pour le traitement et l'analyse de données non quantitatives en utilisant des molécules chimiques comme exemple.
Il s'agit d'une introduction aux techniques de régression et de classification appliquées aux données chimiques.


## Contenu
Ce dossier contient quatre notebooks Jupyter et un fichier .py:
- `representation_fingerprint.ipynb` : Utilise une représentation simple des molécules basée sur les "fingerprints" pour effectuer des tâches de régression et de classification.
- 'methods_fingerprint.py' : propose les méthodes automatisées issues de representation_fingerprint.ipynb, afin de rendre la lecture des prochains notebooks plus agréable.
- 'regression_fingerprint_family.ipynb' : Propose une approche plus complexe en combinant les fingerprints avec l'information de la famille chimique.
- 'regression_fingerprint_family.ipynb' : Propose une approche plus complexe en combinant les fingerprints avec l'information de la famille chimique.
- 'regression_fingerprint_family.ipynb' : Propose une approche plus complexe en combinant les fingerprints avec l'information de la famille chimique.

## Objectifs
- Fournir un point de départ pour l'analyse de données non quantitatives
- Comparer l'efficacité de différentes représentations moléculaires
- Démontrer l'utilisation de base des algorithmes de machine learning sur des données chimiques.

## Prérequis
- Python 3.7+
- jupyter notebook
- Biibliothèques: rdkit, deepchem, sklearn, pandas, numpy

## Utilisation
1. Assurez-vous d'avoir installé toutes les dépendances nécessaires
2. Ouvrer le notebook de votre choix dans Jupyter (ou google collab)
3. Exécutez les cellules dans l'ordre pour voir les résultats.

## Description des notebooks
### representation_fingerprints.ipynb
Ce notebook se concentre sur: 
- la création de fingerprints moléculaires. Une alternative pourrait être les graphes.
- L'entrainement d'un modèle de regréssion pour prédire le logSw (méthode SVR)
- L'entrainement de modèles de classification (régression logistique, arbre de décision, random forest) pour catégoriser les molécules, en seuillant la valeur logSw à 0.
- L'évaluation des performances des modèles.

### representation_fingerprint_family.ipynb
Ce notebook étend l'approche précédente en;
- intégrant l'information de la famille chimique aux fingerprints avec une représentation one_hot encoder
- comparant les performances avec cette information supplémentaire

### representation_fingerprint_family_V2.ipynb
Ce notebook propose une méthode ensembliste pour associer la famille chimique aux fingerprints.

### representation_fingerprint_familyl_V3.ipynb
Ce notebook propose d'utiliser un réseau de neurones pré-entrainé pour obtenir une représentation des molécules à partir de la famille chimique et des fingerprints.


## Données
Les notebooks utilisent le fichier `sweetnersDB.xlsx" qui doit être placé dans le dossier `data/`à la racine du projet.

## Résultats attendus
A la fin de chaque notebook, vous devriez obtenir des scores de performance pour différents modèles.

## Prochaines étapes
Après avoir exploré ce niveau basic, vous pouvez passer au niveau "medium" pour des analyses plus approfondes et des techniques d'optimisation.


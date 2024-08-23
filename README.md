# Scoring_MSI: Cadre réglementé pour l'IA avec des données non quantitatives.
## 1.	Introduction
Ce projet vise à établir un cadre méthodologique pour le traitement et l'analyse de données non quantitatives, spécifiquement conçu pour la MSI. Il propose une approche progressive pour le scoring de molécules chimiques, allant de méthodes basiques à des techniques plus avancées.
## 2. Objectifs
-	Développer un cadre documenté pour le traitement et l’analyse de données non quantitatives
-	Fournir un outil accessible aux ingénieurs MSI, y compris ceux n’ayant pas un profil IA / machine learning
-	Permettre aux équipes de tester des approches simples avant de faire appel à des experts en IA si nécessaire

## 3. Structure du projet

CLe projet est organisé en 3 niveaux de complexité croissante.
### 3.1. Niveau "basic"
- `basic/0_tuto_basic_finger_prints.ipynb`: régression et scoring (classificationà avec représentation des molécules chimiques par les "fingerprints"
- `basic/0_tuto_basic_fingerprint_et_famille.ipynb`: les mêmes algorithmes sont appliqués, mais la représentation des molécules chimiques intègre la famille chimique. 3 versions sont proposées pour représenter la famille chimique.
  - `basic/0_tuto_basic_fingerprint_et_famille.ipynb`: onehot encoder
  - `basic/0_tuto_basic_fingerprint_et_famille_v2.ipynb`: méthode ensembliste
  - `basic/0_tuto_basic_fingerprint_et_famille_V3.ipynb`: représentation obtenue par apprentissage d'un réseau de neuron pré-entrainé.

### Niveau "medium"
- `medium/0_tuto_medium_remove.ipynb`: Analyse approfondie des données, focus sur les classes sous-représentées
- `medium/0_tuto_medium_gridsearch.ipynb`: Optimisation des paramètres via grid-search

### Niveau "Premium"
- `premium/0_tuto_premium.ipynb`: Validation croisée avec grid-search sur chaque pli.

## 4. Installation
### 4.1. Prérequis
- Python 3.7+
- Librairies requises
  - rdkit
  - deepchem
  - sklearn
  - pandas
  - numpy

### 4.2. Instructions d'installation
Le fichier de données est sweetDB.xlsx.
Clonez ce dépôt :
text
git clone https://github.com/votre-username/Scoring_MSI.git

Assurez-vous que le fichier sweetnersDB.xlsx est présent dans le dossier data/.

## 5. Utilisation
Chaque notebook peut être exécuté indépendamment. Commencez par le niveau "Basic" pour une compréhension initiale, puis progressez vers les niveaux "Medium" et "Premium" pour des analyses plus approfondies.

## 6. Description du Projet
Le projet utilise des formules chimiques et des informations sur la famille chimique pour :
Classer les molécules en 2 catégories
Prédire leur valeur de logSw

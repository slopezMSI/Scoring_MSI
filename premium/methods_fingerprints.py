import deepchem as dc
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from deepchem.feat import MolGraphConvFeaturizer
from deepchem.feat import CircularFingerprint
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from deepchem.feat import ConvMolFeaturizer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
import deepchem as dc
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def load_data(file_name):
    df = pd.read_excel(file_name, engine='openpyxl')
    print(df.columns)
    df_train = df[df.Dataset=='Training']
    df_test = df[df.Dataset=='Test']
    
    smiles_tr = df_train['Smiles']
    smiles_test = df_test['Smiles']
    
    logSweeter_tr = df_train['logSw']
    logSweeter_test = df_test['logSw']
    
    
    families_tr = pd.factorize(df_train['Chemical family'])[0]
    families_test = pd.factorize(df_test['Chemical family'])[0]
 
    return smiles_tr,smiles_test, logSweeter_tr, logSweeter_test, families_tr, families_test

def load_data_remove(file_name):
    df = pd.read_excel(file_name, engine='openpyxl')
    print(df.columns)
    df = df[df.groupby('Chemical family')['Chemical family'].transform('count')>5]
    df_train = df[df.Dataset=='Training']
    df_test = df[df.Dataset=='Test']
    
    smiles_tr = df_train['Smiles']
    smiles_test = df_test['Smiles']
    
    logSweeter_tr = df_train['logSw']
    logSweeter_test = df_test['logSw']
    
    
    families_tr = pd.factorize(df_train['Chemical family'])[0]
    families_test = pd.factorize(df_test['Chemical family'])[0]
 
    return smiles_tr,smiles_test, logSweeter_tr, logSweeter_test, families_tr, families_test

def load_data_full(file_name):
    df = pd.read_excel(file_name, engine='openpyxl')
    print(df.columns)
    df = df[df.groupby('Chemical family')['Chemical family'].transform('count')>5]
    
    smiles = df['Smiles']
    logSweeter = df["logSw"]
    families = pd.factorize(df['Chemical family'])[0]
    
    X = pd.DataFrame({
        'Smiles': smiles,
        "families": families
    })
    
    return X, logSweeter
    
    
def prepare_fingerprint(smiles_tr,smiles_test):
    featurizer = CircularFingerprint(size=1024)
    
    X_tr = featurizer.featurize(smiles_tr)
    X_test = featurizer.featurize(smiles_test)
    
    
    # Normaliser les données
    scaler_X = StandardScaler()
    
    
    X_train_scaled = scaler_X.fit_transform(X_tr)
    # Évaluer le modèle
    X_test_scaled = scaler_X.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler_X
    
    
    
def featurize_smiles(smiles_list, radius=2, nBits=1024):
    """Convertit une liste de SMILES en empreintes Morgan."""
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprints.append(np.array(fp))
        else:
            # Gérer les SMILES invalides
            fingerprints.append(np.zeros(nBits))
    return np.array(fingerprints)
    
def prepare_fingerprint_update(smiles_tr,smiles_test):
    
    # Featuriser les ensembles d'entraînement et de test
    X_tr = featurize_smiles(smiles_tr)
    X_test = featurize_smiles(smiles_test)
    
    # Normaliser les données
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_tr)
    X_test_scaled = scaler_X.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler_X
    
def prepare_famille_chimique(families_tr, families_test):
    
    # One-hot encoding de la classe cible
    onehot_encoder = OneHotEncoder(sparse=False)
    onehot_tr = onehot_encoder.fit_transform(pd.DataFrame(families_tr))
    onehot_test = onehot_encoder.fit_transform(pd.DataFrame(families_test))
    
    '''# Créer un DataFrame avec l'encodage one-hot
    tr_onehot_df = pd.DataFrame(
        onehot_tr,
        columns=[f'target_{cls}' for cls in onehot_encoder.categories_[0]]
    )
    test_onehot_df = pd.DataFrame(
        onehot_test,
        columns=[f'target_{cls}' for cls in onehot_encoder.categories_[0]]
    )'''
  
    
    return onehot_tr,onehot_test
    
def prepare_targets(logSweeter_tr,logSweeter_test):
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(logSweeter_tr.values.reshape(-1, 1)).ravel()
    y_test_scaled = scaler_y.transform(logSweeter_test.values.reshape(-1, 1)).ravel()
    
    return y_train_scaled,y_test_scaled, scaler_y
    





def method_SVR(X_tr,y_tr):
    
    # Créer et entraîner le modèle SVR
    svr = SVR(kernel='rbf', C=100, gamma='scale', epsilon=0.1)
    svr.fit(X_tr, y_tr)
    # Évaluer le modèle
    score = svr.score(X_test, y_test)
    print(f"Score R² du modèle: {score:.2f}")
    
   

    
def prediction_SVR(X_tr,X_test): 
    
    
    
    # Faire des prédictions
    y_pred_train = svr.predict(X_tr)
    y_pred_test = svr.predict(X_test)
    
    
    # Inverser la normalisation pour obtenir les valeurs réelles
    y_pred_train = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1)).ravel()
    y_pred_test = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1)).ravel()
    y_train = scaler_y.inverse_transform(y_tr.reshape(-1, 1)).ravel()
    y_test = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        
    
    
def scoring(y_tr,y_test,X_tr,X_test,sweet_thr,model):
    sweet_thr = 0
    ytr = pd.DataFrame(y_tr,columns=['label'])
    target_tr =  ytr['label'].apply(lambda x: 0 if x <= sweet_thr else 1)
    ytest = pd.DataFrame(y_test,columns=['label'])
    target_test =  ytest['label'].apply(lambda x: 0 if x <= sweet_thr else 1)
    
    if model == "LogReg":
        model = LogisticRegression()
    elif model == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
    elif model == "RandomF":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_tr, target_tr)

    # Prédiction sur l'ensemble de test
    y_pred = model.predict(X_test)
    
    # Évaluation du modèle
    accuracy = accuracy_score(target_test, y_pred)
    conf_matrix = confusion_matrix(target_test, y_pred)
    class_report = classification_report(target_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    return accuracy,conf_matrix,class_report,y_pred,target_test
    
    
def scoring_GridSearch(y_tr,y_test,X_tr,X_test,sweet_thr,model):
    sweet_thr = 0
    ytr = pd.DataFrame(y_tr,columns=['label'])
    target_tr =  ytr['label'].apply(lambda x: 0 if x <= sweet_thr else 1)
    ytest = pd.DataFrame(y_test,columns=['label'])
    target_test =  ytest['label'].apply(lambda x: 0 if x <= sweet_thr else 1)
    
    if model == "LogReg":
        model = LogisticRegression(max_iter=10000, random_state=42)
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'l1_ratio': [0, 0.5, 1]  # Utilisé uniquement si penalty est 'elasticnet'
        }
    elif model == "DecisionTree":
        model = DecisionTreeClassifier(random_state=42)
        param_grid = {
            'criterion': ['gini', 'entropy'],  # 'log_loss' peut être utilisé pour des problèmes de classification multiclasse
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    elif model == "RandomF":
        model = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2']
        }
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=1, verbose=2)

        
        
    
    grid_search.fit(X_tr, target_tr)
    
    
    # Affichage des meilleurs paramètres et du meilleur score
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-validation Score: {grid_search.best_score_}")
    
    # Prédiction sur l'ensemble de test avec le meilleur modèle
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)


    
    # Évaluation du modèle
    accuracy = accuracy_score(target_test, y_pred)
    conf_matrix = confusion_matrix(target_test, y_pred)
    class_report = classification_report(target_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Classification Report:")
    print(class_report)
    
    return accuracy,conf_matrix,class_report,y_pred,target_test    
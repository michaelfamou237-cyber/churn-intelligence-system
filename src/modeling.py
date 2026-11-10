import numpy as np
import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


import os
import joblib
import json



# Importation de modèles pour benchmarking
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                              GradientBoostingClassifier, AdaBoostClassifier, 
                              BaggingClassifier)
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def benchmark_models(X_train, y_train, X_test, y_test, preprocessor) -> pd.DataFrame:
    """
    Entraîne tous les modèles disponibles et compare leurs métriques.
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'SGD Classifier': SGDClassifier(loss='log_loss', class_weight='balanced', random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'SVM': SVC(probability=True, class_weight='balanced', random_state=42), # probability=True requis pour predict_proba
        'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Bagging': BaggingClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=3, random_state=42),
        'LightGBM': LGBMClassifier(class_weight='balanced', random_state=42, verbose=-1),
        'CatBoost': CatBoostClassifier(verbose=0, auto_class_weights='Balanced', random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        # Création et entraînement du pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        # Prédictions
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        # Matrice de confusion
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Métriques
        results.append({
            'Modèle': name,
            'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
            'Recall': round(recall_score(y_test, y_pred), 3),
            'Precision': round(precision_score(y_test, y_pred), 3),
            'F1-Score': round(f1_score(y_test, y_pred), 3),
            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 3)
        })
        
    return pd.DataFrame(results).sort_values(by='F1-Score', ascending=False).reset_index(drop=True)



    

"""
    Otenir les coefficients de la regression logistique depuis le pipeline entrainé
"""
def get_logistic_coefficients(pipeline, top_n: int = 10) -> pd.DataFrame:

    # Récupérer les noms des colonnes après le One-Hot Encoding
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
    
    # Nettoyer les noms (enlever num__ et cat__)
    clean_names = [name.replace('num__', '').replace('cat__', '') for name in feature_names]
    
    # Récupérer les coefficients du modèle
    coefficients = pipeline.named_steps['model'].coef_[0]
    
    # Créer le DataFrame
    coef_df = pd.DataFrame({
        'Variable': clean_names,
        'Coefficient': coefficients
    })
    
    # Trier par valeur absolue pour voir les plus gros impacts
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values(by='Abs_Coefficient', ascending=False).drop(columns='Abs_Coefficient')
    
    # On sépare les effets positifs (risque) et négatifs (protection)
    top_risk = coef_df[coef_df['Coefficient'] > 0].head(top_n)
    top_protect = coef_df[coef_df['Coefficient'] < 0].head(top_n)
    
    return pd.concat([top_risk, top_protect]).reset_index(drop=True)




def save_model_pipeline(pipeline, filename: str, directory: str = '../artifacts/models') -> str:
    """Sauvegarde le pipeline complet (Preprocessing + Modèle) avec joblib. """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    joblib.dump(pipeline, filepath)
    print(f"Modèle sauvegardé : {filepath}")
    return filepath

def save_business_config(threshold: float, filename: str = 'config.json', directory: str = '../artifacts') -> str:
    """
    Sauvegarde les paramètres métier (comme le seuil) dans un fichier JSON.
    """
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    config = {
        "business_threshold": threshold,
        "model_name": "Logistic Regression",
        "metric_optimized": "Recall / F1-Score compromise" }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Configuration métier sauvegardée : {filepath}")
    return filepath




"""
Optimise la Régression Logistique sur le Train via GridSearchCV.
Teste la force de régularisation (C) avec une validation croisée 5-fold.
"""
def tune_logistic_regression(X_train, y_train, preprocessor):

    # 1. Création du pipeline de base
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    
    # 2. Grille des hyperparamètres à tester : Différentes forces de régularisation
    param_grid = {
        'model__C': [0.01, 0.1, 1, 10, 100]
    }
    
    # 3. Configuration du GridSearch
    # cv=5 : validation croisée à 5 plis
    # scoring='f1' : on optimise le F1-Score (compromis Précision/Rappel)
    # n_jobs=-1 : utilise tous les cœurs du processeur pour aller plus vite
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='f1', 
        n_jobs=-1,
        verbose=1
    )
    
    # 4. Lancement de la recherche sur le TRAIN UNIQUEMENT
    grid_search.fit(X_train, y_train)
    
    print(f"Meilleur paramètre C trouvé : {grid_search.best_params_['model__C']}")
    print(f"Meilleur F1-Score (Validation Croisée sur le Train) : {grid_search.best_score_:.3f}")
    
    
    return grid_search



import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score

def evaluate_final_model(pipeline, X_test, y_test, threshold=0.40) -> dict:
    """
    Évalue le modèle sur le Test (données non vues) avec le seuil métier.
    Retourne un dictionnaire des métriques.
    """
    # 1. Prédiction des probabilités
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # 2. Application du seuil métier
    y_pred = (y_proba >= threshold).astype(int)
    
    # 3. Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    # 4. Création du dictionnaire de résultats
    metrics = {
        'threshold_used': float(threshold),
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn),
        'Recall': round(recall_score(y_test, y_pred), 3),
        'Precision': round(precision_score(y_test, y_pred), 3),
        'F1_Score': round(f1_score(y_test, y_pred), 3),
        'ROC_AUC': round(roc_auc_score(y_test, y_proba), 3)
    }
    return metrics



"""
    Évalue le modèle final sur les données de test avec le seuil métier spécifié.
    Retourne un dictionnaire contenant les métriques.
"""

def evaluate_final_model(pipeline, X_test, y_test, threshold: float = 0.40) -> dict:

    # Prédictions de probabilité
    y_proba_test = pipeline.predict_proba(X_test)[:, 1]

    # Application du seuil métier
    y_pred_metier = (y_proba_test >= threshold).astype(int)

    # Matrice de confusion
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_metier).ravel()

    # Création du dictionnaire de métriques
    metrics = {
        'threshold_used': float(threshold),
        'TP': int(tp),
        'TN': int(tn),
        'FP': int(fp),
        'FN': int(fn),
        'Recall': round(recall_score(y_test, y_pred_metier), 3),
        'Precision': round(precision_score(y_test, y_pred_metier), 3),
        'F1_Score': round(f1_score(y_test, y_pred_metier), 3),
        'ROC_AUC': round(roc_auc_score(y_test, y_proba_test), 3)
    }
    
    return metrics












"""
    Sauvegarde le modèle, les métriques et la config du seuil dans le dossier spécifié.
"""
def save_final_artifacts(pipeline, metrics: dict, threshold: float, directory: str = '../model'):

    # Création du dossier s'il n'existe pas
    os.makedirs(directory, exist_ok=True)
    
    # Sauvegarde du modèle (Pipeline complet)
    model_path = os.path.join(directory, 'logistic_regression_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"Modèle sauvegardé : {model_path}")
    
    # Sauvegarde des métriques
    metrics_path = os.path.join(directory, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Métriques sauvegardées : {metrics_path}")
    
    # Sauvegarde de la configuration (seuil)
    config_path = os.path.join(directory, 'business_config.json')
    with open(config_path, 'w') as f:
        json.dump({"business_threshold": float(threshold)}, f, indent=4)
    print(f"Configuration sauvegardée : {config_path}")
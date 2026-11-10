
import pandas as pd
import numpy as np
from scipy import stats

from itertools import combinations
from typing import Dict, Any, List
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix


from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from itertools import combinations  # <--- L'IMPORT DOIT ETRE ICI, TOUT EN HAUT

# statistique descriptive de la variable cible Churn
def descriptive_churn(df:pd.DataFrame)-> None:
    effectifs = df['Churn'].value_counts()
    pourcentage = df['Churn'].value_counts(normalize=True)*100
    resume_churn = pd.DataFrame({
    'Effectifs' : effectifs,
    'Pourcentage' : pourcentage
    })
    return resume_churn

# statistiques descriptives des variables numeriques
def descriptive_variables_numeriques(df:pd.DataFrame)-> None:
    num_col = ['tenure', 'MonthlyCharges', 'TotalCharges']
    return df[num_col].describe().T

    
'''  ETUDE BIVARIE DES VARIABLES NUMERIQUES  '''
def analyse_numerique_by_target(df:pd.DataFrame, variables:list, target_col:str='Churn') -> pd.DataFrame:
    results = []
    m = len(variables)

    for var in variables:
        if var not in df.columns:
            raise ValueError(f"La variable {var} est absente du DataFrame")
            
        group_yes = df[df['Churn']=='Yes'][var].dropna()
        group_no = df[df['Churn']=='No'][var].dropna()

        n1, n2 = len(group_yes), len(group_no)

        u_stat, p_value = stats.mannwhitneyu(group_yes, group_no, alternative='two-sided')

        # taille d'effet : probabilite de supériorité
        ps = u_stat / (n1*n2)

        # correlation bisserale de rang 
        rg_biss = (2*ps)-1

        results.append({
            'variable': var,
            'Moyenne_Churn_yes' : group_yes.mean().round(4),
            'Moyenne_Churn_no' : group_no.mean().round(4),
            
            'Mediane_Churn_yes' : group_yes.median().round(4),
            'Mediane_Churn_no' : group_no.median().round(4),
            
            'Q1_Churn_yes' : group_yes.quantile(0.25).round(4),
            'Q3_Churn_no' : group_no.quantile(0.25).round(4),
            
            'Q1_Churn_yes' : group_yes.quantile(0.75).round(4),
            'Q3_Churn_no' : group_no.quantile(0.75).round(4),
            
            'U_stat' : u_stat,
            'p_value': p_value,
            'Probability_superiorite': ps,
            'correlation_bisserale_rang':rg_biss
        })
        resultat = pd.DataFrame(results)
        resultat = resultat.sort_values('p_value').reset_index(drop=True)
        
        my = len(resultat)
        p_value_adj = []

        # correction de Holm
        for i in range(my):
            p_adj = resultat.loc[i, 'p_value']*(my-i)
            p_value_adj.append(p_adj)

        p_value_adj = np.maximum.accumulate(p_value_adj)
        p_value_adj = np.minimum(p_value_adj, 1.0)

        resultat['p_value_adj_holm'] = p_value_adj
        resultat = resultat.sort_values('p_value_adj_holm')
    return resultat
        


def analyse_categorielle_by_target(df:pd.DataFrame, variables:list, target_col:str='Churn') -> pd.DataFrame:
    alpha = 0.05
    results = []
    for var in variables:
        if var not in df.columns:
            raise ValueError(f"La variable {var} est absente du DataFrame")

        contingency_table = pd.crosstab(df[var], df[target_col])
        chi_2,p_value,dof, expected = stats.chi2_contingency(contingency_table)

        if chi_2 < alpha :
            dec = "Pas lien existant"
        else:
            dec = " Existence de lien"
            

        # calcul du V cramer(effet de taille)
        n= contingency_table.sum().sum()
        min_dim= min(contingency_table.shape)-1
        cramers_v = np.sqrt(chi_2/ (n*min_dim)) if min_dim > 0 else 0

        results.append({
            'Variable': var,
            'Chi_2_stat' : chi_2.round(4),
            'p_value' : p_value,
            'Cramers_V': cramers_v.round(3),
            'Decision' : dec
        })

        resultat = pd.DataFrame(results)

        # correction de Holms
        resultat = resultat.sort_values('p_value').reset_index(drop=True)
        my = len(resultat)
        p_value_adj =[]

        for i in range(my):
            p_adj = resultat.loc[i, 'p_value'] * (my-i)
            p_value_adj.append(p_adj)
            
        p_value_adj = np.maximum.accumulate(p_value_adj)
        p_value_adj = np.minimum(p_value_adj, 1.0)

        resultat['p_value_adj_holm'] = p_value_adj

    return resultat.sort_values('Cramers_V', ascending=False)


# correlation de spearmann car les variables n'ont pas de distributions normales
def correlation_spearman(df:pd.DataFrame, num : list, target_col:str='Churn') -> pd.DataFrame:
    df_corr = df.copy()
    df_corr['Churn_binaire'] = df_corr[target_col].map({'Yes':1, 'No':0})

    # ajout de la cible a la correction
    cols_to_corr = num + ['Churn_binaire']
    corr_matrix = df_corr[cols_to_corr].corr(method='spearman')
    return corr_matrix
    
'''
    ANALYSE MULTIVARIEE ENTRE VARIABLES NUM
'''
def interaction_num(df:pd.DataFrame, target_col:str='Churn') -> pd.DataFrame:
    df_temp = df.copy()
    df_temp['Churn_bin'] = df_temp[target_col].map({'Yes':1, 'No':0})

    # tranche d'ancienneté
    bins = [0, 12, 36, 72]
    labels = ['Nouveau(0-12) mois', 'intermediaire(12-36)mois', 'Anciens(37+)mois']
    df_temp['Tenure_Group'] = pd.cut(df_temp['tenure'], bins=bins, labels = labels, include_lowest=True)

    resultat_intermediaire = []
    for group in labels:
        subset = df_temp[df_temp['Tenure_Group']==group]
        if len(subset) > 0:
            corr = subset['MonthlyCharges'].corr(subset['Churn_bin'], method='spearman')
            resultat_intermediaire.append({
                'Segment_anciennete' : group,
                'Nombre_clients' : len(subset),
                'Correlation_Prix_Churn' : corr.round(3)
            })
    return pd.DataFrame(resultat_intermediaire)




'''
    ETUDE MULTIVARIEE AVEC 
'''


""" Analyse les combinaisons de modalités associées au churn. """

def evaluate_all_combinations(df: pd.DataFrame, variables: list, max_depth: int = 3, min_clients: int = 50) -> pd.DataFrame:
    results = []
    
    for r in range(2, max_depth + 1):
        for combo in combinations(variables, r):
            crosstab = pd.crosstab(
                index=[df[col] for col in combo], 
                columns=df['Churn'], 
                normalize='index'
            ) * 100
            
            counts = df.groupby(list(combo)).size().to_frame(name='Count')
            comb_df = crosstab.join(counts)
            comb_df = comb_df[comb_df['Count'] >= min_clients]
            
            if len(comb_df) > 1:
                max_churn = comb_df['Yes'].max()
                min_churn = comb_df['Yes'].min()
                churn_range = max_churn - min_churn
                
                worst_profile = comb_df['Yes'].idxmax()
                best_profile = comb_df['Yes'].idxmin()
                
                def format_profile(profile):
                    if isinstance(profile, tuple):
                        return ' & '.join([f"{col}={val}" for col, val in zip(combo, profile)])
                    return f"{combo[0]}={profile}"
                
                results.append({
                    'Combinaison': ' + '.join(combo),
                    'Taille': r,
                    'Range_Churn (%)': round(churn_range, 1),
                    'Max_Churn (%)': round(max_churn, 1),
                    'Min_Churn (%)': round(min_churn, 1),
                    'Pire_Profil': format_profile(worst_profile),
                    'Meilleur_Profil': format_profile(best_profile),
                    'Nb_Clients_Max': int(comb_df.loc[worst_profile, 'Count'])
                })
                
    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by='Range_Churn (%)', ascending=False).reset_index(drop=True)
        
    return df_results



''' trier les 20 variables qui reviennent dans les combinaisons des variables qui produisent des Churn'''
def get_feature_importance_from_combinations(df_combinations: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Compte la fréquence d'apparition de chaque variable dans les N meilleures combinaisons.
    """
    # On prend le top N des combinaisons (les plus grands écarts)
    top_df = df_combinations.head(top_n)
    
    # On récupère toutes les variables utilisées dans ces combinaisons
    # (On sépare les noms par " + ")
    all_vars = []
    for comb in top_df['Combinaison']:
        # On sépare "Contract + InternetService" en ['Contract', 'InternetService']
        vars_in_comb = comb.split(' + ')
        all_vars.extend(vars_in_comb)
        
    # On compte les occurrences
    var_counts = pd.Series(all_vars).value_counts().reset_index()
    var_counts.columns = ['Variable', 'Fréquence dans le Top ' + str(top_n)]
    
    return var_counts








"""
Calcule le Variance Inflation Factor (VIF) pour détecter la multicolinéarité.
Un VIF > 5 indique une multicolinéarité modérée, > 10 une multicolinéarité sévère.
"""
def calculate_vif(df: pd.DataFrame, numeric_cols: list) -> pd.DataFrame:

    # On sélectionne uniquement les variables numériques
    X = df[numeric_cols].dropna()
    
    # On ajoute une constante (obligatoire pour le calcul du VIF)
    X_const = add_constant(X)
    
    # Calcul du VIF pour chaque variable
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
    
    # On retire la ligne de la constante (qui n'a pas d'intérêt métier) et on arrondit
    vif_data = vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)
    vif_data['VIF'] = vif_data['VIF'].round(2)
    
    return vif_data


'''
    l'importance des variables (Feature Importance) via un Random Forest.
    Evite le Data Leakage : preprocessing pipeline pour realiser de la Importance Variable
'''

def calculate_feature_importance(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> pd.DataFrame:

    # 1. On applique le preprocessing UNIQUEMENT sur le jeu d'entraînement : fit_transform garantit qu'il n'apprend que sur le train
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # 2. On récupère les noms des colonnes après transformation (One-Hot Encoding)
    feature_names = preprocessor.get_feature_names_out()
    
    # 3. Entraînement du Random Forest sur les données transformées
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_processed, y_train)
    
    # 4. Création du tableau des importances
    importances = pd.DataFrame({
        'Variable': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Nettoyage des noms (ex: "num__tenure" -> "tenure")
    importances['Variable'] = importances['Variable'].str.split('__').str[-1]
    
    # Arrondir pour la lisibilité
    importances['Importance'] = importances['Importance'].round(4)
    
    return importances






"""
    Calcule l'importance des variables via un Random Forest.
    RÈGLE D'OR : Prend uniquement X_train pour éviter le Data Leakage.
"""
def calculate_feature_importance(X_train: pd.DataFrame, y_train: pd.Series, preprocessor) -> pd.DataFrame:

    # Fit et Transform sur le Train UNIQUEMENT
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Récupération des noms de colonnes générés
    feature_names = preprocessor.get_feature_names_out()
    
    # Entraînement du modèle
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train_processed, y_train)
    
    # Création du DataFrame
    df_importance = pd.DataFrame({
        'Variable_Complete': feature_names,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
    
    # Nettoyage des noms (on enlève "num__" et "cat__")
    df_importance['Variable'] = df_importance['Variable_Complete'].str.replace('num__', '', regex=False).str.replace('cat__', '', regex=False)
    
    return df_importance[['Variable', 'Importance']]





"""
    Cherchons le seuil de probabilité qui maximise le F1-Score.
"""
def find_optimal_threshold(y_true, y_proba) -> dict:

    # 1. Calcul des courbes de précision et de rappel
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # 2. Calcul du F1-score pour chaque seuil
    # (Ajout d'une petite constante pour éviter la division par zéro)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    
    # 3. Trouver l'index du meilleur F1-Score
    # On ignore le dernier élément car 'thresholds' est plus court d'un élément
    best_idx = np.argmax(f1_scores[:-1]) 
    
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    return {
        'optimal_threshold': round(best_threshold, 3),
        'max_f1_score': round(best_f1, 3)
    }



"""
    Tracer de la courbe Précision-Rappel.
"""
def get_errors_by_threshold(y_true, y_proba, thresholds=np.linspace(0.1, 0.9, 81)):

    results = []
    for t in thresholds:
        # On applique le seuil
        y_pred = (y_proba >= t).astype(int)
        
        # On calcule la matrice de confusion
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        results.append({
            'Threshold': t,
            'FN': fn,
            'FP': fp,
            'Total_Erreurs': fn + fp
        })
        
    return pd.DataFrame(results)



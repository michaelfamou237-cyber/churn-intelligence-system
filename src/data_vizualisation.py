import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import shap




# distribution de la variable cible : Churn
def plot_target_distribution(df:pd.DataFrame) -> None:
    plt.figure(figsize=(8, 12))

    #verifiez le desequilibre de nos données
    sns.countplot(data=df, x='Churn', palette='Set2')
    plt.title("Repartition des clients : fideles VS Churn")
    plt.ylabel("Nombre de clients")
    plt.show()


# distribution des variables numeriques 
def graphiques_numeriques(df:pd.DataFrame) -> None:       
    fig, axes = plt.subplots(1,3, figsize=(18,10))
    sns.histplot(df['tenure'], bins=30, kde=True, ax=axes[0], color='salmon')
    axes[0].set_title("Distribution de Tenure")
    axes[0].set_xlabel("Mois")
    
    # distribution de la varianle : MonthlyCharges
    sns.histplot(df['MonthlyCharges'], bins=30, kde=True, ax=axes[1])
    axes[1].set_title("Distribution des factures mensuelles")
    axes[1].set_xlabel("Montant en euros")

    # distribution de la varianle : TotalCharges
    sns.histplot(df['TotalCharges'], bins=30, kde=True, ax=axes[2])
    axes[2].set_title("Distribution des factures totales")
    axes[2].set_xlabel("Montant cumulés en euros")

    plt.tight_layout()
    plt.show()


'''
    ETUDE BIVARIE DES VARIABLES NUMERIQUES ET DU CHURN
'''
# affihage boxplot comparant la distribution d'une variable numerique selon le Churn
def plot_bivarie_numerical(df:pd.DataFrame, variable:str) -> None:
    plt.figure(figsize=(8,4))
    sns.boxplot(data=df, x='Churn', y=variable, palette='Set1')
    plt.title(f"Distribution de {variable} selon le Churn")
    plt.show()

   
def plot_correlation(corr_matrix:pd.DataFrame) -> None:
    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Matrice de correlation de Spearman')
    plt.tight_layout()
    plt.show()
    
# evolution du taux de churn en fonction de en fonction des mensualités avec une courbe par tranche d'ancienneté
def plot_interaction(df:pd.DataFrame, var: str, target_col:str='Churn') -> None:

    df_temp = df.copy()
    
    # tranche d'ancienneté
    bins = [0, 12, 36, 72]
    labels = ['Nouveau(0-12) mois', 'intermediaire(12-36)mois', 'Anciens(37+)mois']
    df_temp['Tenure_Group'] = pd.cut(df_temp['tenure'], bins=bins, include_lowest=True)

    # on cree des tranches des prix
    df_temp['Price_Bin'] = pd.qcut(df[var], q=5, duplicates='drop')

    # calcul du taux de churn par groupe et par tranche de prix
    churn_rate = df_temp.groupby(['Tenure_Group', 'Price_Bin'], observed=True)[target_col].apply(lambda x:(x=='Yes').mean()*100).reset_index()
    churn_rate['Price_Median'] = churn_rate['Price_Bin'].apply(lambda x: x.mid).astype(float)

    plt.figure(figsize=(10,6))
    sns.lineplot(data=churn_rate, x='Price_Median', y=target_col, hue='Tenure_Group', marker='o')
    plt.title('Taux de churn(%) selon le prix mensuel et ancienneté')
    plt.xlabel(f"{var} Mediane en euros")
    plt.ylabel("Taux de Churn(%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


'''
    ETUDE BIVARIE DES VARIABLES NUMERIQUES ET DU CHURN
'''
def plot_categorical_multivarie(results_dict:dict)-> None:
    num_vars = len(results_dict)
    fig, axes = plt.subplots(1, num_vars, figsize=(5*num_vars, 5))
    if num_vars ==1:
        axes = [axes]

    for ax, (var_name, series_data) in zip(axes, results_dict.items()):
        sns.barplot(x=series_data.index, y=series_data.values, ax=ax, palette='Set2')

        ax.set_title(f"Taux de Churn par {var_name}")
        ax.set_ylabel("Taux de Churn(%)")
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=45)

        # affichage au dessus des graphes
        for i, v in enumerate(series_data.values):
            ax.text(i, v+1, f"{v:.1f}%", ha='center', va='bottom', fontsize=9)
        
    plt.tight_layout()
    plt.show()


''' 
Etude bivariée du Churn avec les variables categorielles une a une 
'''

def plot_churn_by_category(df, categorical_columns, target="Churn"):
    
    n = len(categorical_columns)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, 5 * n))

    if n == 1:
        axes = [axes]

    for ax, column in zip(axes, categorical_columns):
        churn_rate = (
            df.groupby(column, observed=True)[target]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=False))

        sns.barplot(
            x=churn_rate.values,
            y=churn_rate.index.astype(str),
            ax=ax )

        ax.set_title(f"Taux de churn par modalité : {column}")
        ax.set_xlabel("Taux de churn (%)")
        ax.set_ylabel(column)
        ax.set_xlim(0, 100)

        for i, value in enumerate(churn_rate.values):
            ax.text(value + 1, i, f"{value:.1f}%", va="center")

    plt.tight_layout()
    plt.show()



"""
    Affiche un barplot horizontal des variables les plus importantes.
"""
def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20) -> None:

    plt.figure(figsize=(10, 6))
    
    # On prend le Top N
    df_plot = importance_df.head(top_n)
    
    sns.barplot(
        data=df_plot, 
        x='Importance', 
        y='Variable', 
        hue='Variable',
        palette='viridis',
        legend=False  
    )
    
    plt.title(f"Top {top_n} des variables les plus importantes (Random Forest)")
    plt.xlabel("Importance relative")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.show()



    
"""
    Trace la courbe Précision-Rappel et met en évidence le seuil optimal: FN et FP en fonction du seuil de décision.
"""
def plot_error_curve(df_errors, chosen_threshold=0.562):
    
    plt.figure(figsize=(12, 6))
    
    # Tracer les courbes FN et FP
    sns.lineplot(data=df_errors, x='Threshold', y='FN', label='Faux Négatifs (FN) - Clients perdus', color='red', linewidth=2)
    sns.lineplot(data=df_errors, x='Threshold', y='FP', label='Faux Positifs (FP) - Appels inutiles', color='orange', linewidth=2)
    
    # Tracer la ligne du total des erreurs (pointillés)
    sns.lineplot(data=df_errors, x='Threshold', y='Total_Erreurs', 
                 label='Total des erreurs (FN+FP)', color='blue', linestyle='--', alpha=0.5)
    
    # Ajouter la ligne verticale du seuil choisi
    plt.axvline(x=chosen_threshold, color='black', linestyle=':', label=f'Seuil optimal ({chosen_threshold})')
    
    # Trouver les valeurs exactes à ce seuil pour l'annotation
    closest_row = df_errors.iloc[(df_errors['Threshold'] - chosen_threshold).abs().argmin()]
    fn_at_threshold = closest_row['FN']
    fp_at_threshold = closest_row['FP']
    
    plt.scatter(chosen_threshold, fn_at_threshold, color='red', s=100, zorder=5)
    plt.scatter(chosen_threshold, fp_at_threshold, color='orange', s=100, zorder=5)
    
    plt.text(chosen_threshold + 0.01, fn_at_threshold, f"FN: {fn_at_threshold}", color='red', fontweight='bold')
    plt.text(chosen_threshold + 0.01, fp_at_threshold, f"FP: {fp_at_threshold}", color='orange', fontweight='bold')

    # Affichage
    plt.title('Compromis FN vs FP selon le Seuil de Décision', fontsize=14)
    plt.xlabel('Seuil de probabilité (Threshold)', fontsize=12)
    plt.ylabel("Nombre d'erreurs", fontsize=12)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()





"""
    Affiche le graphique global (Summary Plot).
"""
def plot_shap_summary(shap_values, X_transformed_df):

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed_df, show=False)
    plt.title("Impact global des variables sur le Churn (SHAP)", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_shap_waterfall(shap_values, client_index: int):
    """
    Affiche le graphique individuel (Waterfall Plot).
    """
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[client_index], max_display=10, show=False)
    plt.title(f"Décomposition du score SHAP pour le client n°{client_index}", fontsize=14)
    plt.tight_layout()
    plt.show()



"""
    Affiche la courbe d'évolution du F1-Score en fonction des hyperparamètres testés.
"""
def plot_gridsearch_results(grid_search, param_name: str = 'model__C'):
    
    #  Extraction des résultats dans un DataFrame
    results = pd.DataFrame(grid_search.cv_results_)
    
    # On récupère uniquement la colonne du paramètre qui nous intéresse et le score
    results['Param_C'] = results['param_' + param_name]
    
    # Création du graphique
    plt.figure(figsize=(10, 6))
    
    # Tracé de la courbe
    sns.lineplot(data=results, x='Param_C', y='mean_test_score', marker='o', color='blue', linewidth=2)
    
    # Ajout de la zone d'incertitude (écart-type des 5 folds)
    plt.fill_between(results['Param_C'], 
                     results['mean_test_score'] - results['std_test_score'],
                     results['mean_test_score'] + results['std_test_score'], 
                     alpha=0.2, color='blue')
    
    # Mise en évidence du meilleur paramètre
    best_param = grid_search.best_params_[param_name]
    best_score = grid_search.best_score_
    
    plt.scatter(best_param, best_score, color='red', s=100, zorder=5, label=f'Meilleur C = {best_param}\n(F1 = {best_score:.3f})')
    
   
    plt.xscale('log') # Échelle logarithmique car on teste 0.01, 0.1, 1, 10, 100
    plt.title("Évolution du F1-Score selon la force de régularisation (C)", fontsize=14)
    plt.xlabel("Paramètre C (Régularisation)", fontsize=12)
    plt.ylabel("F1-Score (Validation Croisée)", fontsize=12)
    plt.xticks(results['Param_C'], labels=[str(x) for x in results['Param_C']]) # Afficher les vraies valeurs
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
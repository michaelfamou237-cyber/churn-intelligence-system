import shap
import pandas as pd


"""
    Calcule les valeurs SHAP. Retourne un tuple (shap_values, X_transformed_df).
"""
def compute_shap_values(pipeline, X_data):

    # 1. Extraire le préprocesseur et le modèle
    preprocessor = pipeline.named_steps['preprocessor']
    model = pipeline.named_steps['model']
    
    # 2. Transformer les données brutes
    X_transformed = preprocessor.transform(X_data)
    
    # 3. Récupérer et nettoyer les noms des colonnes
    feature_names = preprocessor.get_feature_names_out()
    clean_names = [name.split('__')[-1] for name in feature_names]
    
    # 4. Créer un DataFrame 
    X_transformed_df = pd.DataFrame(X_transformed, columns=clean_names)
    
    # 5. Créer l'explainer (LinearExplainer pour la Régression Logistique)
    explainer = shap.LinearExplainer(model, X_transformed_df)
    
    # 6. Calculer les valeurs SHAP
    shap_values = explainer(X_transformed_df)
    
    return shap_values, X_transformed_df
### Churn Intelligence System
Système d'aide à la décision pour la prévention du départ des clients (Churn).

Ce projet est une démonstration complète d'un pipeline de Data Science de bout en bout, allant de la compréhension métier à la mise en production d'une application interactive (Streamlit).

Il a été conçu avec une approche rigoriste notamment par la séparation des responsabilités (code modulaire), la prévention du Data Leakage, la validation statistique stricte, et la priorisation des métriques métier sur les métriques purement mathématiques.

<img width="1280" height="720" alt="1_Accueil_page1" src="https://github.com/user-attachments/assets/5c5b4cfb-d454-48e5-b4bf-8155d43c78af" />



### 1. Contexte & Enjeux Métier
Dans le secteur des télécommunications (et plus largement dans les services par abonnement), le départ d'un client (Churn) représente une perte de revenus directe.

Le problème : Un client qui part coûte nettement plus cher qu'un client qui reste (coût d'acquisition vs coût de rétention).

L'objectif : Construire un modèle capable d'identifier les clients à risque avant qu'ils ne résilient, afin que les conseillers puissent déclencher des actions de rétention ciblées.

L'arbitrage métier (Coût des erreurs) : Rater un client qui va partir (Faux Négatif - FN) coûte beaucoup plus cher que d'appeler un client qui allait rester (Faux Positif - FP). La stratégie du modèle est donc de maximiser le Recall (la capacité à détecter les churners), au détriment d'une Précision plus faible.

2. Données & Analyse Exploratoire (EDA)
Dataset : IBM Telco Customer Churn (7 043 clients, 21 variables).

Audit & Nettoyage : Détection et correction de la variable TotalCharges (initialement stockée en texte à cause d'espaces vides, convertie en numérique).

Analyse Univariée & Bivariée :

Utilisation de tests statistiques rigoureux : Test de Mann-Whitney U (pour les numériques), Test du Chi-2 et V de Cramér (pour les catégorielles).

Application de la correction de Holm pour contrôler le risque d'erreur de type I (Faux Positifs) lors des tests multiples.

Analyse de la taille d'effet via la Probabilité de Supériorité (PS) et la corrélation bisérielle de rang pour quantifier la force des relations.

Insights clés : Le type de contrat (Month-to-month), l'ancienneté (tenure) et la présence de Fibre Optique sont les principaux marqueurs de risque. Les variables démographiques comme le genre (gender) n'ont aucun impact significatif.

3. Feature Selection & Prévention du Data Leakage
Multicolinéarité : Calcul du VIF (Variance Inflation Factor). La variable TotalCharges a été exclue (VIF > 9.5) car mécaniquement redondante avec tenure et MonthlyCharges.

Validation par le modèle : Utilisation de la Feature Importance (Random Forest) pour confirmer l'élimination des variables non prédictives (gender, PhoneService, MultipleLines).

Rigueur (Zéro Data Leakage) : Le préprocesseur (StandardScaler, OneHotEncoder) est intégré dans un Pipeline Scikit-Learn. Il est fit uniquement sur le jeu d'entraînement (Train) et appliqué (transform) sur le jeu de test. Aucune information du Test n'a fuité dans l'entraînement.

4. Modélisation & Benchmarking
14 algorithmes ont été testés (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, CatBoost, etc.).
Le benchmark a révélé que la Régression Logistique offrait le meilleur compromis pour notre cas d'usage.

Optimisation (GridSearchCV)
Un GridSearchCV (Validation Croisée 5-fold) a été effectué sur le Train pour trouver le meilleur hyperparamètre de régularisation.

Résultat : C = 0.01 (Forte régularisation). Le modèle est ainsi plus stable et évite le sur-apprentissage.

Le Choix du Modèle Final (Logistic Regression vs SVM)
Bien que le SVM ait un F1-Score légèrement supérieur à seuil égal, la Régression Logistique a été choisie pour des raisons de mise en production (MLOps) :

Priorité métier (Recall) : La LR optimisée offre un Recall de 0.861 (seulement 52 FN), surpassant le SVM.

Interprétabilité : La LR fournit des coefficients directs, indispensables pour que les conseillers expliquent le score au client. Le SVM est une "boîte noire".

Contraintes de déploiement : La LR a une latence minimale, une taille de modèle négligeable et ne subit pas de Data Drift soudain. Elle est beaucoup plus facile à maintenir dans le temps.

5. Le Seuil Métier (Business Threshold)
Le seuil mathématique optimal (maximisant le F1-Score) était de 0.562. Cependant, il générait 95 Faux Négatifs (clients perdus).
Pour aligner le modèle avec les objectifs de l'entreprise, le seuil a été abaissé à 0.40.

Modèle	Seuil	TP	TN	FP	FN	Recall	Precision	F1-Score
LR (Benchmark)	0.50	300	737	298	74	0.802	0.502	0.617
LR (Final)	0.40	322	654	381	52	0.861	0.458	0.598
Arbitrage assumé : Nous acceptons d'augmenter les Faux Positifs (de 298 à 381) pour faire chuter les Faux Négatifs (de 74 à 52). L'entreprise préfère faire 83 appels de vérification "inutiles" plutôt que de perdre 22 clients supplémentaires.

6. Interprétabilité (SHAP)
Le modèle est totalement "boîte blanche" grâce à l'intégration de SHAP (SHapley Additive exPlanations).

Niveau Global (Beeswarm) : Confirme que tenure, Contract_Two year (protecteur) et MonthlyCharges sont les moteurs principaux du modèle.

Niveau Individuel (Waterfall) : Intégré dans l'application Streamlit. Pour chaque client, l'application affiche exactement quelles variables ont poussé le score vers le haut (risque) ou vers le bas (protection), permettant au conseiller d'adapter son discours.


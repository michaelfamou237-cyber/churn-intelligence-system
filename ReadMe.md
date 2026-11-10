### Churn Intelligence System
Système d'aide à la décision pour la prévention du départ des clients (Churn).

Ce projet est une démonstration complète d'un pipeline de Data Science de bout en bout, allant de la compréhension métier à la mise en production d'une application interactive (Streamlit).

Il a été conçu avec une approche rigoriste notamment par la séparation des responsabilités (code modulaire), la prévention du Data Leakage, la validation statistique stricte, et la priorisation des métriques métier sur les métriques purement mathématiques.

<img width="1280" height="720" alt="1_Accueil_page1" src="https://github.com/user-attachments/assets/5c5b4cfb-d454-48e5-b4bf-8155d43c78af" />

<img width="1280" height="720" alt="2_Contexte_KPI_1" src="https://github.com/user-attachments/assets/11bf3b97-5ff6-4e2e-bfed-47e0430467e4" />
<img width="1280" height="720" alt="3_ExplorationDonnees_4" src="https://github.com/user-attachments/assets/73e03685-aa7b-4ade-8538-3337461256df" />
<img width="1280" height="720" alt="3_ExplorationDonnees_2" src="https://github.com/user-attachments/assets/6d63dcf6-7bf7-4a57-9c67-b83720e95437" />
<img width="1280" height="720" alt="3_EcplorationDonnees_1" src="https://github.com/user-attachments/assets/9dfaa021-ceba-4870-97ca-d53fb850d44d" />
<img width="1280" height="720" alt="3_ExplorationDonnees_3" src="https://github.com/user-attachments/assets/1b8a3668-d28b-4018-a32a-b1d67ad1cd87" />
<img width="1280" height="720" alt="2_Contexte_KPI_2" src="https://github.com/user-attachments/assets/2f9bad25-1818-4a1c-b05b-330c3ef439b5" />
<img width="1280" height="720" alt="3_ExplorationDonnees_4" src="https://github.com/user-attachments/assets/466fe07d-d477-4fdc-ade8-7a9565d201ab" />



### 1. Contexte & Enjeux Métier
Dans le secteur des télécommunications (et plus largement dans les services par abonnement), le départ d'un client (Churn) représente une perte de revenus directe.

Le problème est du au fait qu'un client qui part coûte nettement plus cher qu'un client qui reste (coût d'acquisition vs coût de rétention).

L'objectif est donc de construire un modèle capable d'identifier les clients à risque avant qu'ils ne résilient, afin que les conseillers puissent déclencher des actions de rétention ciblées.

L'arbitrage métier (Coût des erreurs) est le suivant:
 - un client qui va partir (Faux Négatif - FN) coûte beaucoup plus cher que d'appeler un client qui allait rester (Faux Positif - FP).
 - La stratégie du modèle est donc de maximiser le Recall (la capacité à détecter les churners), au détriment d'une Précision plus faible.

### 2. Données & Analyse Exploratoire (EDA)
La dimension de notre dataset :nommé IBM Telco Customer Churn est de 7 043 clients, et  21 variables.
Nous avons entrepris les actions suivantes :
- Audit & Nettoyage : au travers de la détection et de la correction de la variable TotalCharges (initialement stockée en texte à cause d'espaces vides, convertie en numérique).
- Analyse Univariée & Bivariée notamment par utilisation de tests statistiques rigoureux : Test de Mann-Whitney U (pour les numériques), Test du Chi-2 et V de Cramér (pour les catégorielles).
- Application de la correction de Holm pour contrôler le risque d'erreur de type I (Faux Positifs) lors des tests multiples.
- Analyse de la taille d'effet via la Probabilité de Supériorité (PS) et la corrélation bisérielle de rang pour quantifier la force des relations.

Un Insight clés de l'analyse numerique est que le type de contrat (Month-to-month), l'ancienneté (tenure) et la présence de Fibre Optique sont les principaux marqueurs de risque. Les variables démographiques comme le genre (gender) n'ont aucun impact significatif.

### 3. Feature Selection & Prévention du Data Leakage
Nous avons procédé par traitement des opérations de : 
- Multicolinéarité : Calcul du VIF (Variance Inflation Factor). La variable TotalCharges a été exclue (VIF > 9.5) car mécaniquement redondante avec tenure et MonthlyCharges.
- Validation par le modèle : Utilisation de la Feature Importance (Random Forest) pour confirmer l'élimination des variables non prédictives (gender, PhoneService, MultipleLines).
- Rigueur (Zéro Data Leakage) : Le préprocesseur (StandardScaler, OneHotEncoder) est intégré dans un Pipeline Scikit-Learn. Il est fit uniquement sur le jeu d'entraînement (Train) et appliqué (transform) sur le jeu de test. Aucune information du Test n'a fuité dans l'entraînement.


<img width="1280" height="720" alt="vif_num" src="https://github.com/user-attachments/assets/f6e1601e-acc2-4f5e-a9fa-36031e22a959" />
<img width="1280" height="720" alt="Importance_variables" src="https://github.com/user-attachments/assets/00bb35da-a0b2-4abf-b798-6bfe20b3d181" />



### 4. Modélisation & Benchmarking
14 algorithmes ont été testés (Logistic Regression, SVM, Random Forest, XGBoost, LightGBM, CatBoost, etc.).
Le benchmark a révélé que la Régression Logistique offrait le meilleur compromis pour notre cas d'usage.


Un GridSearchCV (Validation Croisée 5-fold) a été effectué sur le Train pour trouver le meilleur hyperparamètre de régularisation C.
Nous avons obtenu un résultat : C = 0.01 (Forte régularisation). Le modèle est ainsi plus stable et évite le sur-apprentissage.

#### Remarques: 
Bien que le SVM ait un F1-Score légèrement supérieur à seuil égal, la Régression Logistique a été choisie pour des raisons de mise en production (MLOps) :
Priorité métier (Recall) : La LR optimisée offre un Recall de 0.861 (seulement 52 FN), surpassant le SVM.
La LR fournit des coefficients directs, indispensables pour que les conseillers expliquent le score au client. Le SVM est une "boîte noire".
Les contraintes de déploiement notamment pour  La LR a une latence minimale, une taille de modèle négligeable et ne subit pas de Data Drift soudain. Elle est beaucoup plus facile à maintenir dans le temps.

### 5. Le Seuil Métier (Business Threshold)
Le seuil mathématique optimal (maximisant le F1-Score) était de 0.562. Cependant, il générait 95 Faux Négatifs (clients perdus).
Pour aligner le modèle avec les objectifs de l'entreprise, le seuil a été abaissé à 0.40.
<img width="1280" height="720" alt="seuil_Metier" src="https://github.com/user-attachments/assets/0e55277d-1b71-483f-944d-459fb925101e" />

<img width="1280" height="720" alt="seuil_compromis_1" src="https://github.com/user-attachments/assets/b0c71b4d-5065-443b-b9f9-730008d38435" />

<img width="1280" height="720" alt="seuil_compromis_equilibre" src="https://github.com/user-attachments/assets/8e5ef952-7d79-4187-aae6-7802802c2d98" />

# Modèle	Seuil	TP	TN	FP	FN	Recall	Precision	F1-Score
LR (Benchmark)	0.50	300	737	298	74	0.802	0.502	0.617
LR (Final)	0.40	322	654	381	52	0.861	0.458	0.598
Nous acceptons d'augmenter les Faux Positifs (de 298 à 381) pour faire chuter les Faux Négatifs (de 74 à 52). L'entreprise préfère faire 83 appels de vérification "inutiles" plutôt que de perdre 22 clients supplémentaires.



### 6. Interprétabilité (SHAP)
Le modèle est totalement "boîte blanche" grâce à l'intégration de SHAP (SHapley Additive exPlanations).
Le n<img width="1280" height="720" alt="info3_remplir_prediction" src="https://github.com/user-attachments/assets/b2865343-027b-4f02-b8fc-8b54a3eb3630" />

le niveau Global (Beeswarm) confirme que tenure, Contract_Two year (protecteur) et MonthlyCharges sont les moteurs principaux du modèle.
le niveau Individuel (Waterfall) : Intégré dans l'application Streamlit. Pour chaque client, l'application affiche exactement quelles variables ont poussé le score vers le haut (risque) ou vers le bas (protection), permettant au conseiller d'adapter son discours.


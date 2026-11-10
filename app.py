
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import plotly.express as px

from pathlib import Path

import os
from pathlib import Path

# ============================================================
# 1. CONFIGURATION ET CHEMINS DU PROJET (Relatifs)
# ============================================================
st.set_page_config(
    page_title="Churn Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ROOT pointe vers le dossier principal (09_Churn)
# __file__ est app/app.py, donc parent.parent remonte à 09_Churn
# ROOT pointe vers le dossier principal
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "model" / "logistic_regression_model.pkl"
CONFIG_PATH = ROOT / "model" / "business_config.json"


# ============================================================
# 2. STYLE DE L'APPLICATION
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #f5f7fb;
    color: #172033;
}

[data-testid="stSidebar"] {
    background-color: #111c31;
}

[data-testid="stSidebar"] * {
    color: #e7edf8;
}

[data-testid="stSidebar"] hr {
    border-color: #33415c;
}

h1, h2, h3 {
    font-family: 'Manrope', sans-serif;
    color: #172033;
}

h1 {
    font-weight: 800;
    letter-spacing: -1px;
}

.eyebrow {
    color: #5470a2;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 2px;
}

.brand-mark {
    background: #56d6b0;
    color: #102139 !important;
    font-weight: 900;
    border-radius: 12px;
    padding: 10px 12px;
    font-size: 18px;
}

.brand-title {
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 1px;
}

.brand-subtitle {
    font-size: 0.58rem;
    color: #aab8d0 !important;
    letter-spacing: 1px;
}

.hero {
    background: linear-gradient(115deg, #14233e, #243d65);
    border-radius: 20px;
    padding: 35px;
    color: white;
    margin: 22px 0;
}

.hero-kicker {
    color: #70e0c0;
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-heading {
    color: white;
    font: 800 2.5rem 'Manrope';
    line-height: 1.15;
    margin: 15px 0;
}

.hero-copy {
    color: #d4deee;
    font-size: 1rem;
    line-height: 1.6;
    max-width: 700px;
}

.feature-card, .metric-card {
    background: white;
    border: 1px solid #e6ebf3;
    border-radius: 15px;
    padding: 22px;
    box-shadow: 0 5px 18px rgba(24,43,77,0.04);
}

.feature-card {
    min-height: 155px;
}

.feature-number {
    color: #18a982;
    font-weight: 800;
    font-size: 0.8rem;
}

.feature-card h3 {
    font-size: 1.1rem;
    margin-top: 12px;
}

.feature-card p {
    color: #68758b;
    font-size: 0.9rem;
    line-height: 1.5;
}

.metric-label {
    color: #68758b;
    font-size: 0.78rem;
    font-weight: 600;
}

.metric-value {
    font: 800 1.8rem 'Manrope';
    color: #182844;
    margin-top: 8px;
}

.metric-note {
    color: #8290a5;
    font-size: 0.72rem;
    margin-top: 4px;
}

.section-note {
    background: #edf3fb;
    border-left: 4px solid #4c79bb;
    padding: 14px 17px;
    border-radius: 7px;
    margin: 15px 0 22px;
    color: #42536d;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e6ebf3;
    padding: 15px;
    border-radius: 13px;
}

.stButton > button {
    border-radius: 9px;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. FONCTIONS DE CHARGEMENT
# ============================================================

@st.cache_data
def load_data():

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    # Conversion de TotalCharges en numérique
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce"
        )

    return df


@st.cache_resource
def load_artifacts():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration introuvable : {CONFIG_PATH}"
        )

    pipeline = joblib.load(MODEL_PATH)

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    threshold = float(config["business_threshold"])

    return pipeline, threshold


# ============================================================
# 4. FONCTIONS D'AFFICHAGE
# ============================================================

def display_metric(label, value, note=""):

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def page_header(number, section, title, description):

    st.markdown(
        f'<div class="eyebrow">{number} / {section}</div>',
        unsafe_allow_html=True
    )

    st.title(title)
    st.write(description)


# ============================================================
# 5. CHARGEMENT DES DONNÉES ET DU MODÈLE
# ============================================================

try:
    df = load_data()
    pipeline, business_threshold = load_artifacts()

except Exception as error:
    st.error(f"Erreur de chargement : {error}")
    st.stop()


# ============================================================
# 6. SIDEBAR ET NAVIGATION
# ============================================================

st.sidebar.markdown("""
<div class="brand">
    <div class="brand-mark">CI</div>
    <div>
        <div class="brand-title">CHURN INTELLIGENCE</div>
        <div class="brand-subtitle">CUSTOMER RETENTION PLATFORM</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.caption("NAVIGATION")

page = st.sidebar.radio(
    "Sélectionner une page",
    [
        "Vue d'ensemble",
        "Contexte métier & KPI",
        "Exploration des données",
        "Simulateur de prédiction"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

st.sidebar.caption("CHURN INTELLIGENCE SYSTEM")
st.sidebar.caption("Customer Analytics & Machine Learning")


# ============================================================
# 7. PAGE D'ACCUEIL
# ============================================================

if page == "Vue d'ensemble":

    st.markdown(
        '<div class="eyebrow">CUSTOMER ANALYTICS / MACHINE LEARNING</div>',
        unsafe_allow_html=True
    )

    st.title("Churn Intelligence System")

    st.write(
        "Une plateforme d'aide à la décision pour identifier les clients "
        "susceptibles de résilier, comprendre les facteurs associés au "
        "départ et orienter les actions de rétention."
    )

    st.markdown("""
    <div class="hero">
        <div class="hero-kicker">FROM DATA TO RETENTION ACTIONS</div>
        <div class="hero-heading">
            Anticiper le départ.<br>
            Agir au bon moment.
        </div>
        <div class="hero-copy">
            Une démarche structurée : comprendre le phénomène,
            explorer les données et simuler le risque individuel
            à l'aide d'un modèle de classification.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">01</div>
            <h3>Comprendre</h3>
            <p>
                Définir le problème métier, les enjeux de rétention
                et les indicateurs de pilotage.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">02</div>
            <h3>Explorer</h3>
            <p>
                Analyser la structure du portefeuille et les associations
                entre caractéristiques et churn.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-number">03</div>
            <h3>Prédire</h3>
            <p>
                Estimer une probabilité individuelle et orienter
                les actions selon un seuil métier.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Parcours de démonstration")

    st.write(
        "Utilisez le menu latéral pour parcourir les trois étapes "
        "de la démarche analytique."
    )


# ============================================================
# 8. PAGE CONTEXTE MÉTIER ET KPI
# ============================================================

elif page == "Contexte métier & KPI":

    page_header(
        "01",
        "BUSINESS UNDERSTANDING",
        "Contexte métier & KPI",
        "Comprendre le phénomène de churn et mesurer son importance "
        "dans le portefeuille clients."
    )

    st.markdown("""
    <div class="section-note">
        <b>Problématique métier.</b>
        Quels clients présentent un risque de résiliation suffisamment
        élevé pour justifier une action de rétention, et comment
        prioriser ces actions ?
    </div>
    """, unsafe_allow_html=True)

    # Calcul des indicateurs
    churn = df["Churn"].astype(str).str.strip()

    total_clients = len(df)
    churned_clients = int(churn.eq("Yes").sum())
    retained_clients = int(churn.eq("No").sum())

    churn_rate = churn.eq("Yes").mean() * 100

    avg_monthly = df["MonthlyCharges"].mean()
    avg_tenure = df["tenure"].mean()

    st.markdown("### Indicateurs du portefeuille")

    a, b, c, d = st.columns(4)

    with a:
        display_metric(
            "Clients analysés",
            f"{total_clients:,}".replace(",", " "),
            "Taille du portefeuille"
        )

    with b:
        display_metric(
            "Taux de churn",
            f"{churn_rate:.1f} %",
            "Clients ayant résilié"
        )

    with c:
        display_metric(
            "Facture mensuelle moyenne",
            f"{avg_monthly:.2f}",
            "Moyenne du portefeuille"
        )

    with d:
        display_metric(
            "Ancienneté moyenne",
            f"{avg_tenure:.1f} mois",
            "Durée de relation client"
        )

    st.markdown("---")

    # Répartition du portefeuille
    st.markdown("### Structure du portefeuille")

    left, right = st.columns(2)

    with left:

        distribution = pd.DataFrame({
            "Statut": ["Clients conservés", "Clients churnés"],
            "Effectif": [retained_clients, churned_clients]
        })

        fig = px.pie(
            distribution,
            names="Statut",
            values="Effectif",
            hole=0.6,
            color="Statut",
            color_discrete_map={
                "Clients conservés": "#43b99b",
                "Clients churnés": "#e47770"
            }
        )

        fig.update_layout(
            margin=dict(t=20, b=20),
            legend_title=""
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        st.markdown("**Lecture métier**")

        st.write(
            f"- {retained_clients:,} clients sont conservés."
        )

        st.write(
            f"- {churned_clients:,} clients ont résilié."
        )

        st.write(
            f"- Le taux de churn observé est de {churn_rate:.1f} %."
        )

        st.write(
            "Ces indicateurs permettent de mesurer l'ampleur du phénomène "
            "et de suivre l'évolution du portefeuille."
        )

    st.markdown("---")

    st.markdown("### Traduction des KPI en décisions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Indicateurs de pilotage**")
        st.write("- Taux de churn : mesure l'ampleur du phénomène.")
        st.write("- Volume de churn : quantifie les pertes observées.")
        st.write("- Facture moyenne : apporte un contexte économique.")
        st.write("- Ancienneté : renseigne sur la durée de relation.")

    with col2:
        st.markdown("**Décisions possibles**")
        st.write("- Prioriser les clients à risque élevé.")
        st.write("- Adapter les offres aux profils clients.")
        st.write("- Définir des actions de rétention ciblées.")
        st.write("- Suivre les résultats des campagnes.")

    st.caption(
        "Les KPI décrivent le portefeuille observé. Ils ne mesurent "
        "pas directement la performance prédictive du modèle."
    )


# ============================================================
# 9. PAGE EXPLORATION DES DONNÉES
# ============================================================

elif page == "Exploration des données":

    page_header(
        "02",
        "EXPLORATORY DATA ANALYSIS",
        "Exploration des données",
        "Explorer la structure du portefeuille et identifier "
        "les variables associées au churn."
    )

    # Aperçu du dataset
    with st.expander("Aperçu du jeu de données"):

        st.write(
            f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes"
        )

        st.dataframe(df.head(15), use_container_width=True)

        missing = (
            df.isna()
            .sum()
            .rename("Valeurs manquantes")
            .to_frame()
        )

        st.markdown("**Valeurs manquantes**")

        st.dataframe(
            missing[missing["Valeurs manquantes"] == 0],
            use_container_width=True
        )


    # Normalisation de la cible
    data_eda = df.copy()
    data_eda["Churn"] = (
        data_eda["Churn"].astype(str).str.strip()
    )

    # Répartition de la cible
    st.markdown("### 1. Répartition de la variable cible")

    target_counts = (
        data_eda["Churn"]
        .value_counts()
        .rename_axis("Churn")
        .reset_index(name="Clients")
    )

    col1, col2 = st.columns([1, 1])

    with col1:

        fig = px.pie(
            target_counts,
            names="Churn",
            values="Clients",
            hole=0.6,
            color="Churn",
            color_discrete_map={
                "Yes": "#e47770",
                "No": "#43b99b"
            }
        )

        fig.update_layout(
            margin=dict(t=20, b=20),
            legend_title=""
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        rate = data_eda["Churn"].eq("Yes").mean() * 100

        st.metric(
            "Taux de churn observé",
            f"{rate:.1f}%"
        )

        st.write(
            "La distribution de la cible permet d'apprécier "
            "l'équilibre entre les clients conservés et les clients churnés."
        )

        st.write(
            "Cette information est importante pour choisir "
            "les métriques d'évaluation du classifieur."
        )

    # Analyse des variables numériques
    st.markdown("---")
    st.markdown("### 2. Analyse des variables numériques")

    numeric_variables = [
        col for col in [
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]
        if col in data_eda.columns
    ]

    if numeric_variables:

        selected_numeric = st.selectbox(
            "Choisir une variable numérique",
            numeric_variables
        )

        fig = px.histogram(
            data_eda,
            x=selected_numeric,
            color="Churn",
            marginal="box",
            barmode="overlay",
            opacity=0.7,
            color_discrete_map={
                "Yes": "#e47770",
                "No": "#3d83c6"
            }
        )

        fig.update_layout(
            legend_title="Churn",
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Analyse des variables catégorielles
    st.markdown("---")
    st.markdown("### 3. Taux de churn par catégorie")

    categorical_variables = [
        col for col in [
            "Contract",
            "InternetService",
            "PaymentMethod",
            "TechSupport",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "Partner",
            "PaperlessBilling"
        ]
        if col in data_eda.columns
    ]

    if categorical_variables:

        selected_category = st.selectbox(
            "Choisir une variable catégorielle",
            categorical_variables
        )

        category_rates = (
            data_eda.groupby(
                selected_category,
                dropna=False
            )["Churn"]
            .apply(
                lambda x: x.eq("Yes").mean() * 100
            )
            .reset_index(name="Taux de churn")
        )

        category_rates = category_rates.sort_values(
            "Taux de churn",
            ascending=False
        )

        fig = px.bar(
            category_rates,
            x=selected_category,
            y="Taux de churn",
            text_auto=".1f",
            color="Taux de churn",
            color_continuous_scale="Tealgrn"
        )

        fig.update_layout(
            yaxis_title="Taux de churn (%)",
            xaxis_title="",
            coloraxis_showscale=False,
            margin=dict(t=20, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            category_rates,
            use_container_width=True,
            hide_index=True
        )

    # Corrélations numériques
    st.markdown("---")
    st.markdown("### 4. Corrélations entre variables numériques")

    if len(numeric_variables) >= 2:

        correlation = data_eda[numeric_variables].corr(
            method="spearman"
        )

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Cette page présente une exploration descriptive. "
        "Les associations observées ne prouvent pas une causalité. "
        "L'intérêt prédictif doit être vérifié lors du benchmarking "
        "et de l'évaluation sur des données non vues."
    )


# ============================================================
# 10. PAGE SIMULATEUR DE PRÉDICTION
# ============================================================

elif page == "Simulateur de prédiction":

    page_header(
        "03",
        "CUSTOMER-LEVEL SCORING",
        "Simulateur de prédiction",
        "Évaluez le risque de résiliation d'un client et identifiez "
        "les facteurs qui influencent sa probabilité de départ."
    )

    # --------------------------------------------------------
    # 10.1 INITIALISATION DE L'ÉTAT DE LA SIMULATION
    # --------------------------------------------------------

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    # --------------------------------------------------------
    # 10.2 ORGANISATION : ZONE CENTRALE ET PANNEAU DROIT
    # --------------------------------------------------------

    main_col, control_col = st.columns([3, 1])

    with control_col:

        st.markdown("### ⚙️ Paramètres métiers")

        threshold = st.slider(
            "Seuil d'alerte",
            min_value=0.10,
            max_value=0.90,
            value=float(business_threshold),
            step=0.05,
            help="Un client est considéré à risque lorsque sa probabilité "
                 "de churn atteint ou dépasse ce seuil."
        )

        st.markdown("---")

        st.markdown("### Indicateurs de décision")

        st.metric(
            "Seuil actuel",
            f"{threshold:.0%}"
        )

        st.caption(
            "Le seuil modifie la classification métier du client. "
            "Il ne modifie pas la probabilité calculée par le modèle."
        )

        st.markdown("---")

        st.markdown("### Lecture du risque")

        st.write("**Risque faible**")
        st.progress(0.30)

        st.write("**Risque intermédiaire**")
        st.progress(0.60)

        st.write("**Risque élevé**")
        st.progress(0.90)

        st.caption(
            "Ces niveaux sont des repères visuels. "
            "La décision effective dépend du seuil sélectionné."
        )

    with main_col:

        # ----------------------------------------------------
        # 10.3 FORMULAIRE DE SAISIE DU CLIENT
        # ----------------------------------------------------

        st.markdown("### 1. Profil du client à évaluer")

        st.info(
            "Renseignez les caractéristiques du client, puis cliquez "
            "sur « Lancer la prédiction » pour obtenir son score de risque."
        )

        with st.form("customer_prediction_form"):

            st.markdown("#### Informations contractuelles")

            c1, c2, c3 = st.columns(3)

            with c1:
                tenure = st.slider(
                    "Ancienneté (mois)",
                    min_value=0,
                    max_value=72,
                    value=12
                )

            with c2:
                monthly_charges = st.slider(
                    "Facture mensuelle",
                    min_value=18.0,
                    max_value=120.0,
                    value=70.0,
                    step=0.01
                )

            with c3:
                contract = st.selectbox(
                    "Type de contrat",
                    ["Month-to-month", "One year", "Two year"]
                )

            st.markdown("---")
            st.markdown("#### Services souscrits")

            c1, c2 = st.columns(2)

            with c1:

                internet_service = st.selectbox(
                    "Service Internet",
                    ["DSL", "Fiber optic", "No"]
                )

                tech_support = st.selectbox(
                    "Support technique",
                    ["Yes", "No", "No internet service"]
                )

            with c2:

                online_security = st.selectbox(
                    "Sécurité en ligne",
                    ["Yes", "No", "No internet service"]
                )

                online_backup = st.selectbox(
                    "Sauvegarde en ligne",
                    ["Yes", "No", "No internet service"]
                )

            c1, c2 = st.columns(2)

            with c1:

                device_protection = st.selectbox(
                    "Protection appareil",
                    ["Yes", "No", "No internet service"]
                )

                payment_method = st.selectbox(
                    "Moyen de paiement",
                    [
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)"
                    ]
                )

            with c2:

                senior_citizen = st.selectbox(
                    "Senior (65+)",
                    [0, 1],
                    format_func=lambda x: "Oui" if x == 1 else "Non"
                )

                partner = st.selectbox(
                    "En couple",
                    ["Yes", "No"]
                )

                paperless_billing = st.selectbox(
                    "Facture dématérialisée",
                    ["Yes", "No"]
                )

            st.markdown("---")

            submitted = st.form_submit_button(
                " **Lancer la prédiction** ",
                type="primary",
                use_container_width=True
            )

        # ----------------------------------------------------
        # 10.4 EXÉCUTION DE LA PRÉDICTION AU CLIC
        # ----------------------------------------------------

        if submitted:

            input_data = pd.DataFrame({

                "tenure": [tenure],

                "MonthlyCharges": [monthly_charges],

                "Contract": [contract],

                "InternetService": [internet_service],

                "TechSupport": [tech_support],

                "OnlineSecurity": [online_security],

                "OnlineBackup": [online_backup],

                "DeviceProtection": [device_protection],

                "PaymentMethod": [payment_method],

                "SeniorCitizen": [senior_citizen],

                "Partner": [partner],

                "PaperlessBilling": [paperless_billing]

            })

            try:

                probability = float(
                    pipeline.predict_proba(input_data)[0, 1]
                )

                # Sauvegarde du profil et du score pour les affichages suivants
                st.session_state.prediction_result = {
                    "input_data": input_data.copy(),
                    "probability": probability
                }

            except Exception as error:

                st.error(
                    f"Erreur pendant la prédiction : {error}"
                )

                st.stop()

        # ----------------------------------------------------
        # 10.5 AFFICHAGE DU RÉSULTAT
        # ----------------------------------------------------

        result = st.session_state.prediction_result

        if result is None:

            st.markdown("---")

            st.markdown(
                """
                <div class="feature-card">
                    <div class="feature-number">EN ATTENTE</div>
                    <h3>Aucune prédiction effectuée</h3>
                    <p>
                        Complétez le formulaire ci-dessus et lancez
                        la prédiction pour afficher le score du client,
                        la décision métier et son explication SHAP.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            input_data = result["input_data"]
            proba = result["probability"]

            # La décision est recalculée à chaque modification du seuil
            is_high_risk = proba >= threshold

            st.markdown("---")
            st.markdown("### 2. Résultat de l'évaluation")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Probabilité de churn",
                    f"{proba:.1%}"
                )

            with m2:
                st.metric(
                    "Seuil d'alerte",
                    f"{threshold:.0%}"
                )

            with m3:
                st.metric(
                    "Écart au seuil",
                    f"{(proba - threshold) * 100:+.1f} points"
                )

            st.progress(
                min(max(proba, 0.0), 1.0)
            )

            if is_high_risk:

                st.error(
                    " **RISQUE ÉLEVÉ** — La probabilité de churn "
                    "atteint ou dépasse le seuil métier."
                )

            else:

                st.success(
                    "✓ SOUS LE SEUIL D'ALERTE — La probabilité "
                    "de churn reste inférieure au seuil sélectionné."
                )

            # ------------------------------------------------
            # 10.6 RECOMMANDATIONS PERSONNALISÉES
            # ------------------------------------------------

            st.markdown("---")
            st.markdown("### 3. Recommandations du conseiller")

            recommendations = []

            if is_high_risk:

                if contract == "Month-to-month":

                    recommendations.append(
                        "Proposer une offre avec engagement de 12 ou 24 mois."
                    )

                if internet_service == "Fiber optic":

                    recommendations.append(
                        "Vérifier la satisfaction du client concernant "
                        "la fibre optique et le rapport qualité-prix."
                    )

                if tech_support == "No":

                    recommendations.append(
                        "Étudier l'intérêt d'une offre incluant "
                        "le support technique."
                    )

                if online_security == "No":

                    recommendations.append(
                        "Présenter les options de sécurité en ligne "
                        "adaptées aux besoins du client."
                    )

                if not recommendations:

                    recommendations.append(
                        "Contacter le client pour identifier ses motifs "
                        "d'insatisfaction avant de proposer une solution."
                    )

            else:

                recommendations.append(
                    "Maintenir le suivi habituel du client et surveiller "
                    "l'évolution de son comportement."
                )

            for recommendation in recommendations:
                st.write(f"- {recommendation}")

            st.caption(
                "Les recommandations sont des règles métier indicatives. "
                "Elles ne sont pas générées automatiquement par le modèle."
            )

            # ------------------------------------------------
            # 10.7 EXPLICABILITÉ SHAP DU CLIENT ÉVALUÉ
            # ------------------------------------------------

            st.markdown("---")
            st.markdown("### 4. Pourquoi ce score ? — Explicabilité SHAP")

            try:

                preprocessor = pipeline.named_steps["preprocessor"]
                model = pipeline.named_steps["model"]

                # Transformation du seul client soumis
                transformed_client = preprocessor.transform(input_data)

                if hasattr(transformed_client, "toarray"):
                    transformed_client = transformed_client.toarray()

                feature_names = preprocessor.get_feature_names_out()

                transformed_client_df = pd.DataFrame(
                    transformed_client,
                    columns=feature_names
                )

                # Jeu de référence représentatif pour l'explication SHAP
                model_features = input_data.columns.tolist()

                background_data = df[model_features].copy()

                # Limitation de la taille du jeu de référence
                background_data = background_data.sample(
                    n=min(200, len(background_data)),
                    random_state=42
                )

                transformed_background = preprocessor.transform(
                    background_data
                )

                if hasattr(transformed_background, "toarray"):
                    transformed_background = transformed_background.toarray()

                transformed_background_df = pd.DataFrame(
                    transformed_background,
                    columns=feature_names
                )

                # Explication du modèle linéaire avec un fond représentatif
                explainer = shap.LinearExplainer(
                    model,
                    transformed_background_df
                )

                shap_values = explainer(
                    transformed_client_df
                )

                # Affichage du waterfall pour le client soumis
                fig = plt.figure(figsize=(10, 5))

                shap.plots.waterfall(
                    shap_values[0],
                    max_display=10,
                    show=False
                )

                st.pyplot(
                    plt.gcf(),
                    clear_figure=True,
                    use_container_width=True
                )

                plt.close("all")

                st.caption(
                    "Chaque barre représente la contribution d'une variable "
                    "au score du client évalué, par rapport à la référence "
                    "utilisée par SHAP. Les contributions positives poussent "
                    "la prédiction vers la classe Churn = Yes ; les "
                    "contributions négatives l'en éloignent."
                )

            except Exception as error:

                st.warning(
                    f"Explication SHAP indisponible : {error}"
                )

            # ------------------------------------------------
            # 10.8 PROFIL TRANSMIS AU MODÈLE
            # ------------------------------------------------

            with st.expander("Consulter le profil du client évalué"):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

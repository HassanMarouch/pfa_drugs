# app.py
# Phase 7 : Dashboard Streamlit interactif
# Sujet : Détection automatique des effets secondaires fréquents

import os
import glob
import pandas as pd
import plotly.express as px
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import sys
APP_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(APP_DIR, "src"))
from nlp import extract_effects, compute_frequency

# --------------------------------------------------------------------------- #
#  Configuration                                                               #
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Effets Secondaires — Drugs.com",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_COLORS = {
    "CNS":             "#7F77DD",
    "Psychiatric":     "#D4537E",
    "GI":              "#1D9E75",
    "Cardiovascular":  "#E24B4A",
    "Dermatological":  "#EF9F27",
    "Musculoskeletal": "#378ADD",
    "Respiratory":     "#5DCAA5",
    "Urogenital":      "#D85A30",
    "Sensory":         "#639922",
    "General":         "#888780",
    "Other":           "#B4B2A9",
}

# --------------------------------------------------------------------------- #
#  Chargement des données                                                      #
# --------------------------------------------------------------------------- #

@st.cache_data
def load_all_drugs() -> dict:
    """Charge tous les fichiers clean une seule fois."""
    pattern = os.path.join(APP_DIR, "data", "clean", "*_clean.csv")
    files = glob.glob(pattern)
    dfs = {}
    for f in files:
        name = os.path.basename(f).replace("_clean.csv", "")
        df = pd.read_csv(f, sep=";", encoding="utf-8-sig")

        # Extraire ou parser les effets
        if "effects" not in df.columns:
            df["effects"] = df["clean_text"].apply(extract_effects)
        else:
            df["effects"] = df["effects"].apply(
                lambda x: x.split("|") if isinstance(x, str) else []
            )

        df["effects_count"] = df["effects"].str.len()
        df["has_effects"]   = df["effects_count"] > 0
        dfs[name] = df
    return dfs


@st.cache_data
def get_freq_for_drug(drug_name: str) -> pd.DataFrame:
    """Calcule la fréquence des effets PAR médicament (cache par nom)."""
    all_data = load_all_drugs()
    if drug_name not in all_data:
        return pd.DataFrame()
    return compute_frequency(all_data[drug_name])


# --------------------------------------------------------------------------- #
#  Page vide                                                                   #
# --------------------------------------------------------------------------- #
def show_empty_state():
    st.title("💊 Détection des effets secondaires — Drugs.com")
    st.warning(
        "Aucune donnée trouvée dans `data/clean/`.\n\n"
        "**Lance d'abord le pipeline :**\n"
        "```bash\n"
        "python src/scraper.py\n"
        "python src/cleaner.py\n"
        "```"
    )


# --------------------------------------------------------------------------- #
#  Dashboard                                                                   #
# --------------------------------------------------------------------------- #
def show_dashboard(df, freq_df, selected_drug, min_freq, selected_categories, top_n):

    # Filtre selon les contrôles sidebar
    freq_filtered = freq_df[
        (freq_df["frequency_pct"] >= min_freq) &
        (freq_df["category"].isin(selected_categories))
    ].reset_index(drop=True)

    # ── En-tête ──────────────────────────────────────────────────────────────
    st.title(f"Analyse des effets secondaires — *{selected_drug.title()}*")
    st.caption("Détection automatique par lexique médical sur les avis Drugs.com")
    st.markdown("---")

    # ── KPIs ─────────────────────────────────────────────────────────────────
    total_reviews   = len(df)
    reviews_effects = int(df["has_effects"].sum())
    unique_effects  = len(freq_df)
    avg_per_review  = round(df["effects_count"].mean(), 2)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total avis", f"{total_reviews:,}")
    c2.metric("Avis avec ≥1 effet", f"{reviews_effects:,}",
              f"{reviews_effects / max(total_reviews, 1):.0%}")
    c3.metric("Effets distincts", unique_effects)
    c4.metric("Effets moyens / avis", avg_per_review)
    st.markdown("---")

    # ── Barplot + Pie ─────────────────────────────────────────────────────────
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"Top {top_n} effets les plus fréquents")
        top_data = freq_filtered.head(top_n).sort_values("frequency_pct")
        if top_data.empty:
            st.info("Aucun effet avec ces filtres.")
        else:
            fig_bar = px.bar(
                top_data, x="frequency_pct", y="effect",
                color="category", color_discrete_map=CATEGORY_COLORS,
                orientation="h",
                labels={"frequency_pct": "Fréquence (%)", "effect": ""},
                text="frequency_pct",
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(
                height=max(350, top_n * 22), showlegend=True,
                margin=dict(l=0, r=40, t=20, b=20),
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{selected_drug}")

    with right:
        st.subheader("Répartition par système")
        cat_counts = (
            freq_filtered.groupby("category")["count"].sum()
            .reset_index().sort_values("count", ascending=False)
        )
        if cat_counts.empty:
            st.info("Aucune catégorie disponible.")
        else:
            fig_pie = px.pie(
                cat_counts, values="count", names="category",
                color="category", color_discrete_map=CATEGORY_COLORS, hole=0.35,
            )
            fig_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_pie.update_layout(showlegend=False, height=350,
                                  margin=dict(l=0, r=0, t=20, b=0))
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{selected_drug}")

    # ── Wordcloud ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Nuage de mots — effets secondaires")
    if not freq_filtered.empty:
        wf = dict(zip(freq_filtered["effect"], freq_filtered["count"]))
        wc = WordCloud(
            width=1200, height=400, background_color="white",
            colormap="RdYlBu_r", max_words=80, collocations=False
        ).generate_from_frequencies(wf)
        fig_wc, ax_wc = plt.subplots(figsize=(14, 4))
        ax_wc.imshow(wc, interpolation="bilinear")
        ax_wc.axis("off")
        st.pyplot(fig_wc)
        plt.close(fig_wc)
    else:
        st.info("Pas de termes à afficher.")

    # ── Boxplot notes ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Note selon présence d'effets secondaires")
    if "rating" in df.columns and df["rating"].notna().any():
        df_r = df.copy()
        df_r["Groupe"] = df_r["has_effects"].map(
            {True: "Avec effets", False: "Sans effets"})
        fig_box = px.box(
            df_r.dropna(subset=["rating"]), x="Groupe", y="rating",
            color="Groupe",
            color_discrete_sequence=["#E24B4A", "#1D9E75"],
            labels={"rating": "Note (/10)"},
        )
        fig_box.update_layout(showlegend=False, height=320,
                              margin=dict(l=0, r=0, t=20, b=20))
        st.plotly_chart(fig_box, use_container_width=True, key=f"box_{selected_drug}")
    else:
        st.info("Données de notation insuffisantes.")

    # ── Download CSV ──────────────────────────────────────────────────────────
    st.markdown("---")
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        st.subheader("⬇️ Télécharger les résultats")
        # CSV des fréquences d'effets
        if not freq_filtered.empty:
            csv_freq = freq_filtered.to_csv(index=False, sep=";").encode("utf-8-sig")
            st.download_button(
                label="📥 Télécharger les fréquences (CSV)",
                data=csv_freq,
                file_name=f"{selected_drug}_effects_freq.csv",
                mime="text/csv",
            )

    with col_dl2:
        # CSV des avis annotés
        cols = ["drug_name", "rating",  "effects_count", "clean_text", "effects"]
        avail = [c for c in cols if c in df.columns]
        df_disp = df[avail].copy()
        df_disp["effects"] = df_disp["effects"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x)
        csv_avis = df_disp.to_csv(index=False, sep=";").encode("utf-8-sig")
        st.download_button(
            label="📥 Télécharger les avis annotés (CSV)",
            data=csv_avis,
            file_name=f"{selected_drug}_annotated.csv",
            mime="text/csv",
        )

    # ── Tableau brut ──────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 Voir les avis annotés"):
        st.dataframe(df_disp, use_container_width=True, height=300)

    st.caption("Projet PFA — Détection automatique des effets secondaires — Drugs.com")


# --------------------------------------------------------------------------- #
#  Point d'entrée principal                                                    #
# --------------------------------------------------------------------------- #
def main():
    st.sidebar.title("💊 Drugs.com")
    st.sidebar.markdown("---")

    # Bouton rechargement
    if st.sidebar.button("🔄 Recharger les données"):
        st.cache_data.clear()
        st.rerun()

    # ── Saisie nouveau médicament ──────────────────────────
    drug_input = st.sidebar.text_input(
        "💊 Nouveau médicament",
        placeholder="ex: metformin, ibuprofen..."
    )

    if st.sidebar.button("🔍 Analyser"):
        if drug_input.strip():
            drug_name = drug_input.strip().lower().replace(" ", "-")

            with st.spinner(f"Scraping {drug_name}..."):
                from scraper import scrape_drug, save_raw
                df_raw = scrape_drug(drug_name)
                save_raw(df_raw, drug_name)

            with st.spinner("Nettoyage..."):
                from cleaner import clean_dataframe, save_clean
                df_clean = clean_dataframe(df_raw)
                save_clean(df_clean, drug_name)

            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.warning("Entre un nom de médicament.")

    st.sidebar.markdown("---")

    # ── Sélection parmi médicaments déjà scrapés ───────────
    all_data = load_all_drugs()

    if not all_data:
        show_empty_state()
        return

    drug_names    = sorted(all_data.keys())
    selected_drug = st.sidebar.selectbox("📋 Médicaments analysés", drug_names)
    df            = all_data[selected_drug]
    freq_df       = get_freq_for_drug(selected_drug)

    # ── Filtres ────────────────────────────────────────────
    min_freq = st.sidebar.slider("Fréquence minimale (%)", 1, 50, 1)

    available_cats = sorted(freq_df["category"].unique().tolist()) if not freq_df.empty else []
    selected_categories = st.sidebar.multiselect(
        "Catégories",
        options=available_cats,
        default=available_cats,
    )

    top_n = st.sidebar.slider("Top N effets", 10, 40, 20)

    # ── Affichage ──────────────────────────────────────────
    show_dashboard(df, freq_df, selected_drug,
                   min_freq, selected_categories, top_n)


main()
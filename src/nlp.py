# src/nlp.py
# Phase 6 : Extraction des effets secondaires par lexique médical
# Sujet : Détection automatique des effets secondaires fréquents
# Approche : matching lexical multi-gram (unigrammes + bigrammes)

import os
import re
import glob
import collections
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud


# --------------------------------------------------------------------------- #
#  Lexique des effets secondaires                                              #
# (termes en anglais — langue des avis Drugs.com)                             #
# --------------------------------------------------------------------------- #
SIDE_EFFECTS_LEXICON: dict[str, str] = {
    # ── Système nerveux central ─────────────────────────────────────────────
    "headache":            "CNS",
    "migraine":            "CNS",
    "dizziness":           "CNS",
    "drowsiness":          "CNS",
    "fatigue":             "CNS",
    "tiredness":           "CNS",
    "insomnia":            "CNS",
    "sleep problems":      "CNS",
    "brain fog":           "CNS",
    "confusion":           "CNS",
    "memory loss":         "CNS",
    "concentration issues":"CNS",
    "tremors":             "CNS",
    "seizures":            "CNS",
    "numbness":            "CNS",
    "tingling":            "CNS",
    "anxiety":             "Psychiatric",
    "depression":          "Psychiatric",
    "mood swings":         "Psychiatric",
    "irritability":        "Psychiatric",
    "suicidal thoughts":   "Psychiatric",
    "hallucinations":      "Psychiatric",
    "panic attacks":       "Psychiatric",
    # ── Gastro-intestinal ───────────────────────────────────────────────────
    "nausea":              "GI",
    "vomiting":            "GI",
    "diarrhea":            "GI",
    "constipation":        "GI",
    "stomach pain":        "GI",
    "abdominal pain":      "GI",
    "bloating":            "GI",
    "indigestion":         "GI",
    "heartburn":           "GI",
    "loss of appetite":    "GI",
    "weight gain":         "GI",
    "weight loss":         "GI",
    "dry mouth":           "GI",
    # ── Cardiovasculaire ────────────────────────────────────────────────────
    "palpitations":        "Cardiovascular",
    "heart pounding":      "Cardiovascular",
    "chest pain":          "Cardiovascular",
    "high blood pressure": "Cardiovascular",
    "low blood pressure":  "Cardiovascular",
    "rapid heartbeat":     "Cardiovascular",
    "slow heartbeat":      "Cardiovascular",
    "swelling":            "Cardiovascular",
    "edema":               "Cardiovascular",
    # ── Cutané ──────────────────────────────────────────────────────────────
    "rash":                "Dermatological",
    "itching":             "Dermatological",
    "hives":               "Dermatological",
    "acne":                "Dermatological",
    "hair loss":           "Dermatological",
    "alopecia":            "Dermatological",
    "dry skin":            "Dermatological",
    "sweating":            "Dermatological",
    "flushing":            "Dermatological",
    "bruising":            "Dermatological",
    # ── Musculo-squelettique ────────────────────────────────────────────────
    "muscle pain":         "Musculoskeletal",
    "joint pain":          "Musculoskeletal",
    "back pain":           "Musculoskeletal",
    "muscle cramps":       "Musculoskeletal",
    "weakness":            "Musculoskeletal",
    "stiffness":           "Musculoskeletal",
    # ── Respiratoire ────────────────────────────────────────────────────────
    "cough":               "Respiratory",
    "shortness of breath": "Respiratory",
    "wheezing":            "Respiratory",
    "nasal congestion":    "Respiratory",
    "runny nose":          "Respiratory",
    "sore throat":         "Respiratory",
    # ── Urogénital ──────────────────────────────────────────────────────────
    "frequent urination":  "Urogenital",
    "urinary retention":   "Urogenital",
    "sexual dysfunction":  "Urogenital",
    "decreased libido":    "Urogenital",
    "irregular periods":   "Urogenital",
    # ── Oculaire / Auditif ──────────────────────────────────────────────────
    "blurred vision":      "Sensory",
    "vision changes":      "Sensory",
    "eye pain":            "Sensory",
    "tinnitus":            "Sensory",
    "ringing in ears":     "Sensory",
    # ── Métabolique / Général ───────────────────────────────────────────────
    "fever":               "General",
    "chills":              "General",
    "night sweats":        "General",
    "dry eyes":            "General",
    "increased thirst":    "General",
    "increased hunger":    "General",
}

# Tri par longueur décroissante pour privilégier les bigrammes/trigrammes
_SORTED_TERMS = sorted(SIDE_EFFECTS_LEXICON.keys(), key=len, reverse=True)

# Pré-compilation des patterns (word boundary)
_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE), term, category)
    for term, category in SIDE_EFFECTS_LEXICON.items()
]


# --------------------------------------------------------------------------- #
#  Extraction                                                                  #
# --------------------------------------------------------------------------- #

def extract_effects(text: str) -> list[str]:
    """
    Extrait la liste des effets secondaires mentionnés dans un texte.
    Retourne des termes canoniques (tels que dans le lexique).
    """
    if not isinstance(text, str) or not text.strip():
        return []
    found = []
    for pattern, term, _ in _PATTERNS:
        if pattern.search(text):
            found.append(term)
    return list(dict.fromkeys(found))   # déduplique en préservant l'ordre


def categorize_effects(effects: list[str]) -> dict[str, list[str]]:
    """Regroupe les effets par catégorie anatomique/système."""
    categories: dict[str, list[str]] = collections.defaultdict(list)
    for effect in effects:
        cat = SIDE_EFFECTS_LEXICON.get(effect, "Other")
        categories[cat].append(effect)
    return dict(categories)


def analyse_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique l'extraction sur tout le corpus.
    Ajoute les colonnes :
      - effects        : liste des effets détectés
      - effects_count  : nombre d'effets
      - has_effects    : booléen
    """
    df = df.copy()
    df["effects"] = df["clean_text"].apply(extract_effects)
    df["effects_count"] = df["effects"].str.len()
    df["has_effects"] = df["effects_count"] > 0
    pct = df["has_effects"].mean() * 100
    print(f"  [NLP] Avis avec ≥1 effet détecté : {pct:.1f}%")
    return df


def compute_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcule la fréquence absolue et relative de chaque effet secondaire.
    """
    all_effects = [e for effects in df["effects"] for e in effects]
    counter = collections.Counter(all_effects)
    total_reviews = len(df)

    rows = []
    for term, count in counter.most_common():
        rows.append({
            "effect":       term,
            "category":     SIDE_EFFECTS_LEXICON.get(term, "Other"),
            "count":        count,
            "frequency_pct": round(count / total_reviews * 100, 2),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  Visualisations                                                              #
# --------------------------------------------------------------------------- #

def plot_top_effects(freq_df: pd.DataFrame, drug_name: str, top_n: int = 20) -> str:
    """Barplot horizontal des N effets les plus fréquents."""
    os.makedirs("data/outputs", exist_ok=True)
    top = freq_df.head(top_n).sort_values("frequency_pct")

    # Palette par catégorie
    category_colors = {
        "CNS":            "#7F77DD",
        "Psychiatric":    "#D4537E",
        "GI":             "#1D9E75",
        "Cardiovascular": "#E24B4A",
        "Dermatological": "#EF9F27",
        "Musculoskeletal":"#378ADD",
        "Respiratory":    "#5DCAA5",
        "Urogenital":     "#D85A30",
        "Sensory":        "#639922",
        "General":        "#888780",
        "Other":          "#B4B2A9",
    }
    colors = [category_colors.get(c, "#888780") for c in top["category"]]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top["effect"], top["frequency_pct"], color=colors, edgecolor="white")
    ax.set_xlabel("Fréquence (%)", fontsize=11)
    ax.set_title(f"Top {top_n} effets secondaires — {drug_name}", fontsize=13, pad=12)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    path = f"data/outputs/{drug_name}_top_effects.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  [NLP] Barplot → {path}")
    return path


def plot_by_category(freq_df: pd.DataFrame, drug_name: str) -> str:
    """Pie chart de répartition par catégorie anatomique."""
    os.makedirs("data/outputs", exist_ok=True)
    cat_counts = freq_df.groupby("category")["count"].sum().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        cat_counts.values,
        labels=cat_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.82,
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(f"Répartition par système — {drug_name}", fontsize=13, pad=12)
    plt.tight_layout()

    path = f"data/outputs/{drug_name}_categories.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  [NLP] Pie chart → {path}")
    return path


def plot_wordcloud(freq_df: pd.DataFrame, drug_name: str) -> str:
    """Wordcloud pondéré par la fréquence."""
    os.makedirs("data/outputs", exist_ok=True)
    word_freq = dict(zip(freq_df["effect"], freq_df["count"]))

    wc = WordCloud(
        width=900, height=450,
        background_color="white",
        colormap="RdYlBu_r",
        max_words=60,
        collocations=False,
    ).generate_from_frequencies(word_freq)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(f"Effets secondaires — {drug_name}", fontsize=13, pad=10)
    plt.tight_layout()

    path = f"data/outputs/{drug_name}_wordcloud.png"
    plt.savefig(path, dpi=130)
    plt.close()
    print(f"  [NLP] Wordcloud → {path}")
    return path


def save_results(df_annotated: pd.DataFrame, freq_df: pd.DataFrame, drug_name: str) -> None:
    """Sauvegarde les résultats NLP."""
    os.makedirs("data/outputs", exist_ok=True)

    # Avis annotés (effets en JSON-like string)
    ann_path = f"data/outputs/{drug_name}_annotated.csv"
    df_out = df_annotated.copy()
    df_out["effects"] = df_out["effects"].apply(lambda x: "|".join(x))
    df_out.to_csv(ann_path, index=False, sep=";", encoding="utf-8-sig")
    print(f"  [NLP] Avis annotés → {ann_path}")

    # Table de fréquences
    freq_path = f"data/outputs/{drug_name}_effects_freq.csv"
    freq_df.to_csv(freq_path, index=False, sep=";", encoding="utf-8-sig")
    print(f"  [NLP] Fréquences → {freq_path}")


# --------------------------------------------------------------------------- #
#  Point d'entrée                                                              #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    clean_files = glob.glob("data/clean/*_clean.csv")
    if not clean_files:
        print("Aucun fichier propre trouvé dans data/clean/")

    for clean_path in clean_files:
        drug_name = os.path.basename(clean_path).replace("_clean.csv", "")
        print(f"\n{'='*55}")
        print(f"  NLP — extraction effets secondaires : {drug_name}")
        print(f"{'='*55}")

        df = pd.read_csv(clean_path, sep=";", encoding="utf-8-sig")
        df_annotated = analyse_corpus(df)
        freq_df = compute_frequency(df_annotated)

        print(f"\n  Top 10 effets détectés :")
        print(freq_df.head(10).to_string(index=False))

        plot_top_effects(freq_df, drug_name)
        plot_by_category(freq_df, drug_name)
        plot_wordcloud(freq_df, drug_name)
        save_results(df_annotated, freq_df, drug_name)
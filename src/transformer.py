# src/transformer.py
# Phase 6 : Encodage vectoriel TF-IDF + features complémentaires
# Sujet : Détection automatique des effets secondaires fréquents

import os
import glob
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import scipy.sparse as sp


# --------------------------------------------------------------------------- #
#  Stopwords anglais médicaux augmentés                                        #
# --------------------------------------------------------------------------- #
MEDICAL_STOPWORDS = {
    # Articles / pronoms
    "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she",
    "it", "its", "they", "them", "this", "that", "these", "those",
    "a", "an", "the", "and", "but", "or", "so", "yet", "nor",
    # Verbes courants non informatifs
    "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "shall", "should", "may", "might", "must", "can", "could",
    # Termes génériques dans les avis médicaux
    "drug", "medication", "medicine", "doctor", "prescribed", "prescription",
    "take", "taking", "taken", "use", "using", "used", "try", "tried",
    "work", "works", "worked", "help", "helped", "feel", "felt", "feeling",
    "one", "two", "three", "day", "week", "month", "year", "time",
    "also", "just", "really", "very", "much", "more", "back", "get", "got",
    "like", "well", "first", "since", "still", "even", "though", "after",
    "before", "than", "other", "about", "when", "now", "there", "start",
    "started", "stop", "stopped",
}


def build_tfidf(
    corpus: list[str],
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
) -> tuple[TfidfVectorizer, sp.csr_matrix]:
    """
    Entraîne un vectoriseur TF-IDF sur le corpus.
    Retourne (vectorizer, matrice TF-IDF).
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,               # ignore les termes trop rares
        max_df=0.90,            # ignore les termes trop fréquents (bruit)
        stop_words=list(MEDICAL_STOPWORDS),
        sublinear_tf=True,      # log(1 + tf) — atténue les hautes fréquences
        strip_accents="unicode",
    )
    matrix = vectorizer.fit_transform(corpus)
    print(f"  [Transformer] Vocabulaire : {len(vectorizer.vocabulary_)} termes")
    print(f"  [Transformer] Matrice TF-IDF : {matrix.shape}")
    return vectorizer, matrix


def build_features(df: pd.DataFrame) -> np.ndarray:
    """
    Construit les features numériques complémentaires :
    - note normalisée (0-1)
    - longueur du texte (normalisée)
    - vote utile normalisé
    """
    features = pd.DataFrame()

    if "rating" in df.columns:
        features["rating_norm"] = df["rating"].fillna(df["rating"].median()) / 10.0
    else:
        features["rating_norm"] = 0.5

    if "clean_text" in df.columns:
        features["text_length"] = df["clean_text"].str.split().str.len().fillna(0)
    else:
        features["text_length"] = 0

    if "useful_count" in df.columns:
        features["useful_norm"] = df["useful_count"].fillna(0).astype(float)
    else:
        features["useful_norm"] = 0.0

    # Normalisation Min-Max des features numériques
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(features)
    return scaled, scaler


def encode(df: pd.DataFrame, drug_name: str) -> dict:
    """
    Pipeline complet : TF-IDF + features numériques → matrice combinée.
    Sauvegarde le vectorizer et retourne les artefacts.
    """
    corpus = df["clean_text"].fillna("").tolist()

    vectorizer, tfidf_matrix = build_tfidf(corpus)
    num_features, scaler = build_features(df)

    # Combine TF-IDF sparse + features denses
    dense_sparse = sp.hstack([
        tfidf_matrix,
        sp.csr_matrix(num_features)
    ])

    print(f"  [Transformer] Matrice finale : {dense_sparse.shape}")

    # Sauvegarde du vectorizer pour réutilisation (cleaner + app)
    os.makedirs("data/outputs", exist_ok=True)
    vec_path = f"data/outputs/{drug_name}_tfidf_vectorizer.pkl"
    with open(vec_path, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"  [Transformer] Vectorizer sauvegardé → {vec_path}")

    # Top termes TF-IDF globaux
    feature_names = vectorizer.get_feature_names_out()
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_idx = mean_scores.argsort()[-30:][::-1]
    top_terms = [(feature_names[i], round(float(mean_scores[i]), 4)) for i in top_idx]

    return {
        "drug_name":   drug_name,
        "vectorizer":  vectorizer,
        "scaler":      scaler,
        "tfidf_matrix": tfidf_matrix,
        "full_matrix":  dense_sparse,
        "top_terms":    top_terms,
    }


def save_top_terms(top_terms: list, drug_name: str) -> str:
    """Sauvegarde les top termes TF-IDF en CSV pour inspection."""
    os.makedirs("data/outputs", exist_ok=True)
    path = f"data/outputs/{drug_name}_top_tfidf.csv"
    pd.DataFrame(top_terms, columns=["term", "mean_tfidf"]).to_csv(
        path, index=False, sep=";", encoding="utf-8-sig"
    )
    print(f"  [Transformer] Top termes → {path}")
    return path


if __name__ == "__main__":
    clean_files = glob.glob("data/clean/*_clean.csv")
    if not clean_files:
        print("Aucun fichier propre trouvé dans data/clean/")
    for clean_path in clean_files:
        drug_name = os.path.basename(clean_path).replace("_clean.csv", "")
        print(f"\n--- Encodage TF-IDF : {drug_name} ---")
        df = pd.read_csv(clean_path, sep=";", encoding="utf-8-sig")
        result = encode(df, drug_name)
        save_top_terms(result["top_terms"], drug_name)
        print("\n  Top 10 termes TF-IDF :")
        for term, score in result["top_terms"][:10]:
            print(f"    {term:<30} {score:.4f}")
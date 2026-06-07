# src/cleaner.py
# Phase 4 : Nettoyage et normalisation des textes
# Sujet : Détection automatique des effets secondaires fréquents

import os
import re
import unicodedata
import pandas as pd

# --------------------------------------------------------------------------- #
#  Patterns de nettoyage                                                       #
# --------------------------------------------------------------------------- #

# Abréviations médicales courantes à préserver (ne pas splitter)
MEDICAL_ABBREVS = {
    "mg", "ml", "mcg", "dr", "dr.", "vs", "approx", "b.p", "hr", "bpm",
}

_URL_RE      = re.compile(r"https?://\S+|www\.\S+")
_HTML_RE     = re.compile(r"<[^>]+>")
_PUNCT_RE    = re.compile(r"[^\w\s\-',./]")   # garde apostrophes et tirets médicaux
_SPACES_RE   = re.compile(r"\s{2,}")
_DIGITS_ONLY = re.compile(r"^\d+$")


def _normalize_unicode(text: str) -> str:
    """Remplace les caractères Unicode exotiques par leurs équivalents ASCII."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def _remove_noise(text: str) -> str:
    text = _HTML_RE.sub(" ", text)          # balises HTML résiduelles
    text = _URL_RE.sub(" ", text)           # URLs
    text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    return text


def _clean_punctuation(text: str) -> str:
    text = _PUNCT_RE.sub(" ", text)
    text = _SPACES_RE.sub(" ", text)
    return text.strip()


def _normalize_rating(val) -> float | None:
    """Assure que la note est un float entre 1 et 10."""
    try:
        f = float(val)
        if 1.0 <= f <= 10.0:
            return round(f, 1)
    except (ValueError, TypeError):
        pass
    return None


def _parse_date(val) -> pd.Timestamp | None:
    """Convertit la date textuelle en Timestamp pandas."""
    try:
        return pd.to_datetime(val, errors="coerce")
    except Exception:
        return None


def clean_text(text: str) -> str:
    """
    Pipeline complet de nettoyage pour un texte brut.
    Retourne le texte nettoyé en minuscules.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = _normalize_unicode(text)
    text = _remove_noise(text)
    text = text.lower()
    text = _clean_punctuation(text)
    # Supprime les tokens purement numériques isolés
    tokens = [t for t in text.split() if not _DIGITS_ONLY.match(t)]
    return " ".join(tokens).strip()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applique le pipeline de nettoyage complet sur le DataFrame brut.
    """
    df = df.copy()

    print(f"  [Cleaner] Lignes initiales : {len(df)}")

    # 1. Supprimer les lignes sans texte
    df = df[df["review_text"].notna() & (df["review_text"].str.strip() != "")]
    print(f"  [Cleaner] Après suppression textes vides : {len(df)}")

    # 2. Nettoyer le texte brut → colonne clean_text
    df["clean_text"] = df["review_text"].apply(clean_text)

    # 3. Supprimer les textes trop courts après nettoyage (< 10 mots)
    df = df[df["clean_text"].str.split().str.len() >= 10]
    print(f"  [Cleaner] Après filtre longueur (≥10 mots) : {len(df)}")

    # 4. Normaliser la note
    df["rating"] = df["rating"].apply(_normalize_rating)

  
    # 5. Réinitialiser l'index
    df = df.reset_index(drop=True)

    print(f"  [Cleaner] Lignes finales propres : {len(df)}")
    return df


def save_clean(df: pd.DataFrame, drug_name: str) -> str:
    """Sauvegarde le DataFrame nettoyé."""
    os.makedirs("data/clean", exist_ok=True)
    path = f"data/clean/{drug_name}_clean.csv"
    df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    print(f"  [Cleaner] Sauvegardé → {path}  ({len(df)} lignes)")
    return path


if __name__ == "__main__":
    import glob
    raw_files = glob.glob("data/raw/*_raw.csv")
    if not raw_files:
        print("Aucun fichier brut trouvé dans data/raw/")
    for raw_path in raw_files:
        drug_name = os.path.basename(raw_path).replace("_raw.csv", "")
        print(f"\n--- Nettoyage : {drug_name} ---")
        df_raw = pd.read_csv(raw_path, sep=";", encoding="utf-8-sig")
        df_clean = clean_dataframe(df_raw)
        save_clean(df_clean, drug_name)
        print(df_clean[["clean_text", "rating"]].head(3))
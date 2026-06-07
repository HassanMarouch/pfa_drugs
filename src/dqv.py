# src/dqv.py
# Phase 5 : Contrôle qualité des données (Data Quality Validation)
# Sujet : Détection automatique des effets secondaires fréquents

import os
import glob
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
#  Seuils de qualité                                                           #
# --------------------------------------------------------------------------- #
THRESHOLDS = {
    "min_rows":           20,     # minimum d'avis par médicament
    "max_missing_text":   0.05,   # max 5 % de textes manquants
    "max_missing_rating": 0.30,   # max 30 % de notes manquantes
    "min_avg_words":      15,     # longueur moyenne minimale (mots)
    "max_duplicate_rate": 0.10,   # max 10 % de doublons
}


def _flag(condition: bool, msg_ok: str, msg_fail: str) -> tuple[bool, str]:
    if condition:
        return True, f"  PASS  {msg_ok}"
    return False, f"  FAIL  {msg_fail}"


def validate(df: pd.DataFrame, drug_name: str) -> dict:
    """
    Valide la qualité du DataFrame nettoyé.
    Retourne un rapport sous forme de dict.
    """
    report = {"drug": drug_name, "checks": [], "passed": 0, "failed": 0}

    # 1. Volume suffisant
    ok, msg = _flag(
        len(df) >= THRESHOLDS["min_rows"],
        f"Volume OK ({len(df)} avis)",
        f"Volume insuffisant ({len(df)} avis < {THRESHOLDS['min_rows']})"
    )
    report["checks"].append(msg)
    report["passed" if ok else "failed"] += 1

    # 2. Taux de textes manquants
    missing_text = df["clean_text"].isna().mean() if "clean_text" in df.columns else 1.0
    ok, msg = _flag(
        missing_text <= THRESHOLDS["max_missing_text"],
        f"Textes manquants OK ({missing_text:.1%})",
        f"Trop de textes manquants ({missing_text:.1%})"
    )
    report["checks"].append(msg)
    report["passed" if ok else "failed"] += 1

    # 3. Taux de notes manquantes
    missing_rating = df["rating"].isna().mean() if "rating" in df.columns else 1.0
    ok, msg = _flag(
        missing_rating <= THRESHOLDS["max_missing_rating"],
        f"Notes manquantes OK ({missing_rating:.1%})",
        f"Trop de notes manquantes ({missing_rating:.1%})"
    )
    report["checks"].append(msg)
    report["passed" if ok else "failed"] += 1

    # 4. Longueur moyenne des textes
    if "clean_text" in df.columns:
        avg_words = df["clean_text"].dropna().str.split().str.len().mean()
    else:
        avg_words = 0
    ok, msg = _flag(
        avg_words >= THRESHOLDS["min_avg_words"],
        f"Longueur moyenne OK ({avg_words:.1f} mots)",
        f"Textes trop courts en moyenne ({avg_words:.1f} mots)"
    )
    report["checks"].append(msg)
    report["passed" if ok else "failed"] += 1

    # 5. Taux de doublons
    if "clean_text" in df.columns:
        dup_rate = df.duplicated(subset=["clean_text"]).mean()
    else:
        dup_rate = 0
    ok, msg = _flag(
        dup_rate <= THRESHOLDS["max_duplicate_rate"],
        f"Doublons OK ({dup_rate:.1%})",
        f"Trop de doublons ({dup_rate:.1%})"
    )
    report["checks"].append(msg)
    report["passed" if ok else "failed"] += 1

    # Statistiques descriptives supplémentaires
    report["stats"] = {
        "n_rows":           len(df),
        "missing_text_pct": round(missing_text * 100, 2),
        "missing_rating_pct": round(missing_rating * 100, 2),
        "avg_words":        round(avg_words, 1),
        "duplicate_rate":   round(dup_rate * 100, 2),
        "rating_mean":      round(df["rating"].mean(), 2) if "rating" in df.columns else None,
        "rating_std":       round(df["rating"].std(), 2) if "rating" in df.columns else None,
    }

    return report


def print_report(report: dict) -> None:
    print(f"\n{'='*55}")
    print(f"  Rapport DQV — {report['drug']}")
    print(f"{'='*55}")
    for check in report["checks"]:
        print(check)
    print(f"{'─'*55}")
    print(f"  Résultat : {report['passed']} PASS / {report['failed']} FAIL")
    stats = report["stats"]
    print(f"\n  Statistiques :")
    print(f"    Avis total       : {stats['n_rows']}")
    print(f"    Note moyenne     : {stats['rating_mean']} ± {stats['rating_std']}")
    print(f"    Mots/avis (moy.) : {stats['avg_words']}")
    print(f"{'='*55}")


def plot_distributions(df: pd.DataFrame, drug_name: str) -> str:
    """
    Génère et sauvegarde un graphique de distribution
    des notes et des longueurs de texte.
    """
    os.makedirs("data/outputs", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Distribution — {drug_name}", fontsize=13)

    # Distribution des notes
    ax1 = axes[0]
    if "rating" in df.columns and df["rating"].notna().any():
        df["rating"].dropna().plot.hist(
            bins=10, ax=ax1, color="#378ADD", edgecolor="white"
        )
        ax1.set_title("Distribution des notes")
        ax1.set_xlabel("Note (/10)")
        ax1.set_ylabel("Nombre d'avis")
    else:
        ax1.text(0.5, 0.5, "Pas de données de notes", ha="center", va="center")

    # Distribution des longueurs de texte
    ax2 = axes[1]
    if "clean_text" in df.columns:
        word_counts = df["clean_text"].dropna().str.split().str.len()
        word_counts.plot.hist(
            bins=30, ax=ax2, color="#1D9E75", edgecolor="white"
        )
        ax2.set_title("Distribution longueur des avis")
        ax2.set_xlabel("Nombre de mots")
        ax2.set_ylabel("Nombre d'avis")

    plt.tight_layout()
    path = f"data/outputs/{drug_name}_dqv.png"
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  [DQV] Graphique sauvegardé → {path}")
    return path


if __name__ == "__main__":
    clean_files = glob.glob("data/clean/*_clean.csv")
    if not clean_files:
        print("Aucun fichier propre trouvé dans data/clean/")
    for clean_path in clean_files:
        drug_name = os.path.basename(clean_path).replace("_clean.csv", "")
        df = pd.read_csv(clean_path, sep=";", encoding="utf-8-sig")
        report = validate(df, drug_name)
        print_report(report)
        plot_distributions(df, drug_name)
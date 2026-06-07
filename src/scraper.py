import os
import time
import random
import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime

MAX_PAGES = 5

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]


def _get_headers() -> dict:
    """Retourne des headers avec un User-Agent aléatoire."""
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }


def _build_url(drug_name: str, page_num: int) -> str:
    if page_num == 1:
        return f"https://www.drugs.com/comments/{drug_name}/"
    return f"https://www.drugs.com/comments/{drug_name}/?page={page_num}"


def _parse_rating(comment) -> float | None:
    """Extrait la note (sur 10) depuis l'élément dédié.
    Structure réelle : <div class="ddc-rating-summary ..."><div>2 / 10</div>...
    """
    # Méthode 1 : div ddc-rating-summary → texte "2 / 10"
    rating_el = comment.find(class_="ddc-rating-summary")
    if rating_el:
        text = rating_el.get_text(separator=" ", strip=True)
        if "/" in text:
            try:
                return float(text.split("/")[0].strip())
            except ValueError:
                pass
 
    # Méthode 2 : attribut aria-label contenant "X / 10"
    rating_el = comment.find(attrs={"aria-label": lambda v: v and "/" in v})
    if rating_el:
        label = rating_el.get("aria-label", "")
        if "/" in label:
            try:
                return float(label.split("/")[0].strip())
            except ValueError:
                pass
 
    return None


def _parse_useful(useful_el) -> int:
    """Caste le compteur 'utile' en entier."""
    if useful_el is None:
        return 0
    try:
        return int(useful_el.get_text(strip=True))
    except ValueError:
        return 0


def scrape_drug(drug_name: str) -> pd.DataFrame:
    
    all_records = []

    for page_num in range(1, MAX_PAGES + 1):
        url = _build_url(drug_name, page_num)
        print(f"  [Scraper] Page {page_num}/{MAX_PAGES} → {url}")

        try:
            headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    }

            response = requests.get(
                     url,
                    headers=headers,
                    timeout=10
                      )

            print("Status Code:", response.status_code)
            print("Final URL:", response.url)
            response.encoding = response.apparent_encoding
        except requests.RequestException as e:
            print(f"  [Scraper] Erreur page {page_num}: {e}")
            break

        soup = BeautifulSoup(response.text, "html.parser")
        containers = soup.find_all("div", class_="ddc-comment")

        if not containers:
            print("  [Scraper] Aucun avis trouvé, arrêt.")
            break

        for comment in containers:
            # Texte du commentaire
            text_el = comment.find("p")
            if text_el:
                if text_el.find("b"):
                    text_el.find("b").decompose()
                review = text_el.get_text(strip=True)
            else:
                review = None
  
            # Note
            rating = _parse_rating(comment)

            # Condition traitée
            # Remplace tout le bloc condition dans ta boucle for comment
            

            all_records.append(
                {
                    "drug_name":    drug_name,
                    "rating":       rating,
                    "review_text":  review,
                    "collected_at": datetime.utcnow().isoformat(),
                }
            )

        print(f"  [Scraper] {len(containers)} avis extraits (page {page_num})")
        # Jitter aléatoire pour imiter un comportement humain
        time.sleep(random.uniform(1.5, 3.5))

    df = pd.DataFrame(all_records)

    # Déduplication sur le texte + date


    return df
   

def save_raw(df: pd.DataFrame, drug_name: str) -> str:
    """Sauvegarde le DataFrame brut et retourne le chemin du fichier."""
    os.makedirs("data/raw", exist_ok=True)
    path = f"data/raw/{drug_name}_raw.csv"
    df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    print(f"  [Scraper] Sauvegardé → {path}  ({len(df)} lignes)")
    return path


if __name__ == "__main__":
    drug = input("Médicament à scraper : ").strip().lower().replace(" ", "-")
    df = scrape_drug(drug)
    path = save_raw(df, drug)
    print(df.head())

from __future__ import annotations

import re
from typing import Any

import requests

API_BASE = "https://api.weather.gov"
HEADERS = {
    "User-Agent": "klas-kalshi-model/0.13 (KLAS temperature research)",
    "Accept": "application/geo+json",
}

HIGH_TERMS = (
    "outflow", "microburst", "dcape", "strong thunderstorm", "severe thunderstorm",
    "gust front", "damaging wind",
)
CONVECTIVE_TERMS = (
    "thunderstorm", "convection", "convective", "monsoon", "lightning", "virga",
    "cloud debris", "showers", "shower", "moisture surge",
)
NEGATIVE_PATTERNS = (
    r"no\s+(?:meaningful\s+)?thunderstorms?",
    r"thunderstorms?\s+(?:are\s+)?not\s+expected",
    r"convection\s+(?:is\s+)?unlikely",
    r"dry\s+conditions\s+(?:are\s+)?expected",
)


def analyze_afd_text(text: str) -> dict[str, Any]:
    compact = re.sub(r"\s+", " ", text or "").strip()
    lower = compact.lower()
    negatives = sum(1 for p in NEGATIVE_PATTERNS if re.search(p, lower))
    high_hits = [t for t in HIGH_TERMS if t in lower]
    conv_hits = [t for t in CONVECTIVE_TERMS if t in lower]

    score = 2 * len(high_hits) + len(conv_hits) - 2 * negatives
    if score >= 4 or len(high_hits) >= 1 and len(conv_hits) >= 1:
        risk = "HIGH"
    elif score >= 1:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    sentences = re.split(r"(?<=[.!?])\s+", compact)
    relevant: list[str] = []
    for sentence in sentences:
        s = sentence.lower()
        if any(t in s for t in HIGH_TERMS + CONVECTIVE_TERMS):
            relevant.append(sentence.strip())
        if len(relevant) >= 2:
            break
    snippet = " ".join(relevant)[:500] if relevant else "No notable convective wording found in the latest Las Vegas AFD."

    return {
        "risk": risk,
        "high_terms": high_hits,
        "convective_terms": conv_hits,
        "negative_signals": negatives,
        "snippet": snippet,
    }


def fetch_latest_vef_afd(timeout: int = 30) -> dict[str, Any]:
    """Fetch the latest NWS Las Vegas Area Forecast Discussion via api.weather.gov."""
    idx = requests.get(
        f"{API_BASE}/products/types/AFD/locations/VEF", headers=HEADERS, timeout=timeout
    )
    idx.raise_for_status()
    graph = idx.json().get("@graph", [])
    if not graph:
        raise RuntimeError("No VEF AFD products returned by NWS")
    latest = graph[0]
    product_id = latest.get("id")
    if not product_id:
        raise RuntimeError("Latest VEF AFD item has no product id")
    detail = requests.get(f"{API_BASE}/products/{product_id}", headers=HEADERS, timeout=timeout)
    detail.raise_for_status()
    payload = detail.json()
    text = payload.get("productText") or ""
    analysis = analyze_afd_text(text)
    return {
        "available": True,
        "issued_at": payload.get("issuanceTime") or latest.get("issuanceTime"),
        "product_id": product_id,
        "text": text,
        **analysis,
        "source": "NWS Las Vegas Area Forecast Discussion",
    }

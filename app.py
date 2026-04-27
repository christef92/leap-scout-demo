import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

st.title("🚀 LeapScout AI - LATAM Deal Sourcing")

# API KEY
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.warning("⚠️ Falta API Key")
    st.stop()

client = OpenAI(api_key=api_key)

# -------- EMBEDDINGS --------
@st.cache_data
def get_embeddings(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in res.data]

# -------- PORTFOLIO --------
def get_portfolio():
    return [
        "Kushki is a fintech company providing payment infrastructure in Latin America",
        "La Haus is a digital real estate marketplace in Latin America",
        "Uala is a neobank offering digital financial services",
        "Rappi is a delivery and logistics super app"
    ]

# -------- SCRAPING --------
def scrape_startups():
    queries = [
        "startup ronda seed mexico fintech",
        "startup pre seed colombia tecnologia",
        "startup etapa temprana peru startup",
        "startup seed chile saas"
    ]

    headers = {"User-Agent": "Mozilla/5.0"}
    results = []

    for query in queries:
        url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        for g in soup.select(".tF2Cxc")[:3]:
            title = g.select_one("h3")
            snippet = g.select_one(".VwiC3b")

            if title and snippet:
                text = f"{title.text}. {snippet.text}"
                results.append(text)

    return list(set(results))[:10]

# -------- EXTRACTION --------
def extract_info(text):
    text_lower = text.lower()

    # país
    countries = ["mexico", "colombia", "peru", "chile", "argentina", "latam"]
    country = next((c for c in countries if c in text_lower), "Unknown")

    # etapa
    if "pre-seed" in text_lower or "pre seed" in text_lower:
        stage = "Pre-Seed"
    elif "seed" in text_lower or "semilla" in text_lower:
        stage = "Seed"
    else:
        stage = "Unknown"

    # nombre (heurística básica)
    name = text.split(".")[0][:60]

    return name, country.title(), stage

# -------- FORMATTING --------
def clean_text(text):
    return f"Startup en Latinoamérica en etapa temprana: {text}"

# -------- MAIN --------
if st.button("🔎 Run Deal Sourcing"):

    portfolio = get_portfolio()
    startups_raw = scrape_startups()

    if len(startups_raw) == 0:
        st.error("No se encontraron startups")
        st.stop()

    # limpiar texto
    portfolio_clean = [clean_text(p) for p in portfolio]
    startups_clean = [clean_text(s) for s in startups_raw]

    # embeddings
    portfolio_emb = get_embeddings(portfolio_clean)
    startup_emb = get_embeddings(startups_clean)

    results = []

    for i, s in enumerate(startups_raw):
        emb = startup_emb[i]
        sims = cosine_similarity([emb], portfolio_emb)
        score = float(np.max(sims))

        name, country, stage = extract_info(s)

        results.append({
            "Startup": name,
            "Country": country,
            "Stage": stage,
            "Similarity Score": round(score, 3),
            "Contact": "N/A"
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Similarity Score", ascending=False)

    df["Rank"] = range(1, len(df)+1)
    df["Fit"] = df["Similarity Score"].apply(
        lambda x: "🔥 High" if x > 0.85 else ("👍 Medium" if x > 0.75 else "Low")
    )

    st.dataframe(df)

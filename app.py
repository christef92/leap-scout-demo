import streamlit as st
import pandas as pd
import feedparser
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

st.title("🚀 LeapScout AI - LATAM Deal Sourcing")

# -------- API KEY --------
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
        "startup latam seed",
        "startup fintech mexico seed",
        "startup colombia tecnologia seed",
        "startup peru pre-seed",
        "startup chile saas seed",
        "startup argentina ronda seed",
        "startup AI latam funding",
        "startup early stage latin america"
    ]

    results = []

    for q in queries:
        url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=es-419&gl=MX&ceid=MX:es-419"
        feed = feedparser.parse(url)

        for entry in feed.entries:
            text = f"{entry.title}. {entry.summary}"
            results.append(text)

    return list(set(results))[:50]

# -------- GPT EXTRACTION --------
@st.cache_data
def extract_structured(texts):
    structured = []

    for t in texts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Extract startup info in JSON:
                        {
                        "name": "",
                        "sector": "",
                        "country": "",
                        "stage": ""
                        }
                        Only LATAM startups. If unknown, use "Unknown".
                        """
                    },
                    {"role": "user", "content": t}
                ],
                temperature=0
            )

            data = json.loads(response.choices[0].message.content)
            structured.append(data)

        except:
            structured.append({
                "name": "Unknown",
                "sector": "Unknown",
                "country": "Unknown",
                "stage": "Unknown"
            })

    return structured

# -------- CLEAN TEXT --------
def clean_text(text):
    return f"Startup en Latinoamérica en etapa temprana: {text}"

# -------- MAIN --------
if st.button("🔎 Run Deal Sourcing"):

    portfolio = get_portfolio()
    startups_raw = scrape_startups()

    if len(startups_raw) == 0:
        st.error("No se encontraron startups")
        st.stop()

    # 🔥 extracción inteligente
    structured = extract_structured(startups_raw)

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

        info = structured[i]

        results.append({
            "Startup": info.get("name", "Unknown"),
            "Sector": info.get("sector", "Unknown"),
            "Country": info.get("country", "Unknown"),
            "Stage": info.get("stage", "Unknown"),
            "Similarity Score": round(score, 3)
        })

    df = pd.DataFrame(results)

    df = df.sort_values(by="Similarity Score", ascending=False)

    df["Rank"] = range(1, len(df)+1)
    df["Fit"] = df["Similarity Score"].apply(
        lambda x: "🔥 High" if x > 0.85 else ("👍 Medium" if x > 0.75 else "Low")
    )

    st.dataframe(df)

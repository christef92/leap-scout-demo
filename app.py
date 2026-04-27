import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

st.title("🚀 LeapScout AI - VC Grade Deal Sourcing")

# -------- API KEY --------
api_key = st.secrets.get("OPENAI_API_KEY", None)
if not api_key:
    st.warning("⚠️ Falta API Key")
    st.stop()

client = OpenAI(api_key=api_key)

# -------- DATASET REAL --------
@st.cache_data
def load_startups():
    url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Startups%20and%20venture%20capital/Startups%20and%20venture%20capital.csv"
    df = pd.read_csv(url)

    # usar columna de nombre (ajustar si cambia)
    df = df.head(100)

    return df

# -------- PORTFOLIO --------
def get_portfolio():
    return [
        "Kushki is a fintech company providing payment infrastructure in Latin America",
        "La Haus is a digital real estate marketplace in Latin America",
        "Uala is a neobank offering digital financial services",
        "Rappi is a delivery and logistics super app"
    ]

# -------- EMBEDDINGS --------
@st.cache_data
def get_embeddings(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in res.data]

# -------- GPT ENRICHMENT --------
@st.cache_data
def enrich_startups(names):
    enriched = []

    for name in names:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Return ONLY JSON:
{
"name": "",
"sector": "",
"country": "",
"stage": ""
}
"""
                    },
                    {
                        "role": "user",
                        "content": f"Startup: {name}. Identify sector, country and if it's early stage."
                    }
                ],
                temperature=0
            )

            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "")

            data = json.loads(content)

            if not isinstance(data, dict):
                raise ValueError

            enriched.append(data)

        except:
            enriched.append({
                "name": name,
                "sector": "Unknown",
                "country": "Unknown",
                "stage": "Unknown"
            })

    return enriched

# -------- MAIN --------
if st.button("🔎 Run VC Deal Sourcing"):

    df = load_startups()

    # tomar nombres reales
    startup_names = df.iloc[:, 0].dropna().astype(str).tolist()[:50]

    portfolio = get_portfolio()

    # enriquecer
    enriched = enrich_startups(startup_names)

    # textos para embeddings
    portfolio_clean = portfolio
    startups_clean = [e["name"] for e in enriched]

    portfolio_emb = get_embeddings(portfolio_clean)
    startup_emb = get_embeddings(startups_clean)

    results = []

    for i, s in enumerate(enriched):
        emb = startup_emb[i]
        sims = cosine_similarity([emb], portfolio_emb)
        score = float(np.max(sims))

        results.append({
            "Startup": s.get("name", "Unknown"),
            "Sector": s.get("sector", "Unknown"),
            "Country": s.get("country", "Unknown"),
            "Stage": s.get("stage", "Unknown"),
            "Similarity Score": round(score, 3)
        })

    df_res = pd.DataFrame(results)

    df_res = df_res.sort_values(by="Similarity Score", ascending=False)

    df_res["Rank"] = range(1, len(df_res)+1)
    df_res["Fit"] = df_res["Similarity Score"].apply(
        lambda x: "🔥 High" if x > 0.85 else ("👍 Medium" if x > 0.75 else "Low")
    )

    st.dataframe(df_res)

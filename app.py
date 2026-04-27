    import streamlit as st
import pandas as pd
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("🚀 LeapScout AI")

api_key = st.secrets.get("OPENAI_API_KEY", None)

if not api_key:
    st.write("⚠️ Falta API Key")
    st.stop()

client = OpenAI(api_key=api_key)

# ✅ NUEVA función en batch
def get_embeddings(texts):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in res.data]

def get_portfolio():
    return [
        "Kushki payments infrastructure",
        "La Haus real estate marketplace",
        "Uala digital banking",
        "Rappi delivery platform"
    ]

def get_startups():
    return [
        "AI payments API Mexico",
        "Real estate AI Peru",
        "Fintech AI Chile",
        "Logistics Colombia",
        "AI agriculture Peru",
        "Legal AI Mexico"
    ]

if st.button("Run Deal Sourcing"):

    portfolio = get_portfolio()
    startups = get_startups()

    # ✅ embeddings en UNA sola llamada
    portfolio_emb = get_embeddings(portfolio)
    startup_emb = get_embeddings(startups)

    results = []

    # ✅ ya no llamamos a OpenAI dentro del loop
    for i, s in enumerate(startups):
        emb = startup_emb[i]
        sims = cosine_similarity([emb], portfolio_emb)
        score = float(np.max(sims))

        results.append({
            "Startup": s,
            "Score": round(score, 3)
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)

    df["Rank"] = range(1, len(df)+1)
    df["Tier"] = df["Rank"].apply(lambda x: "Tier 1" if x <= 3 else "Tier 2")

    st.write(df)

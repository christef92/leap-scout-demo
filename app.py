import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

st.title("🚀 LeapScout AI")

api_key = st.secrets.get("OPENAI_API_KEY", None)

if not api_key:
    st.write("⚠️ Falta API Key")
    st.stop()

client = OpenAI(api_key=api_key)

def get_embedding(text):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return res.data[0].embedding

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

    portfolio_emb = [get_embedding(p) for p in portfolio]

    results = []

    for s in startups:
        emb = get_embedding(s)
        sims = cosine_similarity([emb], portfolio_emb)
        score = float(np.max(sims))

        results.append({
            "Startup": s,
            "Score": round(score, 3)
        })

    df = pd.DataFrame(results)
    df = df.sort_values(by="Score", ascending=False)

    df["Rank"] = range(1, len(df)+1)
    df["Tier"] = df["Rank"].apply(lambda x: "Tier 1" if x <= 66 else "Tier 2")

    st.write(df)

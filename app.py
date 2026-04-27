import streamlit as st
import pandas as pd
import feedparser
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

st.title("🚀 VC OS - Latin Leap")

# -------- CONFIG --------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# -------- PORTFOLIO --------
PORTFOLIO = [
    "Fintech payments infrastructure in Latin America",
    "Real estate marketplace LATAM",
    "Digital banking / neobank LATAM",
    "Logistics and delivery platforms"
]

# -------- 1. SOURCE --------
def fetch_sources():
    queries = [
        "startup latam funding",
        "fintech mexico ronda",
        "startup colombia seed",
        "startup chile inversion",
        "startup argentina venture capital",
        "startup peru pre-seed",
        "AI startup latam funding"
    ]

    data = []

    for q in queries:
        url = f"https://news.google.com/rss/search?q={q.replace(' ', '+')}&hl=es-419"
        feed = feedparser.parse(url)

        for entry in feed.entries:
            data.append(f"{entry.title}. {entry.summary}")

    return list(set(data))

# -------- 2. CLEAN --------
def clean_data(texts):
    filtered = []

    for t in texts:
        tl = t.lower()

        if any(c in tl for c in ["mexico","colombia","peru","chile","argentina","latam"]):
            filtered.append(t)

    return filtered[:100]

# -------- 3. ENTITY EXTRACTION --------
@st.cache_data
def extract_entities(texts):
    results = []

    for t in texts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Extract startup info. Return ONLY JSON:

{
"name": "",
"sector": "",
"country": "",
"stage": ""
}
"""
                    },
                    {"role": "user", "content": t}
                ],
                temperature=0
            )

            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "")

            data = json.loads(content)

            if isinstance(data, dict):
                results.append(data)

        except:
            continue

    return results

# -------- 4. EMBEDDINGS (FIXED BATCH) --------
@st.cache_data
def get_embeddings(texts):
    # limpiar textos
    texts = [str(t)[:300] for t in texts if t and isinstance(t, str)]

    # límite total
    texts = texts[:66]

    SUB_BATCH = 20
    all_embeddings = []

    for i in range(0, len(texts), SUB_BATCH):
        batch = texts[i:i+SUB_BATCH]

        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )

        all_embeddings.extend([d.embedding for d in res.data])

    return all_embeddings

# -------- 5. SCORING --------
def score_startups(startups):

    valid_startups = [
        s for s in startups
        if isinstance(s.get("name"), str) and len(s["name"]) > 2
    ][:66]

    if len(valid_startups) == 0:
        return []

    portfolio_emb = get_embeddings(PORTFOLIO)
    startup_names = [s["name"] for s in valid_startups]
    startup_emb = get_embeddings(startup_names)

    scored = []

    for i, s in enumerate(valid_startups):
        sims = cosine_similarity([startup_emb[i]], portfolio_emb)
        score = float(np.max(sims))

        scored.append({
            "Startup": s.get("name", "Unknown"),
            "Sector": s.get("sector", "Unknown"),
            "Country": s.get("country", "Unknown"),
            "Stage": s.get("stage", "Unknown"),
            "Score": round(score, 3)
        })

    return scored

# -------- 6. OUTPUT --------
def build_table(data):
    df = pd.DataFrame(data)

    df = df.sort_values(by="Score", ascending=False)

    df["Rank"] = range(1, len(df)+1)
    df["Fit"] = df["Score"].apply(
        lambda x: "🔥 High" if x > 0.85 else ("👍 Medium" if x > 0.75 else "Low")
    )

    return df

# -------- MAIN --------
if st.button("🚀 Run VC OS"):

    # 1. source
    raw = fetch_sources()

    # 2. clean
    clean = clean_data(raw)

    # 3. extract
    startups = extract_entities(clean)

    if len(startups) == 0:
        st.error("No startups extracted")
        st.stop()

    # 4. score
    scored = score_startups(startups)

    if len(scored) == 0:
        st.error("No valid startups after cleaning")
        st.stop()

    # 5. output
    df = build_table(scored)

    st.dataframe(df)

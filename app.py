import streamlit as st
import pandas as pd
import feedparser
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

st.title("🚀 VC OS - Latin Leap (Early Stage Focus)")

# -------- CONFIG --------
client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))

# -------- PORTFOLIO --------
PORTFOLIO = [
    "Fintech payments infrastructure in Latin America",
    "Real estate marketplace LATAM",
    "Digital banking / neobank LATAM",
    "Logistics and delivery platforms"
]

# -------- SOURCE --------
def fetch_sources():
    queries = [
        "startup latam seed",
        "fintech mexico ronda seed",
        "startup colombia pre-seed",
        "startup chile inversion seed",
        "startup argentina venture capital early stage",
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

# -------- CLEAN --------
def clean_data(texts):
    filtered = []
    for t in texts:
        tl = t.lower()
        if any(c in tl for c in ["mexico","colombia","peru","chile","argentina","latam"]):
            filtered.append(t)
    return filtered[:100]

# -------- ENTITY EXTRACTION --------
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
                        "content": """You are a VC analyst.

Extract startup info.

Return ONLY JSON:

{
"name": "",
"sector": "",
"country": "",
"stage": "",
"founders": "",
"founder_background": ""
}

Stage rules:
- Only use: Pre-Seed, Seed, Series A, Growth
- If IPO, unicorn, Series B+, mark as "Growth"

Infer missing info when possible.
Never leave empty fields.
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

# -------- FILTER STAGE --------
def is_early_stage(stage):
    stage = stage.lower()

    if any(x in stage for x in ["growth", "ipo", "series b", "series c", "unicorn"]):
        return False

    return True

# -------- ENRICH UNKNOWN --------
def enrich_unknown(s):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Improve startup data.

Return JSON:
{
"sector": "",
"country": "",
"stage": ""
}

Infer best possible values.
"""
                },
                {
                    "role": "user",
                    "content": f"Startup: {s['name']}. Fill missing info."
                }
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "")

        data = json.loads(content)

        for k in ["sector","country","stage"]:
            if s.get(k) == "Unknown" and data.get(k):
                s[k] = data[k]

    except:
        pass

    return s

# -------- EMBEDDINGS --------
@st.cache_data
def get_embeddings(texts):
    texts = [str(t)[:300] for t in texts if t and isinstance(t, str)]
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

# -------- FOUNDER SCORE --------
def score_founder(background):
    bg = background.lower()

    if any(x in bg for x in ["ex-google", "ex-amazon", "ex-mckinsey", "ex-rappi", "ex-uber"]):
        return 3
    elif any(x in bg for x in ["startup", "tech", "fintech", "engineer"]):
        return 2
    elif bg == "unknown":
        return 0
    else:
        return 1

# -------- SCORING --------
def score_startups(startups):

    valid_startups = [
        s for s in startups
        if isinstance(s.get("name"), str) and len(s["name"]) > 2
    ][:66]

    portfolio_emb = get_embeddings(PORTFOLIO)
    startup_names = [s["name"] for s in valid_startups]
    startup_emb = get_embeddings(startup_names)

    scored = []

    for i, s in enumerate(valid_startups):

        # 🔥 enrich missing data
        if "Unknown" in [s.get("sector"), s.get("country"), s.get("stage")]:
            s = enrich_unknown(s)

        # 🔥 filtrar late stage
        if not is_early_stage(s.get("stage", "")):
            continue

        sims = cosine_similarity([startup_emb[i]], portfolio_emb)
        similarity = float(np.max(sims))

        f_score = score_founder(s.get("founder_background", ""))

        scored.append({
            "Startup": s.get("name", "Unknown"),
            "Sector": s.get("sector", "Unknown"),
            "Country": s.get("country", "Unknown"),
            "Stage": s.get("stage", "Unknown"),
            "Founders": s.get("founders", "Unknown"),
            "Founder Background": s.get("founder_background", "Unknown"),
            "Similarity Score": round(similarity, 3),
            "Founder Score": f_score
        })

    return scored

# -------- OUTPUT --------
def build_table(data):
    df = pd.DataFrame(data)

    if df.empty:
        return df

    df["Final Score"] = (
        df["Similarity Score"] * 0.7 +
        df["Founder Score"] * 0.3
    )

    df = df.sort_values(by="Final Score", ascending=False)

    df["Rank"] = range(1, len(df)+1)
    df["Fit"] = df["Final Score"].apply(
        lambda x: "🔥 High" if x > 0.85 else ("👍 Medium" if x > 0.7 else "Low")
    )

    return df

# -------- MAIN --------
if st.button("🚀 Run VC OS"):

    raw = fetch_sources()
    clean = clean_data(raw)
    startups = extract_entities(clean)

    if len(startups) == 0:
        st.error("No startups extracted")
        st.stop()

    scored = score_startups(startups)

    if len(scored) == 0:
        st.warning("No early-stage startups found")
        st.stop()

    df = build_table(scored)

    st.dataframe(df)

import streamlit as st
import pandas as pd
import feedparser
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

st.title("🚀 VC OS - Latin Leap (Clean Deal Flow)")

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
        "startup argentina early stage",
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
    return [
        t for t in texts
        if any(c in t.lower() for c in ["mexico","colombia","peru","chile","argentina","latam"])
    ][:100]

# -------- EXTRACT --------
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
                        "content": """Extract startup info.

Return ONLY JSON:

{
"name": "",
"sector": "",
"country": "",
"stage": "",
"founders": "",
"founder_background": ""
}

Rules:
- Only startups
- If not a startup → name = "Invalid"
- Infer missing data
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

# -------- VALIDATION --------
def is_real_startup(name):
    name_lower = name.lower()

    blacklist = [
        "top", "latam startups", "startup", "ventures",
        "capital", "vc", "fund", "list", "ranking",
        "ecosystem", "report"
    ]

    if len(name.split()) > 5:
        return False

    if any(b in name_lower for b in blacklist):
        return False

    if name_lower in ["startup", "company", "business"]:
        return False

    return True


def validate_startup(name):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Answer YES or NO: is this a real startup?"
                },
                {"role": "user", "content": name}
            ],
            temperature=0
        )

        return "yes" in response.choices[0].message.content.lower()

    except:
        return True

# -------- STAGE FILTER --------
def is_early_stage(stage):
    stage = stage.lower()

    if any(x in stage for x in ["ipo", "growth", "series b", "series c", "unicorn"]):
        return False

    return True

# -------- ENRICH --------
def enrich_unknown(s):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """Fill missing startup info.

Return JSON:
{
"sector": "",
"country": "",
"stage": ""
}
"""
                },
                {"role": "user", "content": f"Startup: {s['name']}"}
            ],
            temperature=0
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "")

        data = json.loads(content)

        for k in ["sector", "country", "stage"]:
            if s.get(k) == "Unknown":
                s[k] = data.get(k, s[k])

    except:
        pass

    return s

# -------- EMBEDDINGS --------
@st.cache_data
def get_embeddings(texts):
    texts = [str(t)[:300] for t in texts if t]
    texts = texts[:66]

    all_embeddings = []
    for i in range(0, len(texts), 20):
        res = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts[i:i+20]
        )
        all_embeddings.extend([d.embedding for d in res.data])

    return all_embeddings

# -------- FOUNDER SCORE --------
def score_founder(bg):
    bg = bg.lower()

    if any(x in bg for x in ["ex-google","ex-amazon","ex-mckinsey","ex-rappi","ex-uber"]):
        return 3
    elif any(x in bg for x in ["startup","tech","fintech"]):
        return 2
    elif bg == "unknown":
        return 0
    else:
        return 1

# -------- SCORING --------
def score_startups(startups):

    valid = []
    for s in startups:
        name = s.get("name", "")
        if name != "Invalid" and is_real_startup(name) and validate_startup(name):
            valid.append(s)

    valid = valid[:66]

    portfolio_emb = get_embeddings(PORTFOLIO)
    startup_emb = get_embeddings([s["name"] for s in valid])

    results = []

    for i, s in enumerate(valid):

        s = enrich_unknown(s)

        if not is_early_stage(s.get("stage", "")):
            continue

        sims = cosine_similarity([startup_emb[i]], portfolio_emb)
        sim = float(np.max(sims))

        f_score = score_founder(s.get("founder_background", ""))

        results.append({
            "Startup": s["name"],
            "Sector": s["sector"],
            "Country": s["country"],
            "Stage": s["stage"],
            "Founders": s["founders"],
            "Founder Score": f_score,
            "Similarity Score": round(sim, 3)
        })

    return results

# -------- OUTPUT --------
def build_table(data):
    df = pd.DataFrame(data)

    if df.empty:
        return df

    df["Final Score"] = df["Similarity Score"] * 0.7 + df["Founder Score"] * 0.3
    df = df.sort_values(by="Final Score", ascending=False)

    df["Rank"] = range(1, len(df)+1)

    return df

# -------- MAIN --------
if st.button("🚀 Run VC OS"):

    raw = fetch_sources()
    clean = clean_data(raw)
    startups = extract_entities(clean)

    if not startups:
        st.error("No startups extracted")
        st.stop()

    scored = score_startups(startups)

    if not scored:
        st.warning("No valid startups after filtering")
        st.stop()

    df = build_table(scored)
    st.dataframe(df)

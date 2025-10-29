import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Therapy Matcher", page_icon="🧬", layout="wide")

# ============================================================
# 1️⃣ Load Data
# ============================================================
@st.cache_data
def load_all():
    civic = pd.read_csv("civic_lung_therapy_map.csv")
    emb_df = pd.read_csv("graph_embeddings.csv", index_col=0)
    depmap = pd.read_csv("depmap_drug_sensitivity.csv")
    fda = pd.read_csv("fda_drugs.csv")
    approved = set(fda["drug_name"].str.title())
    return civic, emb_df, depmap, approved

civic, emb_df, depmap, approved = load_all()

st.sidebar.success(f"✅ Loaded {len(emb_df)} embeddings")

# ============================================================
# 2️⃣ Build lookup functions
# ============================================================
import networkx as nx
G = nx.MultiDiGraph()
for _, r in civic.iterrows():
    gene = r["gene"].upper()
    variant = f"{r['gene'].upper()}_{r['variant']}"
    drug = r["drug"].title()
    disease = r["disease"].title()
    G.add_node(gene, type="gene")
    G.add_node(variant, type="variant")
    G.add_node(drug, type="drug")
    G.add_node(disease, type="disease")
    G.add_edge(variant, drug, relation="treated_by",
               evidence=r["evidence_level"],
               rationale=r["rationale_text"],
               source=r["source_url"])

# ============================================================
# 3️⃣ Recommendation Function
# ============================================================
def recommend_therapies(gene, variant, top_k=5):
    node = f"{gene.upper()}_{variant}"
    if node not in emb_df.index:
        st.warning(f"{node} not in embeddings.")
        return pd.DataFrame()

    v = emb_df.loc[node].values.reshape(1, -1)
    drug_nodes = [n for n, d in G.nodes(data=True) if d["type"] == "drug"]
    sims = cosine_similarity(v, emb_df.loc[drug_nodes])[0]
    idx = np.argsort(sims)[::-1][:top_k]

    df = pd.DataFrame({"Drug": [drug_nodes[i] for i in idx],
                       "Similarity": sims[idx].round(3)})
    df = df.merge(depmap, left_on="Drug", right_on="drug_name", how="left")
    df["Approval_Status"] = df["Drug"].apply(
        lambda x: "FDA-approved" if x in approved else "Experimental"
    )

    results = []
    for _, r in df.iterrows():
        rationale, level, source = "–", "NA", "–"
        if G.has_edge(node, r["Drug"]):
            e = list(G.get_edge_data(node, r["Drug"]).values())[0]
            rationale = e.get("rationale", "–")[:160] + "…"
            level = e.get("evidence", "NA")
            source = e.get("source", "–")
        results.append((r["Drug"], r["Similarity"], r["Approval_Status"],
                        level, rationale, source))
    return pd.DataFrame(results,
        columns=["Drug", "Similarity", "Approval_Status", "Evidence", "Rationale", "Source"])

# ============================================================
# 4️⃣ Streamlit UI
# ============================================================
st.title("🧬 Precision Oncology Therapy Matcher")
st.write("Discover potential targeted therapies based on gene–variant profiles using CIViC knowledge graph embeddings.")

col1, col2 = st.columns(2)
gene = col1.text_input("Gene Symbol", "EGFR")
variant = col2.text_input("Variant", "L858R")
top_k = st.slider("Number of therapies to show", 3, 10, 5)

if st.button("🔎 Recommend Therapies"):
    with st.spinner("Analyzing..."):
        out = recommend_therapies(gene, variant, top_k)
        if not out.empty:
            st.success(f"Results for {gene}_{variant}")
            st.dataframe(out, use_container_width=True)
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download CSV", csv, f"{gene}_{variant}_recommendations.csv", "text/csv")
        else:
            st.error("No matches found.")

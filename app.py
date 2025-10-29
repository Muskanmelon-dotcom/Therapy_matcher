import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="Precision Oncology Therapy Recommender", layout="wide", page_icon="🧬")
st.title("🧬 Precision & Safety-Aware Oncology Therapy Recommender")
st.markdown("Upload graph, knowledge, sensitivity, and FDA safety data to recommend therapies for a given gene–variant pair.")

# ---------------------------
# FILE UPLOADS
# ---------------------------
st.header("1️⃣ Upload Required Files")

col1, col2 = st.columns(2)
with col1:
    emb_file = st.file_uploader("Graph Embeddings (CSV)", type="csv")
    civic_file = st.file_uploader("CIViC Therapy Map (CSV)", type="csv")
with col2:
    depmap_file = st.file_uploader("DepMap Drug Sensitivity (CSV)", type="csv")
    fda_file = st.file_uploader("FDA / UniTox Safety File (CSV/XLSX)", type=["csv", "xlsx"])

if not all([emb_file, civic_file, depmap_file, fda_file]):
    st.info("Please upload all four files above to continue.")
    st.stop()

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_embeddings(file_path):
    """Load embeddings CSV, handle both index-based and column-based versions."""
    df = pd.read_csv(file_path)
    
    # Check if there's a column named 'node'
    if "node" in df.columns:
        df.set_index("node", inplace=True)
    else:
        # If not, assume the first column is the node index
        df.set_index(df.columns[0], inplace=True)
    
    # Ensure the index has a consistent name
    df.index.name = "node"
    
    return df


@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

@st.cache_data
def load_fda(file):
    if file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    else:
        df = pd.read_csv(file)
    df = df.rename(columns={
        "Generic Name": "drug",
        "Cardiotoxicity Binary Rating": "cardiotoxic",
        "Dermatotoxicity Binary Rating": "dermatotoxic"
    })
    return df[["drug", "cardiotoxic", "dermatotoxic"]].drop_duplicates()

graph_df = load_embeddings(emb_file)
civic_df = load_csv(civic_file)
depmap_df = load_csv(depmap_file)
fda_df = load_fda(fda_file)

# ---------------------------
# USER INPUTS
# ---------------------------
st.header("2️⃣ Query Parameters")

col1, col2, col3 = st.columns(3)
with col1:
    gene = st.text_input("Gene", value="BRAF")
with col2:
    variant = st.text_input("Variant", value="V600E")
with col3:
    top_n = st.number_input("Number of Recommendations", 1, 10, 5)

query_node = f"{gene}_{variant}"

# ---------------------------
# RECOMMENDATION ENGINE
# ---------------------------
if st.button("🔍 Recommend Therapies"):
    if query_node not in graph_df.index:
        st.error(f"❌ Variant `{query_node}` not found in embeddings.")
        st.stop()

    # Compute similarities
    query_emb = graph_df.loc[[query_node]].values
    drug_nodes = [n for n in graph_df.index if n.startswith("DRUG_")]
    drug_embs = graph_df.loc[drug_nodes].values
    similarities = cosine_similarity(query_emb, drug_embs)[0]

    rec_df = pd.DataFrame({
        "drug": [d.replace("DRUG_", "") for d in drug_nodes],
        "similarity": similarities
    }).sort_values("similarity", ascending=False).head(top_n)

    # ---------------------------
    # MERGE CIVIC KNOWLEDGE GRAPH
    # ---------------------------
    civic_merge = civic_df[["gene", "variant", "drug", "evidence_level", "rationale"]].drop_duplicates()
    civic_merge = civic_merge.rename(columns={"evidence_level": "evidence"})
    rec_df = rec_df.merge(civic_merge, on="drug", how="left")

    # ---------------------------
    # MERGE DEPMAP DRUG SENSITIVITY
    # ---------------------------
    if "Drug" in depmap_df.columns:
        depmap_df = depmap_df.rename(columns={"Drug": "drug"})
    if "IC50" in depmap_df.columns:
        depmap_df["IC50"] = depmap_df["IC50"].astype(float)
    rec_df = rec_df.merge(depmap_df[["drug", "IC50"]].drop_duplicates(), on="drug", how="left")

    # ---------------------------
    # MERGE FDA SAFETY FLAGS
    # ---------------------------
    rec_df = rec_df.merge(fda_df, on="drug", how="left")
    rec_df["cardiotoxic"] = rec_df["cardiotoxic"].fillna("NA")
    rec_df["dermatotoxic"] = rec_df["dermatotoxic"].fillna("NA")
    rec_df["similarity"] = rec_df["similarity"].round(3)

    # ---------------------------
    # DISPLAY RESULTS
    # ---------------------------
    st.subheader("📊 Top Therapy Recommendations")
    st.dataframe(rec_df, use_container_width=True)

    # Download
    csv = rec_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", data=csv,
                       file_name=f"therapy_recommendations_{gene}_{variant}.csv", mime="text/csv")

    # ---------------------------
    # INTERPRETATION
    # ---------------------------
    st.markdown("""
    ---
    ### 🧠 How It Works
    1. Finds embedding-space neighbors to your input variant (`graph_embeddings.csv`).
    2. Links drugs via **CIViC knowledge graph** relationships (`civic_lung_therapy_map.csv`).
    3. Integrates **DepMap IC50 sensitivity** as a pharmacologic confidence metric.
    4. Adds **FDA toxicity** signals for safety-aware prioritization.
    """)

    st.success(f"✅ Generated {len(rec_df)} ranked therapy recommendations for **{gene}_{variant}**.")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# ============================================================
# 1️⃣  LOAD DATA (from repo, not upload)
# ============================================================

@st.cache_data
def load_all_data():
    try:
        emb_df = pd.read_csv("graph_embeddings.csv")
        if "node" in emb_df.columns:
            emb_df.set_index("node", inplace=True)
        else:
            emb_df.set_index(emb_df.columns[0], inplace=True)
        emb_df.index.name = "node"
    except Exception as e:
        st.error(f"Error loading embeddings: {e}")
        emb_df = pd.DataFrame()

    try:
        civic_df = pd.read_csv("civic_lung_therapy_map.csv")
    except Exception as e:
        st.error(f"Error loading CIViC therapy map: {e}")
        civic_df = pd.DataFrame()

    try:
        depmap_df = pd.read_csv("depmap_drug_sensitivity.csv")
    except Exception as e:
        st.error(f"Error loading DepMap data: {e}")
        depmap_df = pd.DataFrame()

    try:
        fda_df = pd.read_csv("fda_drugs.csv")
    except Exception as e:
        st.error(f"Error loading FDA drug data: {e}")
        fda_df = pd.DataFrame()

    # Ensure expected columns exist to prevent key errors
    for df, name in [(depmap_df, "DepMap"), (fda_df, "FDA")]:
        for col in ["drug", "ic50", "dermatotoxic", "hepatotoxic", "cardiotoxic", "nephrotoxic"]:
            if col not in df.columns:
                df[col] = np.nan
    
    return emb_df, civic_df, depmap_df, fda_df


emb_df, civic_df, depmap_df, fda_df = load_all_data()
st.sidebar.success("✅ Data successfully loaded from repository.")

# ============================================================
# 2️⃣  THERAPY RECOMMENDER FUNCTION
# ============================================================

def recommend_therapies(gene, variant, top_k=5):
    if emb_df.empty:
        st.error("Embeddings not loaded.")
        return None

    node = f"{gene.upper()}_{variant}"
    if node not in emb_df.index:
        st.warning(f"Variant node {node} not found in embeddings.")
        return None

    # Compute cosine similarity between variant node and all drug nodes
    v = emb_df.loc[node].values.reshape(1, -1)
    drug_nodes = [n for n in emb_df.index if any(x in n.lower() for x in ["ib", "inib", "mab", "nib", "tib", "raf", "met", "tinib"])]
    
    if len(drug_nodes) == 0:
        st.warning("No drug nodes found in embeddings.")
        return None
    
    drug_vectors = emb_df.loc[drug_nodes]
    sims = cosine_similarity(v, drug_vectors)[0]
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = pd.DataFrame({
        "Drug": [drug_nodes[i] for i in top_idx],
        "Similarity": [round(float(sims[i]), 3) for i in top_idx]
    })

    # Merge with CIViC evidence, toxicity & IC50
    if not civic_df.empty and "drug" in civic_df.columns:
        civic_cols = [c for c in ["drug", "evidence_level", "rationale_text", "source_url"] if c in civic_df.columns]
        results = results.merge(civic_df[civic_cols], how="left", left_on="Drug", right_on="drug")
        if "drug" in results.columns:
            results.drop(columns=["drug"], inplace=True)
    
    if not depmap_df.empty and "drug" in depmap_df.columns and "ic50" in depmap_df.columns:
        results = results.merge(depmap_df[["drug", "ic50"]], how="left", left_on="Drug", right_on="drug")
        if "drug" in results.columns:
            results.drop(columns=["drug"], inplace=True)
    
    # --- FDA Toxicity Merge ---
    if not fda_df.empty and "Generic Name" in fda_df.columns:
        fda_df["Generic Name"] = fda_df["Generic Name"].astype(str).str.strip().str.lower()
        results["Drug_lower"] = results["Drug"].astype(str).str.strip().str.lower()
        
        fda_cols = ["Generic Name"]
        for col in ["Cardiotoxicity Binary Rating", "DermatologicalToxicity Binary Rating"]:
            if col in fda_df.columns:
                fda_cols.append(col)
        
        results = results.merge(
            fda_df[fda_cols],
            how="left",
            left_on="Drug_lower",
            right_on="Generic Name"
        )
        results.drop(columns=["Generic Name", "Drug_lower"], errors="ignore", inplace=True)
        
        # Rename toxicity columns
        rename_dict = {}
        if "Cardiotoxicity Binary Rating" in results.columns:
            rename_dict["Cardiotoxicity Binary Rating"] = "Cardiotoxicity"
        if "DermatologicalToxicity Binary Rating" in results.columns:
            rename_dict["DermatologicalToxicity Binary Rating"] = "Dermatotoxicity"
        
        if rename_dict:
            results.rename(columns=rename_dict, inplace=True)

    # Final column renaming
    rename_map = {
        "evidence_level": "Evidence Level",
        "rationale_text": "Rationale",
        "source_url": "Source",
        "ic50": "IC50 (μM)",
        "dermatotoxic": "Dermal Toxicity",
        "hepatotoxic": "Liver Toxicity",
        "cardiotoxic": "Cardiac Toxicity",
        "nephrotoxic": "Renal Toxicity"
    }
    results.rename(columns={k: v for k, v in rename_map.items() if k in results.columns}, inplace=True)

    return results

# ============================================================
# 3️⃣  STREAMLIT UI
# ============================================================

st.title("🧬 TherapyMatcher — Precision Oncology Recommender")
st.markdown("""
Use biological embeddings + CIViC evidence + DepMap drug sensitivity to identify therapy matches
for specific **gene variants** in lung and related cancers.
""")

col1, col2 = st.columns(2)
gene = col1.text_input("Gene symbol", "EGFR")
variant = col2.text_input("Variant", "L858R")
top_k = st.slider("Top therapies to display", 3, 10, 5)

if st.button("🔍 Recommend Therapies"):
    results = recommend_therapies(gene, variant, top_k)
    if results is not None and not results.empty:
        st.success(f"Recommendations for {gene}_{variant}")
        st.dataframe(results, use_container_width=True)

        # CSV Download
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Results as CSV", csv, f"{gene}_{variant}_recommendations.csv", "text/csv")
    else:
        st.error("No matching drugs found for this variant.")

# ============================================================
# 4️⃣  EMBEDDING VISUALIZATION
# ============================================================

st.markdown("---")
st.subheader("📊 Node Embedding Map (PCA projection)")

try:
    if not emb_df.empty and len(emb_df) > 2:
        coords = PCA(n_components=2).fit_transform(emb_df.values)
        df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])
        df_plot["Type"] = ["drug" if "ib" in n.lower() or "mab" in n.lower() else "gene/variant" for n in emb_df.index]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df_plot, x="PC1", y="PC2", hue="Type",
                        palette={"drug": "#1f77b4", "gene/variant": "#ff7f0e"},
                        s=30, alpha=0.75, ax=ax)
        ax.set_title("Node2Vec Embedding Map (CIViC + DepMap Fusion)")
        st.pyplot(fig)
    else:
        st.warning("Not enough embeddings to visualize.")
except Exception as e:
    st.warning(f"Could not generate embedding plot: {e}")

st.caption("Powered by CIViC • DepMap • Node2Vec • Streamlit • FDA Toxicity Data")

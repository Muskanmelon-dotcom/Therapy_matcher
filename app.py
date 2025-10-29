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

    return emb_df, civic_df, depmap_df, fda_df


emb_df, civic_df, depmap_df, fda_df = load_all_data()

# Debug info in sidebar
with st.sidebar:
    st.success("✅ Data successfully loaded from repository.")
    with st.expander("📊 Dataset Info"):
        st.write(f"**Embeddings:** {len(emb_df)} nodes" if not emb_df.empty else "**Embeddings:** Not loaded")
        st.write(f"**CIViC Evidence:** {len(civic_df)} records" if not civic_df.empty else "**CIViC Evidence:** Not loaded")
        st.write(f"**DepMap Compounds:** {len(depmap_df)} compounds" if not depmap_df.empty else "**DepMap Compounds:** Not loaded")
        st.write(f"**FDA Drugs:** {len(fda_df)} drugs" if not fda_df.empty else "**FDA Drugs:** Not loaded")

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

    # Normalize drug names for merging
    results["Drug_normalized"] = results["Drug"].astype(str).str.strip().str.lower()

    # Merge with CIViC evidence
    if not civic_df.empty and "drug" in civic_df.columns:
        civic_merge = civic_df.copy()
        civic_merge["drug_normalized"] = civic_merge["drug"].astype(str).str.strip().str.lower()
        
        civic_cols = ["drug_normalized"]
        for col in ["evidence_level", "rationale_text", "source_url"]:
            if col in civic_merge.columns:
                civic_cols.append(col)
        
        results = results.merge(
            civic_merge[civic_cols],
            how="left",
            left_on="Drug_normalized",
            right_on="drug_normalized"
        )
        results.drop(columns=["drug_normalized"], errors="ignore", inplace=True)
    
    # Merge with DepMap data using CompoundName
    if not depmap_df.empty and "CompoundName" in depmap_df.columns:
        depmap_merge = depmap_df.copy()
        depmap_merge["drug_normalized"] = depmap_merge["CompoundName"].astype(str).str.strip().str.lower()
        
        # Include relevant DepMap columns
        depmap_cols = ["drug_normalized"]
        for col in ["GeneSymbolOfTargets", "TargetOrMechanism", "ChEMBLID"]:
            if col in depmap_merge.columns:
                depmap_cols.append(col)
        
        results = results.merge(
            depmap_merge[depmap_cols],
            how="left",
            left_on="Drug_normalized",
            right_on="drug_normalized"
        )
        results.drop(columns=["drug_normalized"], errors="ignore", inplace=True)
    
    # Merge with FDA Toxicity data
    if not fda_df.empty and "Generic Name" in fda_df.columns:
        fda_merge = fda_df.copy()
        fda_merge["drug_normalized"] = fda_merge["Generic Name"].astype(str).str.strip().str.lower()
        
        fda_cols = ["drug_normalized"]
        for col in ["Cardiotoxicity Binary Rating", "DermatologicalToxicity Binary Rating"]:
            if col in fda_merge.columns:
                fda_cols.append(col)
        
        results = results.merge(
            fda_merge[fda_cols],
            how="left",
            left_on="Drug_normalized",
            right_on="drug_normalized"
        )
        results.drop(columns=["drug_normalized"], errors="ignore", inplace=True)

    # Drop the normalized column used for merging
    results.drop(columns=["Drug_normalized"], errors="ignore", inplace=True)

    # Rename columns to user-friendly names
    rename_map = {
        "evidence_level": "Evidence Level",
        "rationale_text": "Rationale",
        "source_url": "Source",
        "GeneSymbolOfTargets": "Targets",
        "TargetOrMechanism": "Mechanism",
        "ChEMBLID": "ChEMBL ID",
        "Cardiotoxicity Binary Rating": "Cardiotoxicity",
        "DermatologicalToxicity Binary Rating": "Dermatotoxicity"
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
    with st.spinner("Analyzing embeddings and evidence..."):
        results = recommend_therapies(gene, variant, top_k)
    
    if results is not None and not results.empty:
        st.success(f"✅ Recommendations for {gene}_{variant}")
        
        # Display results with better formatting
        st.dataframe(
            results,
            use_container_width=True,
            column_config={
                "Drug": st.column_config.TextColumn("Drug", width="medium"),
                "Similarity": st.column_config.NumberColumn("Similarity", format="%.3f"),
                "Evidence Level": st.column_config.TextColumn("Evidence Level", width="small"),
                "Targets": st.column_config.TextColumn("Targets", width="large"),
                "Mechanism": st.column_config.TextColumn("Mechanism", width="medium"),
                "Source": st.column_config.LinkColumn("Source", width="small"),
            }
        )

        # CSV Download
        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Results as CSV",
            csv,
            f"{gene}_{variant}_recommendations.csv",
            "text/csv",
            key="download-csv"
        )
        
        # Display summary statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Drugs Found", len(results))
        with col2:
            avg_sim = results["Similarity"].mean()
            st.metric("Avg Similarity", f"{avg_sim:.3f}")
        with col3:
            if "Evidence Level" in results.columns:
                evidence_count = results["Evidence Level"].notna().sum()
                st.metric("With CIViC Evidence", evidence_count)
    else:
        st.error("❌ No matching drugs found for this variant.")

# ============================================================
# 4️⃣  EMBEDDING VISUALIZATION
# ============================================================

st.markdown("---")
st.subheader("📊 Node Embedding Map (PCA projection)")

try:
    if not emb_df.empty and len(emb_df) > 2:
        with st.spinner("Generating PCA visualization..."):
            coords = PCA(n_components=2).fit_transform(emb_df.values)
            df_plot = pd.DataFrame(coords, columns=["PC1", "PC2"])
            df_plot["Type"] = ["drug" if any(x in n.lower() for x in ["ib", "mab", "nib", "tinib"]) else "gene/variant" for n in emb_df.index]

            fig, ax = plt.subplots(figsize=(10, 7))
            sns.scatterplot(
                data=df_plot, 
                x="PC1", 
                y="PC2", 
                hue="Type",
                palette={"drug": "#1f77b4", "gene/variant": "#ff7f0e"},
                s=40, 
                alpha=0.7,
                ax=ax
            )
            ax.set_title("Node2Vec Embedding Map (CIViC + DepMap Fusion)", fontsize=14, fontweight="bold")
            ax.set_xlabel("Principal Component 1", fontsize=11)
            ax.set_ylabel("Principal Component 2", fontsize=11)
            ax.legend(title="Node Type", loc="best")
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            
            # Add explanation
            with st.expander("ℹ️ About this visualization"):
                st.write("""
                This 2D projection shows the relationships between gene variants and drugs in embedding space.
                - **Blue points**: Drug compounds
                - **Orange points**: Gene variants and molecular features
                - **Proximity**: Closer points suggest stronger biological relationships
                """)
    else:
        st.warning("⚠️ Not enough embeddings to visualize.")
except Exception as e:
    st.error(f"❌ Could not generate embedding plot: {e}")

# ============================================================
# 5️⃣  FOOTER
# ============================================================

st.markdown("---")
st.caption("🔬 Powered by CIViC • DepMap • Node2Vec • Streamlit • FDA Toxicity Data")

with st.expander("📖 About TherapyMatcher"):
    st.markdown("""
    **TherapyMatcher** leverages graph neural networks to recommend precision oncology therapies based on:
    
    - **CIViC**: Clinical interpretations of variants in cancer
    - **DepMap**: Drug sensitivity and target mechanism data
    - **FDA**: Toxicity profiles for safety assessment
    - **Node2Vec**: Graph embeddings capturing biological relationships
    
    **How it works:**
    1. Enter a gene and variant (e.g., EGFR L858R)
    2. The system computes similarity between the variant and all known drugs
    3. Results are ranked by embedding similarity and enriched with clinical evidence
    
    **Limitations:**
    - Recommendations are computational predictions, not clinical advice
    - Always validate with current literature and clinical guidelines
    - Drug approvals and evidence levels may have changed since data collection
    """)

# ============================================================
# 6️⃣  OPTIONAL: BATCH QUERY
# ============================================================

st.markdown("---")
st.subheader("🔄 Batch Query (Multiple Variants)")

with st.expander("Run batch analysis"):
    batch_input = st.text_area(
        "Enter variants (one per line, format: GENE VARIANT)",
        "EGFR L858R\nKRAS G12C\nBRAF V600E"
    )
    
    batch_top_k = st.slider("Top therapies per variant", 3, 10, 3, key="batch_slider")
    
    if st.button("🚀 Run Batch Analysis"):
        variants = [line.strip() for line in batch_input.split("\n") if line.strip()]
        
        batch_results = []
        progress_bar = st.progress(0)
        
        for idx, variant_line in enumerate(variants):
            parts = variant_line.split()
            if len(parts) >= 2:
                g, v = parts[0], parts[1]
                res = recommend_therapies(g, v, batch_top_k)
                if res is not None and not res.empty:
                    res["Query"] = f"{g}_{v}"
                    batch_results.append(res)
            
            progress_bar.progress((idx + 1) / len(variants))
        
        if batch_results:
            combined = pd.concat(batch_results, ignore_index=True)
            st.success(f"✅ Analyzed {len(variants)} variants")
            st.dataframe(combined, use_container_width=True)
            
            # Download batch results
            batch_csv = combined.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Batch Results",
                batch_csv,
                "batch_therapy_recommendations.csv",
                "text/csv",
                key="download-batch"
            )
        else:
            st.warning("No results found for any variants")

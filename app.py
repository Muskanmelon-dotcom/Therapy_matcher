import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import json
from openai import OpenAI

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
    
    st.markdown("---")
    st.subheader("🔑 OpenAI API Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password", help="Enter your OpenAI API key for LLM-generated rationales")
    use_llm = st.checkbox("Enable LLM Rationales", value=False, help="Generate clinical explanations using GPT-4")

# ============================================================
# 2️⃣  LLM RATIONALE GENERATOR
# ============================================================

def generate_rationale(drug, biomarkers, evidence_level, mechanism, targets, api_key):
    """Generate clinical rationale using OpenAI GPT-4"""
    if not api_key:
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        biomarker_str = ", ".join([f"{k}: {v}" for k, v in biomarkers.items() if v])
        
        prompt = f"""You are an oncology clinical decision support system. Generate a concise 2-3 sentence clinical rationale for recommending this therapy.

Drug: {drug}
Patient Biomarkers: {biomarker_str}
Evidence Level: {evidence_level if pd.notna(evidence_level) else 'Not specified'}
Mechanism: {mechanism if pd.notna(mechanism) else 'Not specified'}
Targets: {targets if pd.notna(targets) else 'Not specified'}

Provide a clear, evidence-based rationale that an oncologist would find helpful. Focus on:
1. Why this drug matches the biomarker profile
2. The strength of clinical evidence
3. Any relevant clinical considerations

Keep it professional, concise, and actionable."""

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert oncology clinical decision support assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0.3
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        st.warning(f"LLM rationale generation failed: {e}")
        return None

# ============================================================
# 3️⃣  ENHANCED THERAPY RECOMMENDER WITH CLINICAL CONTEXT
# ============================================================

def recommend_therapies_clinical(patient_profile, top_k=5, api_key=None, use_llm_rationales=False):
    """
    Recommend therapies based on full clinical profile
    patient_profile: dict with diagnosis, line_of_therapy, biomarkers, comorbidities, prior_therapies
    """
    if emb_df.empty:
        st.error("Embeddings not loaded.")
        return None
    
    biomarkers = patient_profile.get("biomarkers", {})
    line_of_therapy = patient_profile.get("line_of_therapy", "1L")
    diagnosis = patient_profile.get("diagnosis", "")
    
    # Find all positive biomarkers
    positive_biomarkers = {k: v for k, v in biomarkers.items() if v and v != "false" and v != False}
    
    if not positive_biomarkers:
        st.warning("No positive biomarkers found in patient profile.")
        return None
    
    # Try each biomarker and collect all recommendations
    all_results = []
    
    for gene, variant in positive_biomarkers.items():
        # Handle different variant formats
        if isinstance(variant, str) and variant.lower() in ['true', 'positive', 'present']:
            node = f"{gene.upper()}"
        else:
            node = f"{gene.upper()}_{variant}"
        
        # Fallback logic
        if node not in emb_df.index:
            gene_node = gene.upper()
            if gene_node in emb_df.index:
                node = gene_node
            else:
                continue
        
        # Compute similarities
        v = emb_df.loc[node].values.reshape(1, -1)
        drug_nodes = [n for n in emb_df.index if any(x in n.lower() for x in ["ib", "inib", "mab", "nib", "tib", "raf", "met", "tinib"])]
        
        if len(drug_nodes) == 0:
            continue
        
        drug_vectors = emb_df.loc[drug_nodes]
        sims = cosine_similarity(v, drug_vectors)[0]
        top_idx = np.argsort(sims)[::-1][:top_k*2]  # Get more candidates
        
        for idx in top_idx:
            all_results.append({
                "Drug": drug_nodes[idx],
                "Similarity": round(float(sims[idx]), 3),
                "Biomarker": f"{gene}_{variant}" if not isinstance(variant, bool) else gene
            })
    
    if not all_results:
        st.warning("No drug matches found for the biomarker profile.")
        return None
    
    # Convert to DataFrame and deduplicate
    results = pd.DataFrame(all_results)
    results = results.sort_values("Similarity", ascending=False).drop_duplicates(subset=["Drug"]).head(top_k)
    
    # Normalize drug names for merging
    results["Drug_normalized"] = results["Drug"].astype(str).str.strip().str.lower()
    
    # Merge with CIViC evidence
    if not civic_df.empty and "drug" in civic_df.columns:
        civic_merge = civic_df.copy()
        civic_merge["drug_normalized"] = civic_merge["drug"].astype(str).str.strip().str.lower()
        
        # Add line of therapy filtering if available in CIViC data
        if "line_of_therapy" in civic_merge.columns:
            civic_merge = civic_merge[
                (civic_merge["line_of_therapy"] == line_of_therapy) | 
                (civic_merge["line_of_therapy"].isna())
            ]
        
        civic_cols = ["drug_normalized"]
        for col in ["evidence_level", "rationale_text", "source_url", "approval_status"]:
            if col in civic_merge.columns:
                civic_cols.append(col)
        
        results = results.merge(
            civic_merge[civic_cols],
            how="left",
            left_on="Drug_normalized",
            right_on="drug_normalized"
        )
        results.drop(columns=["drug_normalized"], errors="ignore", inplace=True)
    
    # Merge with DepMap data
    if not depmap_df.empty and "CompoundName" in depmap_df.columns:
        depmap_merge = depmap_df.copy()
        depmap_merge["drug_normalized"] = depmap_merge["CompoundName"].astype(str).str.strip().str.lower()
        
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
    
    # Determine on-label vs off-label
    if "approval_status" in results.columns:
        results["Label Status"] = results["approval_status"].apply(
            lambda x: "✅ On-Label" if pd.notna(x) and "approved" in str(x).lower() else "⚠️ Off-Label"
        )
    elif "evidence_level" in results.columns:
        results["Label Status"] = results["evidence_level"].apply(
            lambda x: "✅ On-Label" if pd.notna(x) and str(x) in ["A", "B"] else "⚠️ Off-Label"
        )
    else:
        results["Label Status"] = "⚠️ Off-Label"
    
    # Sort: on-label first, then by similarity
    results["sort_key"] = results["Label Status"].apply(lambda x: 0 if "On-Label" in x else 1)
    results = results.sort_values(["sort_key", "Similarity"], ascending=[True, False]).drop(columns=["sort_key"])
    
    # Generate LLM rationales if enabled
    if use_llm_rationales and api_key:
        st.info("🤖 Generating AI-powered clinical rationales...")
        rationales = []
        for _, row in results.iterrows():
            rationale = generate_rationale(
                drug=row["Drug"],
                biomarkers=positive_biomarkers,
                evidence_level=row.get("evidence_level", None),
                mechanism=row.get("TargetOrMechanism", None),
                targets=row.get("GeneSymbolOfTargets", None),
                api_key=api_key
            )
            rationales.append(rationale if rationale else row.get("rationale_text", ""))
        results["AI Rationale"] = rationales
    
    # Drop the normalized column
    results.drop(columns=["Drug_normalized"], errors="ignore", inplace=True)
    
    # Rename columns
    rename_map = {
        "evidence_level": "Evidence Level",
        "rationale_text": "CIViC Rationale",
        "source_url": "Source",
        "GeneSymbolOfTargets": "Targets",
        "TargetOrMechanism": "Mechanism",
        "ChEMBLID": "ChEMBL ID",
        "Cardiotoxicity Binary Rating": "Cardiotoxicity",
        "DermatologicalToxicity Binary Rating": "Dermatotoxicity"
    }
    
    results.rename(columns={k: v for k, v in rename_map.items() if k in results.columns}, inplace=True)
    
    return results

def recommend_therapies_simple(gene, variant, top_k=5, api_key=None, use_llm_rationales=False):
    """Simple mode: just gene and variant"""
    patient_profile = {
        "diagnosis": "NSCLC",
        "line_of_therapy": "1L",
        "biomarkers": {gene: variant}
    }
    return recommend_therapies_clinical(patient_profile, top_k, api_key, use_llm_rationales)

# ============================================================
# 4️⃣  STREAMLIT UI
# ============================================================

st.title("🧬 TherapyMatcher — Precision Oncology Recommender")
st.markdown("""
Use biological embeddings + CIViC evidence + DepMap drug sensitivity to identify therapy matches
for specific **gene variants** in lung and related cancers.
""")

# Input mode selector
input_mode = st.radio(
    "Select Input Mode",
    ["🔹 Simple (Gene + Variant)", "🔸 Clinical Profile (JSON)"],
    help="Simple mode for quick queries, Clinical Profile for comprehensive patient data"
)

if input_mode == "🔹 Simple (Gene + Variant)":
    col1, col2 = st.columns(2)
    gene = col1.text_input("Gene symbol", "EGFR")
    variant = col2.text_input("Variant", "L858R")
    top_k = st.slider("Top therapies to display", 3, 10, 5)
    
    if st.button("🔍 Recommend Therapies"):
        with st.spinner("Analyzing embeddings and evidence..."):
            results = recommend_therapies_simple(gene, variant, top_k, openai_api_key, use_llm)
        
        if results is not None and not results.empty:
            st.success(f"✅ Recommendations for {gene}_{variant}")
            display_results(results, f"{gene}_{variant}")
        else:
            st.error("❌ No matching drugs found for this variant.")

else:  # Clinical Profile JSON mode
    st.markdown("### 📋 Patient Clinical Profile")
    
    default_json = """{
  "diagnosis": "Metastatic NSCLC",
  "line_of_therapy": "1L",
  "biomarkers": {
    "EGFR": "L858R",
    "ALK": false,
    "KRAS": "G12C",
    "PD-L1": ">=50%"
  },
  "comorbidities": ["no interstitial lung disease"],
  "prior_therapies": []
}"""
    
    json_input = st.text_area(
        "Paste patient profile (JSON format)",
        value=default_json,
        height=250,
        help="Enter patient biomarkers, diagnosis, line of therapy, and clinical context"
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k_json = st.slider("Top therapies to display", 3, 15, 6, key="json_slider")
    
    if st.button("🔍 Match Therapies", key="json_match"):
        try:
            patient_profile = json.loads(json_input)
            
            # Display parsed profile
            with st.expander("📊 Parsed Patient Profile"):
                st.json(patient_profile)
            
            with st.spinner("Analyzing clinical profile and generating recommendations..."):
                results = recommend_therapies_clinical(
                    patient_profile, 
                    top_k_json, 
                    openai_api_key, 
                    use_llm
                )
            
            if results is not None and not results.empty:
                diagnosis = patient_profile.get("diagnosis", "Patient")
                line = patient_profile.get("line_of_therapy", "")
                st.success(f"✅ Recommendations for {diagnosis} ({line})")
                
                # Display biomarker summary
                biomarkers = patient_profile.get("biomarkers", {})
                positive = [f"{k}: {v}" for k, v in biomarkers.items() if v and v != "false" and v != False]
                st.info(f"🧬 **Actionable Biomarkers:** {', '.join(positive)}")
                
                display_results(results, f"{diagnosis}_{line}")
            else:
                st.error("❌ No matching therapies found for this profile.")
        
        except json.JSONDecodeError as e:
            st.error(f"❌ Invalid JSON format: {e}")
        except Exception as e:
            st.error(f"❌ Error processing profile: {e}")

# ============================================================
# 5️⃣  RESULTS DISPLAY FUNCTION
# ============================================================

def display_results(results, query_name):
    """Display results with enhanced formatting"""
    
    # Reorder columns for better display
    display_cols = ["Label Status", "Drug", "Similarity", "Biomarker"]
    if "Evidence Level" in results.columns:
        display_cols.append("Evidence Level")
    if "AI Rationale" in results.columns:
        display_cols.append("AI Rationale")
    elif "CIViC Rationale" in results.columns:
        display_cols.append("CIViC Rationale")
    if "Mechanism" in results.columns:
        display_cols.append("Mechanism")
    if "Targets" in results.columns:
        display_cols.append("Targets")
    if "Cardiotoxicity" in results.columns:
        display_cols.append("Cardiotoxicity")
    if "Dermatotoxicity" in results.columns:
        display_cols.append("Dermatotoxicity")
    if "Source" in results.columns:
        display_cols.append("Source")
    
    # Filter to only existing columns
    display_cols = [col for col in display_cols if col in results.columns]
    results_display = results[display_cols]
    
    # Display with custom column config
    st.dataframe(
        results_display,
        use_container_width=True,
        column_config={
            "Label Status": st.column_config.TextColumn("Status", width="small"),
            "Drug": st.column_config.TextColumn("Drug", width="medium"),
            "Similarity": st.column_config.NumberColumn("Similarity", format="%.3f", width="small"),
            "Evidence Level": st.column_config.TextColumn("Evidence", width="small"),
            "AI Rationale": st.column_config.TextColumn("AI-Generated Rationale", width="large"),
            "CIViC Rationale": st.column_config.TextColumn("Clinical Rationale", width="large"),
            "Targets": st.column_config.TextColumn("Targets", width="medium"),
            "Mechanism": st.column_config.TextColumn("Mechanism", width="medium"),
            "Source": st.column_config.LinkColumn("Source", width="small"),
        },
        hide_index=True
    )
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Drugs Found", len(results))
    with col2:
        avg_sim = results["Similarity"].mean()
        st.metric("Avg Similarity", f"{avg_sim:.3f}")
    with col3:
        on_label = len(results[results["Label Status"].str.contains("On-Label", na=False)])
        st.metric("On-Label Options", on_label)
    with col4:
        if "Evidence Level" in results.columns:
            high_evidence = len(results[results["Evidence Level"].isin(["A", "B"])])
            st.metric("High Evidence", high_evidence)
    
    # CSV Download
    csv = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Results as CSV",
        csv,
        f"{query_name}_recommendations.csv",
        "text/csv",
        key=f"download-{query_name}"
    )

# ============================================================
# 6️⃣  EMBEDDING VISUALIZATION
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
# 7️⃣  BATCH QUERY
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
                res = recommend_therapies_simple(g, v, batch_top_k, openai_api_key, False)  # Disable LLM for batch
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

# ============================================================
# 8️⃣  FOOTER
# ============================================================

st.markdown("---")
st.caption("🔬 Powered by CIViC • DepMap • Node2Vec • Streamlit • FDA Toxicity Data • OpenAI GPT-4")

with st.expander("📖 About TherapyMatcher"):
    st.markdown("""
    **TherapyMatcher** leverages graph neural networks and clinical evidence to recommend precision oncology therapies.
    
    **Data Sources:**
    - **CIViC**: Clinical interpretations of variants in cancer
    - **DepMap**: Drug sensitivity and target mechanism data
    - **FDA**: Toxicity profiles for safety assessment
    - **Node2Vec**: Graph embeddings capturing biological relationships
    
    **Features:**
    - ✅ Simple gene+variant queries or comprehensive JSON patient profiles
    - ✅ On-label vs off-label therapy identification
    - ✅ Evidence-level stratification (A/B = high confidence)
    - ✅ Optional AI-generated clinical rationales (GPT-4)
    - ✅ Multi-biomarker support for complex cases
    - ✅ Toxicity assessment (cardio, dermatological)
    
    **How it works:**
    1. Enter patient biomarkers (simple or JSON format)
    2. System computes similarity in embedding space
    3. Results ranked by evidence + similarity
    4. Optional: GPT-4 generates clinical explanations
    
    **Limitations:**
    - Computational predictions, not clinical advice
    - Validate with current literature and guidelines
    - Evidence may have changed since data collection
    - Requires OpenAI API key for LLM rationales
    
    **Developed for:** Evolved Boston 2025 Hackathon | Team 14: AI for Precision Medicine
    """)

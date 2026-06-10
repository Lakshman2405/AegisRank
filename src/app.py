# src/app.py
import os
import sys
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agents.retrieval_agent import IntelligentRetrievalEngine

st.set_page_config(page_title="AegisRank Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ AegisRank: Real-Time Multi-Agent Talent Verification System")

clean_dataset_cache = os.path.join("data", "clean_candidates.jsonl")

# Open src/app.py and locate lines 16-24
if "retrieval_engine_instance" not in st.session_state:
    if not os.path.exists(clean_dataset_cache):
        st.error(f"Missing dependency: '{clean_dataset_cache}' not found. Please run your file ingestion streaming scripts first.")
        st.stop()
        
    with st.spinner("🚀 Connecting Agent Mesh & Syncing Cloud Clusters..."):
        engine = IntelligentRetrievalEngine()
        # FIX: Lower batch size to 200 to prevent network write timeouts
        engine.synchronize_vector_repository(
            clean_cache_path=clean_dataset_cache, 
            indexing_batch_limit=200
        )
        st.session_state["retrieval_engine_instance"] = engine

    st.success("⚡ Cloud Vector Cluster fully synchronized with the 96k payload!")

active_engine = st.session_state["retrieval_engine_instance"]

target_jd_input = st.text_area(
    "🔍 Job Description Query Matrix (Inputs matching challenge specs)",
    value="Senior AI Engineer with 5-9 years experience, expert in semantic search retrieval, vector databases like Qdrant, and engineering offline evaluation frameworks...",
    height=150
)

if st.button("🚀 Execute Agentic Pipeline & Compile Submission Pack", use_container_width=True):
    with st.spinner("🤖 Invoking Agents: Scanning Qdrant Index & Synthesizing Target Reasonings..."):
        results = active_engine.run_agentic_ranking_pipeline(target_jd=target_jd_input)
        
    st.success("🎉 Processing Complete! submission.csv and submission_metadata.yaml compiled!")
    if st.button("Download Submission Pack"):
        with open("submission/submission.csv", "rb") as f:
            st.download_button("Download CSV", f, "submission.csv")
        with open("submission/submission_metadata.yaml", "rb") as f:
            st.download_button("Download YAML", f, "submission_metadata.yaml")
    import pandas as pd
    st.markdown("### 📊 Top Ranked Candidates (Groq Agent Reasonings Embedded)")
    st.dataframe(pd.DataFrame(results), use_container_width=True)
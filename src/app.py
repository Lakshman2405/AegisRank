# src/app.py
import os
import sys
import streamlit as st
import pandas as pd
import importlib  # Add this built-in module
from dotenv import load_dotenv

load_dotenv()
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Force Python to reload the module freshly on execution
import agents.retrieval_agent
importlib.reload(agents.retrieval_agent)
from agents.retrieval_agent import IntelligentRetrievalEngine

st.set_page_config(page_title="AegisRank Dashboard", page_icon="🛡️", layout="wide")
st.title("🛡️ AegisRank: Real-Time Multi-Agent Talent Verification System")

clean_dataset_cache = os.path.join("data", "clean_candidates.jsonl")

# --- INITIALIZE AND CACHE ENGINE SINGLETON ---
if "retrieval_engine_instance" not in st.session_state:
    if not os.path.exists(clean_dataset_cache):
        st.error(f"Missing dependency: '{clean_dataset_cache}' not found. Please run your file ingestion streaming scripts first.")
        st.stop()
        
    with st.spinner("🚀 Connecting Agent Mesh & Syncing Cloud Clusters..."):
        engine = IntelligentRetrievalEngine()
        engine.synchronize_vector_repository(
            clean_cache_path=clean_dataset_cache, 
            indexing_batch_limit= 1000
        )
        st.session_state["retrieval_engine_instance"] = engine
    st.success("⚡ Cloud Vector Cluster fully synchronized with the 96k payload!")

active_engine = st.session_state["retrieval_engine_instance"]

# --- APP WORKSPACE STATE TRACKING ---
if "pipeline_executed" not in st.session_state:
    st.session_state["pipeline_executed"] = False
if "cached_results" not in st.session_state:
    st.session_state["cached_results"] = None

target_jd_input = st.text_area(
    "🔍 Job Description Query Matrix (Inputs matching challenge specs)",
    value="Senior AI Engineer with 5-9 years experience, expert in semantic search retrieval, vector databases like Qdrant, and engineering offline evaluation frameworks...",
    height=150
)

# --- EXECUTION TRIGGER ---
if st.button("🚀 Execute Agentic Pipeline & Compile Submission Pack", use_container_width=True):
    with st.spinner("🤖 Invoking Agents: Scanning Qdrant Index & Synthesizing Target Reasonings..."):
        results = active_engine.run_agentic_ranking_pipeline(target_jd=target_jd_input)
        st.session_state["cached_results"] = results
        st.session_state["pipeline_executed"] = True

# --- RENDER RESULTS & ARTIFACT DOWNLOAD LAYERS ---
if st.session_state["pipeline_executed"] and st.session_state["cached_results"] is not None:
    st.success("🎉 Processing Complete! Submission artifacts compiled cleanly into your local directories.")
    
    # Target files mapped exactly to your retrieval backend names
    csv_file_path = os.path.join("submission", "team_6a26b0c93c432ee4828a149d.csv")
    yaml_file_path = os.path.join("submission", "submission_metadata.yaml")
    
    # Non-nested download layout row
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists(csv_file_path):
            with open(csv_file_path, "rb") as f:
                st.download_button(
                    label="📥 Download Submission CSV", 
                    data=f, 
                    file_name="team_6a26b0c93c432ee4828a149d.csv", 
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.error("Submission CSV file not found at local target path.")
            
    with col2:
        if os.path.exists(yaml_file_path):
            with open(yaml_file_path, "rb") as f:
                st.download_button(
                    label="📄 Download Metadata YAML", 
                    data=f, 
                    file_name="submission_metadata.yaml", 
                    mime="text/yaml",
                    use_container_width=True
                )
        else:
            st.error("Submission Metadata YAML file not found at local target path.")
            
    # Interactive Dataframe Inspection View
    st.markdown("---")
    st.markdown("### 📊 Top Ranked Candidates (Factual Evaluation Profiles)")
    display_df = pd.DataFrame(st.session_state["cached_results"])
    st.dataframe(display_df, use_container_width=True)
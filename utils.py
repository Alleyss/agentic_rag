# utils.py
import streamlit as st
import os
from dotenv import load_dotenv
import re

def get_config():
    """Loads API keys and Pinecone serverless config ONLY from the .env file."""
    print("Attempting to load configuration from .env file...")
    # Use find_dotenv to search in parent directories as well
    env_path = '.env'
    if env_path:
        print(f"Loading .env file from: {env_path}")
        loaded = load_dotenv(dotenv_path=env_path)
    else:
        print("Warning: .env file not found in current or parent directories.")
        loaded = False


    # loaded = load_dotenv() # Simpler version if .env is always in root

    if not loaded:
        st.warning("Config Warning: Could not find/load .env file.", icon="⚠️")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    pinecone_region = os.getenv("PINECONE_REGION")
    pinecone_cloud = os.getenv("PINECONE_CLOUD")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    print(f"PINECONE_API_KEY loaded: {'Yes' if pinecone_api_key else 'No'}")
    print(f"PINECONE_REGION loaded: {'Yes' if pinecone_region else 'No'}")
    print(f"PINECONE_CLOUD loaded: {'Yes' if pinecone_cloud else 'No'}")
    print(f"GOOGLE_API_KEY loaded: {'Yes' if google_api_key else 'No'}")

    required_keys = {
        "PINECONE_API_KEY": pinecone_api_key, "PINECONE_REGION": pinecone_region,
        "PINECONE_CLOUD": pinecone_cloud, "GOOGLE_API_KEY": google_api_key
    }
    missing_keys = [name for name, value in required_keys.items() if not value]
    if missing_keys:
         st.error(f"🔴 Config Error: Missing in .env: {', '.join(missing_keys)}.", icon="🚨")
         # Consider stopping execution if keys are absolutely mandatory
         # st.stop()

    return {
        "pinecone_api_key": pinecone_api_key, "pinecone_region": pinecone_region,
        "pinecone_cloud": pinecone_cloud, "google_api_key": google_api_key,
    }

# --- CHANGED: Updated regex for simple numeric citations like [1], [2] ---
def extract_citations_from_answer(answer_text):
    """Finds simple numeric citation labels (e.g., [1], [23]) in the text."""
    citation_pattern = r"(\[\d+\])" # Matches one or more digits inside square brackets
    citations = re.findall(citation_pattern, answer_text)
    unique_citations = []
    for cit in citations:
        if cit not in unique_citations:
            unique_citations.append(cit)
    return unique_citations

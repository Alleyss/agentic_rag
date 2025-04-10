# utils.py
import streamlit as st
import os
from dotenv import load_dotenv
import re

def get_config():
    """Loads API keys and Pinecone serverless config ONLY from the .env file."""
    print("Attempting to load configuration from .env file...")
    loaded = load_dotenv()

    if not loaded:
        print("Warning: .env file not found or failed to load.")
        st.warning("Configuration Error: Could not find or load the .env file.", icon="⚠️")

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    # pinecone_environment = os.getenv("PINECONE_ENVIRONMENT") # Keep if used elsewhere?
    pinecone_region = os.getenv("PINECONE_REGION") # Load region
    pinecone_cloud = os.getenv("PINECONE_CLOUD")   # Load cloud
    google_api_key = os.getenv("GOOGLE_API_KEY")

    print(f"PINECONE_API_KEY loaded: {'Yes' if pinecone_api_key else 'No'}")
    print(f"PINECONE_REGION loaded: {'Yes' if pinecone_region else 'No'}")
    print(f"PINECONE_CLOUD loaded: {'Yes' if pinecone_cloud else 'No'}")
    print(f"GOOGLE_API_KEY loaded: {'Yes' if google_api_key else 'No'}")

    # Error checks for essential keys
    required_keys = {
        "PINECONE_API_KEY": pinecone_api_key,
        "PINECONE_REGION": pinecone_region,
        "PINECONE_CLOUD": pinecone_cloud,
        "GOOGLE_API_KEY": google_api_key
    }
    missing_keys = [name for name, value in required_keys.items() if not value]
    if missing_keys:
         st.error(f"🔴 Critical Error: Required config variables not found in .env: {', '.join(missing_keys)}.", icon="🚨")

    return {
        "pinecone_api_key": pinecone_api_key,
        # "pinecone_environment": pinecone_environment, # Return if needed elsewhere
        "pinecone_region": pinecone_region,
        "pinecone_cloud": pinecone_cloud,
        "google_api_key": google_api_key,
    }

def extract_citations_from_answer(answer_text):
    # ... (function remains the same) ...
    citation_pattern = r"(\[(?:PDF Page|Image Page|Web):?\s*[^\]]+\])"
    citations = re.findall( citation_pattern, answer_text)
    unique_citations = []
    for cit in citations:
        if cit not in unique_citations:
            unique_citations.append(cit)
    return unique_citations
# vector_store.py
import os
import time
import hashlib
import streamlit as st
# --- CHANGE: Import Pinecone and ServerlessSpec ---
from pinecone import Pinecone, ServerlessSpec
# -------------------------------------
import google.generativeai as genai

# --- Embedding Model (Google) ---
embedding_model_name = 'models/embedding-001'
embedding_dim = 768 # Dimension for 'models/embedding-001'

# ... (get_embedding function remains the same) ...
def get_embedding(text):
    """Generates embedding for given text using Google GenAI."""
    if not text or not isinstance(text, str):
        print(f"Warning: Invalid text provided for embedding: {text}")
        return None
    if not genai.get_key():
         print("Error: Google API Key not configured for embedding generation.")
         return None
    try:
        result = genai.embed_content(
            model=embedding_model_name,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error generating embedding for text chunk: '{text[:50]}...' - {e}")
        return None


# --- Pinecone Setup (Updated for Serverless Creation) ---
def init_pinecone(api_key: str, index_name: str, dimension: int, region: str, cloud: str = 'aws', metric: str = 'cosine'):
    """
    Initializes Pinecone connection for a Serverless index,
    creating it if it doesn't exist.

    Args:
        api_key: Your Pinecone API key.
        index_name: The name of the serverless index.
        dimension: The vector dimension for the index.
        region: The cloud region (e.g., 'us-east-1').
        cloud: The cloud provider ('aws', 'gcp', 'azure'). Defaults to 'aws'.
        metric: The distance metric ('cosine', 'euclidean', 'dotproduct'). Defaults to 'cosine'.

    Returns:
        A Pinecone Index object or None if initialization fails.
    """
    if not all([api_key, index_name, dimension, region, cloud, metric]):
        st.error("Missing required arguments for Pinecone initialization (API Key, Index Name, Dimension, Region, Cloud, Metric).")
        print("Error: Missing required arguments for init_pinecone.")
        return None

    try:
        print(f"Initializing Pinecone client...")
        # --- Initialize client - environment not typically needed here for serverless focus ---
        # The API key identifies your project which is tied to resources.
        pc = Pinecone(api_key=api_key)
        print("Pinecone client initialized.")

        existing_indexes = pc.list_indexes().names
        print(f"Existing indexes found: {existing_indexes}")

        if index_name not in existing_indexes:
            print(f"Index '{index_name}' not found. Creating a new serverless index...")
            try:
                # --- Define the serverless spec ---
                spec = ServerlessSpec(cloud=cloud, region=region)
                # ---------------------------------
                print(f"Attempting to create index '{index_name}' with spec: {spec}, dimension: {dimension}, metric: {metric}")
                pc.create_index(
                    name=index_name,
                    dimension=dimension,
                    metric=metric,
                    spec=spec # Pass the ServerlessSpec here
                )
                # --- Wait for index readiness ---
                wait_time = 5
                max_wait = 300 # Wait max 5 minutes
                current_wait = 0
                print(f"Waiting for index '{index_name}' to be ready (up to {max_wait} seconds)...")
                while current_wait < max_wait:
                    try:
                        index_status = pc.describe_index(index_name).status
                        if index_status['ready']:
                             print(f"Index '{index_name}' created and ready.")
                             break # Exit loop if ready
                        else:
                             state = index_status.get('state', 'Unknown')
                             print(f"  Index state: {state}. Waiting {wait_time}s...")
                    except Exception as desc_wait_e:
                         # Handle cases where describe_index might fail transiently during creation
                         print(f"  Waiting... (Error describing index during creation: {desc_wait_e})")

                    time.sleep(wait_time)
                    current_wait += wait_time
                else: # Loop finished without breaking (timeout)
                    error_msg = f"Index '{index_name}' did not become ready after {max_wait} seconds."
                    print(error_msg)
                    st.error(error_msg)
                    # Consider attempting deletion? pc.delete_index(index_name)
                    return None
                # ---------------------------------
            except Exception as create_e:
                error_msg = f"Error creating Pinecone serverless index '{index_name}': {create_e}"
                print(error_msg)
                st.error(error_msg)
                return None # Stop if index creation fails

        else:
            # --- Index exists, verify configuration ---
            print(f"Index '{index_name}' already exists. Verifying configuration...")
            try:
                index_description = pc.describe_index(index_name)

                # Verify Dimension
                existing_dimension = index_description.dimension
                if existing_dimension != dimension:
                     error_msg = f"Dimension mismatch! Existing index '{index_name}' has dimension {existing_dimension}, expected {dimension}."
                     st.error(error_msg)
                     print(f"CRITICAL ERROR: {error_msg}")
                     return None

                # Verify Metric
                existing_metric = index_description.metric
                if existing_metric != metric:
                    # This might be acceptable, but log a warning
                    print(f"Warning: Metric mismatch for index '{index_name}'. Found '{existing_metric}', expected '{metric}'.")
                    st.warning(f"Metric mismatch for index '{index_name}'. Found '{existing_metric}', expected '{metric}'. Queries might yield unexpected results.")

                # Verify Type (Serverless) and Region/Cloud
                if hasattr(index_description, 'spec') and hasattr(index_description.spec, 'serverless'):
                    existing_cloud = index_description.spec.serverless.get('cloud')
                    existing_region = index_description.spec.serverless.get('region')
                    if existing_cloud != cloud or existing_region != region:
                        error_msg = (f"Configuration mismatch! Existing index '{index_name}' is serverless "
                                     f"but in {existing_cloud}/{existing_region}. Expected {cloud}/{region}.")
                        st.error(error_msg)
                        print(f"CRITICAL ERROR: {error_msg}")
                        return None
                    print(f"Index '{index_name}' confirmed as serverless in {cloud}/{region} with correct dimension.")
                else:
                    # Index exists but is not serverless
                    error_msg = f"Configuration mismatch! Index '{index_name}' exists but is not a serverless index."
                    st.error(error_msg)
                    print(f"CRITICAL ERROR: {error_msg}")
                    return None # Index exists but is the wrong type (e.g., pod-based)

            except Exception as desc_e:
                 error_msg = f"Error describing existing Pinecone index '{index_name}': {desc_e}"
                 print(error_msg)
                 st.error(error_msg)
                 return None # Cannot proceed if we can't verify the index

        # --- Get the Index object ---
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"Successfully connected to index '{index_name}'. Stats: {stats}")
        return index # Return the index object

    except Exception as e:
        error_msg = f"Error during Pinecone client initialization or index handling: {e}"
        print(error_msg)
        if "api key" in str(e).lower():
             st.error(f"Pinecone Authentication Error: Invalid API Key.")
        else:
             st.error(f"An error occurred with Pinecone setup: {e}")
        return None

# ... (create_unique_id, upsert_data_to_pinecone, query_pinecone functions remain the same) ...
def create_unique_id(doc_hash, item_type, page_num, index):
    """Creates a unique and deterministic ID for Pinecone vectors."""
    base_id = f"{doc_hash}-{item_type}-p{page_num}-{index}"
    return base_id

def upsert_data_to_pinecone(index, data_list, doc_hash, pdf_filename):
    """Embeds and upserts text chunks and image descriptions to Pinecone."""
    if not genai.get_key():
         st.error("Google API Key not configured. Cannot proceed with embedding.")
         print("Error: Google API Key not configured before upsert process.")
         return # Stop the upsert process

    vectors_to_upsert = []
    batch_size = 100 # Pinecone recommends batching upserts
    item_count = len(data_list)
    print(f"Starting upsert process for {item_count} items...")
    progress_bar = st.progress(0) # Add progress bar in Streamlit if possible
    status_text = st.empty()

    for i, item in enumerate(data_list):
        status_text.text(f"Processing item {i+1}/{item_count} for embedding...")
        item_type = item['type']
        page_num = item['page_number']
        content_to_embed = None
        metadata = {
            'source_type': 'pdf', 'document_hash': doc_hash, 'pdf_filename': pdf_filename,
            'page_number': page_num,
        }
        if item_type == 'text':
            content_to_embed = item['content']
            metadata['content_type'] = 'text'; metadata['text'] = content_to_embed
            item_index = item.get('chunk_index', 0)
            unique_id = create_unique_id(doc_hash, 'text', page_num, item_index)
        elif item_type == 'image_description':
            content_to_embed = item['description']
            metadata['content_type'] = 'image'; metadata['image_description'] = content_to_embed
            metadata['image_path'] = item['image_path']
            item_index = item.get('img_index', 0)
            unique_id = create_unique_id(doc_hash, 'image', page_num, item_index)
        else: continue

        if content_to_embed:
            embedding = get_embedding(content_to_embed)
            if embedding:
                vectors_to_upsert.append({ 'id': unique_id, 'values': embedding, 'metadata': metadata })
            else: print(f"Skipping item due to embedding failure: Type={item_type}, Page={page_num}, Index={item_index}")

        if len(vectors_to_upsert) >= batch_size:
            print(f"Upserting batch of {len(vectors_to_upsert)} vectors...")
            try: index.upsert(vectors=vectors_to_upsert); vectors_to_upsert = []
            except Exception as e: print(f"Error upserting batch: {e}"); st.warning(f"Error upserting batch: {e}."); time.sleep(2)
        progress_bar.progress((i + 1) / item_count)

    if vectors_to_upsert:
        status_text.text(f"Upserting final batch of {len(vectors_to_upsert)} vectors...")
        print(f"Upserting final batch of {len(vectors_to_upsert)} vectors...")
        try: index.upsert(vectors=vectors_to_upsert)
        except Exception as e: print(f"Error upserting final batch: {e}"); st.warning(f"Error upserting final batch: {e}.")

    progress_bar.empty(); status_text.text("Upsert process complete.")
    print("Finished upserting data to Pinecone.")
    stats = index.describe_index_stats()
    print(f"Pinecone index stats after upsert: {stats}")

def query_pinecone(index, query_text, top_k=5, doc_hash=None):
    """Queries Pinecone for relevant documents."""
    if not genai.get_key():
         print("Error: Google API Key not configured for query embedding generation.")
         st.error("Google API Key not configured. Cannot perform search.")
         return []
    try:
        result = genai.embed_content(model=embedding_model_name, content=query_text, task_type="retrieval_query")
        query_embedding = result['embedding']
    except Exception as e:
        print(f"Error generating query embedding: {e}"); st.error(f"Failed to generate query embedding: {e}"); return []
    if query_embedding is None: return []

    filter_dict = {}
    if doc_hash: filter_dict = {"document_hash": {"$eq": doc_hash}}; print(f"Querying Pinecone with filter for doc_hash: {doc_hash}")
    else: print("Querying Pinecone without document filter.")

    try:
        results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=filter_dict if filter_dict else None)
        if not results or not results.get('matches'): print("Query successful, but no matches found."); return []
        print(f"Found {len(results['matches'])} matches in Pinecone.")
        return results['matches']
    except Exception as e:
        print(f"Error querying Pinecone: {e}"); st.error(f"Error querying Pinecone: {e}"); return []
# vector_store.py
import os
import time
import hashlib
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
# Removed traceback import

# --- Embedding Model (Google) ---
embedding_model_name = 'models/embedding-001'
embedding_dim = 768
MAX_CHUNK_LENGTH_CHARS = 5000
CHUNK_OVERLAP = 100
# --- Text Chunking Helper ---
def chunk_text(text: str, max_length: int = MAX_CHUNK_LENGTH_CHARS, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Splits text into chunks based on character length with overlap."""
    if not isinstance(text, str) or len(text) <= max_length:
        return [text] if text else []

    chunks = []
    start_index = 0
    while start_index < len(text):
        end_index = min(start_index + max_length, len(text))
        chunks.append(text[start_index:end_index])
        start_index += max_length - overlap # Move start index back by overlap
        if start_index >= len(text): # Break if overlap pushes us past the end
            break
        # Ensure start_index doesn't become negative in edge cases
        start_index = max(0, start_index)
    return chunks
# --------------------------


# --- Cleaned get_embedding function (no changes needed here) ---
def get_embedding(text):
    """Generates embedding for given text using Google GenAI."""
    if not text or not isinstance(text, str):
        # Log minimal warning if needed, avoid excessive printing
        # print(f"Warning (get_embedding): Invalid text type ({type(text)}).")
        return None

    # Relies on global genai.configure() from app.py
    try:
        result = genai.embed_content(
            model=embedding_model_name,
            content=text,
            task_type="retrieval_document"
        )
        if 'embedding' in result:
            return result['embedding']
        else:
            print(f"ERROR (get_embedding): 'embedding' key not found in Google API response.")
            # print(f"Full Response: {result}") # Keep commented for brevity unless needed
            return None
    except Exception as e:
        print(f"ERROR (get_embedding): Exception during embed_content: {e}")
        # Consider logging exception type: print(f"Exception Type: {type(e).__name__}")
        st.toast(f"Error generating embedding: {e}", icon="⚠️") # Inform user via toast
        return None
# -----------------------------------------

# --- Pinecone Setup (No changes from previous correct version) ---
def init_pinecone(api_key: str, index_name: str, dimension: int, region: str, cloud: str = 'aws', metric: str = 'cosine'):
    """Initializes Pinecone connection for a Serverless index, creating it if it doesn't exist."""
    if not all([api_key, index_name, dimension, region, cloud, metric]):
        st.error("Missing required arguments for Pinecone initialization.")
        return None
    try:
        print(f"Initializing Pinecone client...") # Keep essential logs
        pc = Pinecone(api_key=api_key)
        print("Pinecone client initialized.")
        try:
            existing_indexes = [index.name for index in pc.list_indexes()]
            print(f"Existing indexes found: {existing_indexes}")
        except Exception as list_e:
            print(f"Warning: Error listing indexes: {list_e}"); existing_indexes = []

        if index_name not in existing_indexes:
            print(f"Index '{index_name}' not found. Creating new serverless index...")
            try:
                spec = ServerlessSpec(cloud=cloud, region=region)
                print(f"Attempting creation: spec={spec}, dim={dimension}, metric={metric}")
                pc.create_index(name=index_name, dimension=dimension, metric=metric, spec=spec)
                wait_time, max_wait, current_wait = 10, 30, 0 # Increased wait time slightly
                print(f"Waiting up to {max_wait}s for index readiness...")
                while current_wait < max_wait:
                    try:
                        index_desc = pc.describe_index(index_name)
                        if hasattr(index_desc, 'status') and isinstance(index_desc.status, dict):
                            is_ready = index_desc.status.get('ready', False); state = index_desc.status.get('state', 'Unknown')
                        else: is_ready = getattr(index_desc, 'ready', False); state = getattr(index_desc, 'state', 'Unknown')

                        if is_ready: print(f"Index '{index_name}' ready."); break
                        else: print(f"  Index state: {state}. Waiting {wait_time}s...")
                    except Exception as desc_wait_e: print(f"  Waiting... (Error checking status: {desc_wait_e})")
                    time.sleep(wait_time); current_wait += wait_time
                else: print(f"Index '{index_name}' timeout."); st.error(f"Index creation timeout."); return None
            except Exception as create_e: print(f"Error creating index: {create_e}"); st.error(f"Error creating index: {create_e}"); return None
        else:
            print(f"Index '{index_name}' exists. Verifying config...")
            try: # Verification logic (seems okay)
                index_desc = pc.describe_index(index_name)
                exist_dim = getattr(index_desc, 'dimension', None)
                if exist_dim != dimension: raise ValueError(f"Dimension mismatch! Existing={exist_dim}, Expected={dimension}.")
                exist_metric = getattr(index_desc, 'metric', None)
                if exist_metric != metric: print(f"Warning: Metric mismatch. Found={exist_metric}, Expected={metric}.")
                if hasattr(index_desc, 'spec') and hasattr(index_desc.spec, 'serverless'):
                    exist_cloud = getattr(index_desc.spec.serverless, 'cloud', None); exist_region = getattr(index_desc.spec.serverless, 'region', None)
                    if exist_cloud != cloud or exist_region != region: raise ValueError(f"Config mismatch! Found {exist_cloud}/{exist_region}, Expected {cloud}/{region}.")
                    print(f"Config OK: Serverless {cloud}/{region}, Dim={dimension}.")
                else: raise TypeError(f"Index '{index_name}' is not a serverless index.")
            except Exception as desc_e: print(f"Error verifying index: {desc_e}"); st.error(f"Error verifying index: {desc_e}"); return None
        try: # Connect to index
            index = pc.Index(index_name)
            stats = index.describe_index_stats()
            print(f"Connected to index '{index_name}'. Stats: {stats}")
            return index
        except Exception as index_e: print(f"Error connecting to index: {index_e}"); st.error(f"Error connecting index: {index_e}"); return None
    except Exception as e: print(f"Error in Pinecone setup: {e}"); st.error(f"Pinecone setup error: {e}"); return None


def create_unique_id(doc_hash, item_type, page_num, index, chunk_num=0):
    """Creates a unique ID, adding chunk number if needed."""
    # Ensure index is treated as string if it's like 't0'
    base_id = f"{doc_hash}-{item_type}-p{page_num}-{str(index)}"
    # Append chunk number only if chunk_num > 0 (i.e., more than one chunk)
    if chunk_num > 0:
        return f"{base_id}-c{chunk_num}"
    return base_id
# --- Cleaned upsert_data_to_pinecone ---
def upsert_data_to_pinecone(index, data_list, doc_hash, pdf_filename, status_callback=None):
    """Embeds and upserts potentially chunked data to Pinecone."""
    print(f"\n--- Starting upsert ({len(data_list)} items) ---")

    vectors_to_upsert = []
    batch_size = 100
    total_processed_items = 0
    total_chunks_generated = 0
    successful_embeddings = 0
    failed_embeddings = 0
    upserted_count_total = 0

    for i, item in enumerate(data_list):
        total_processed_items += 1
        if status_callback:
            # Update status less frequently inside the item loop if chunking is fast
            # Or update based on percentage of items processed
             if i % 10 == 0: # Update every 10 items
                 status_callback(f"Preparing item {i+1}/{len(data_list)}...")

        item_type = item.get('type')
        page_num = item.get('page_number')
        original_content = None # Content before chunking
        content_type_for_id_and_meta = None # Will be 'text', 'image', or 'table'
        item_index = item.get('chunk_index', item.get('img_index', 'N/A'))

        # --- Base metadata: Store original info ---
        base_metadata = {
            'source_type': 'pdf', 'document_hash': doc_hash, 'pdf_filename': pdf_filename,
            'page_number': page_num,
            'page_screenshot_path': item.get('page_screenshot_path'),
            'linked_items': item.get('linked_items', [])
            # Add bbox only if valid
        }
        bbox_floats = item.get('bounding_box')
        if isinstance(bbox_floats, list) and len(bbox_floats) == 4:
            try: base_metadata['bounding_box'] = [str(coord) for coord in bbox_floats]
            except: pass
        # -------------------------------------------

        # --- Get original content and set correct content type ---
        if item_type == 'text':
            original_content = item.get('content')
            content_type_for_id_and_meta = 'text'
            base_metadata['content_type'] = 'text'
            base_metadata['text'] = original_content # Store original full text
        elif item_type == 'image_description':
            original_content = item.get('description')
            content_type_for_id_and_meta = 'image'
            base_metadata['content_type'] = 'image'
            base_metadata['image_description'] = original_content
            base_metadata['image_path'] = item.get('image_path')
        elif item_type == 'table':
            original_content = item.get('content') # Markdown content
            content_type_for_id_and_meta = 'table'
            base_metadata['content_type'] = 'table'
            base_metadata['table_markdown'] = original_content
        elif item_type == 'image_ref':
            continue # Skip raw image refs
        else:
            print(f"Warning (upsert): Unknown item type '{item_type}' at index {i}.")
            continue
        # ---------------------------------------------------------

        # --- Chunk the content ---
        if original_content:
            content_chunks = chunk_text(original_content) # Use the helper function
            if not content_chunks: # Handle empty string case
                failed_embeddings += 1 # Treat as failed if no chunks generated
                continue
            total_chunks_generated += len(content_chunks)

            # --- Loop through CHUNKS ---
            for chunk_idx, chunk_content in enumerate(content_chunks):
                # Generate unique ID for the chunk
                unique_id = create_unique_id(
                    doc_hash,
                    content_type_for_id_and_meta, # Use 'text', 'image', or 'table'
                    page_num,
                    item_index,
                    chunk_num=chunk_idx + 1 if len(content_chunks) > 1 else 0 # Only add chunk num if chunked
                )

                # Embed the individual chunk
                embedding = get_embedding(chunk_content)

                if embedding:
                    successful_embeddings += 1
                    # Add chunk to batch with its unique ID and COPIED metadata
                    # The metadata refers to the ORIGINAL item the chunk came from
                    vectors_to_upsert.append({
                        'id': unique_id,
                        'values': embedding,
                        'metadata': base_metadata.copy() # Use a copy!
                    })
                else:
                    failed_embeddings += 1
                    print(f"Embedding failed for chunk {chunk_idx+1}/{len(content_chunks)} of item {i} (ID: {unique_id})")
            # --- End of chunk loop ---

        else: # Original content was missing or empty
             failed_embeddings += 1

        # --- Batch upsert logic (remains the same) ---
        if len(vectors_to_upsert) >= batch_size:
            current_batch_size = len(vectors_to_upsert) # Store size before clearing
            if status_callback: status_callback(f"Upserting batch ({current_batch_size} vectors)...")
            print(f"Upserting batch ({current_batch_size} vectors)...")
            try: # Upsert batch
                upsert_response = index.upsert(vectors=vectors_to_upsert)
                count = getattr(upsert_response, 'upserted_count', 0); upserted_count_total += count
                print(f"Batch upsert successful. Count: {count}")
                if count != current_batch_size: print(f"WARNING: Batch upsert count mismatch!")
                vectors_to_upsert = [] # Clear batch
            except Exception as e: print(f"ERROR: Batch upsert failed: {e}"); st.toast(f"Pinecone Error: {e}", icon="🚨"); time.sleep(2)
        # ---------------------------------------------

    # --- Final batch upsert (remains the same) ---
    if vectors_to_upsert:
        final_batch_size = len(vectors_to_upsert)
        if status_callback: status_callback(f"Upserting final batch ({final_batch_size} vectors)...")
        print(f"Upserting final batch ({final_batch_size} vectors)...")
        try: # Upsert final batch
            upsert_response = index.upsert(vectors=vectors_to_upsert)
            count = getattr(upsert_response, 'upserted_count', 0); upserted_count_total += count
            print(f"Final batch upsert successful. Count: {count}")
            if count != final_batch_size: print(f"WARNING: Final upsert count mismatch!")
        except Exception as e: print(f"ERROR: Final batch upsert failed: {e}"); st.toast(f"Pinecone Error: {e}", icon="🚨")
    # -------------------------------------------

    print(f"\n--- Finished upsert. Summary ---")
    print(f"Total Original Items Processed: {total_processed_items}")
    print(f"Total Chunks Generated: {total_chunks_generated}")
    print(f"Successful Chunk Embeddings: {successful_embeddings}")
    print(f"Failed/Skipped Chunk Embeddings: {failed_embeddings}")
    print(f"Total Vectors Upserted (Reported by Pinecone): {upserted_count_total}")

    # Get final index stats
    try:
        stats = index.describe_index_stats()
        print(f"Final Pinecone index stats: {stats}")
    except Exception as e: print(f"Warning: Failed to get final index stats: {e}")
# --- Cleaned query_pinecone ---
def query_pinecone(index, query_text, top_k=5, doc_hash=None):
    """Queries Pinecone for relevant documents."""
    # No need for genai.configure() if done globally in app.py

    try:
        result = genai.embed_content(model=embedding_model_name, content=query_text, task_type="retrieval_query")
        if 'embedding' not in result: raise ValueError("Embedding not found in query response.")
        query_embedding = result['embedding']
    except Exception as e: print(f"Error generating query embedding: {e}"); st.error(f"Query embedding error: {e}"); return []

    filter_dict = {}
    if doc_hash: filter_dict = {"document_hash": {"$eq": doc_hash}}

    try:
        query_response = index.query(vector=query_embedding, top_k=top_k, include_metadata=True, filter=filter_dict if filter_dict else None)
        matches = getattr(query_response, 'matches', [])
        # print(f"Query successful. Matches found: {len(matches)}") # Less verbose
        return matches
    except Exception as e: print(f"Error querying Pinecone: {e}"); st.error(f"Pinecone query error: {e}"); return []
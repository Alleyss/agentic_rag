# app.py
import streamlit as st
import os
import time
import re
import google.generativeai as genai
import hashlib
import traceback # Keep for critical error logging if needed
import inspect
# --- Pillow Import ---
try:
    from PIL import Image, ImageDraw
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
# ------------------------------------

# --- Module Imports ---
from utils import get_config, extract_citations_from_answer
from pdf_processor import extract_pdf_data, CAMELOT_AVAILABLE # Import CAMELOT_AVAILABLE flag
from image_analyzer import analyze_image_with_gemini # Keep analyze_image_with_gemini
from vector_store import (
    init_pinecone,
    upsert_data_to_pinecone,
    query_pinecone,
    embedding_dim
)
from web_search import perform_web_search
from answer_generator import generate_answer_with_citations
# --------------------

# --- Configuration ---
CONFIG = get_config()
PINECONE_INDEX_NAME = "agentic-multimodal-rag2" # Consider making this configurable?
IMAGE_DIR = "images"
UPLOAD_DIR = "uploads" # Define upload dir
os.makedirs(UPLOAD_DIR, exist_ok=True) # Ensure upload dir exists
os.makedirs(IMAGE_DIR, exist_ok=True) # Ensure image dir exists
# --------------------

# --- Global Configs (API Keys etc.) ---
google_api_key = CONFIG.get("google_api_key")
pinecone_api_key=CONFIG.get("pinecone_api_key")
pinecone_region=CONFIG.get("pinecone_region")
pinecone_cloud=CONFIG.get("pinecone_cloud")

if not google_api_key:
    st.error("🔴 Error: GOOGLE_API_KEY not found in config (.env). App cannot function."); st.stop()
if not all([pinecone_api_key, pinecone_region, pinecone_cloud]):
    st.error("🔴 Error: Pinecone config (API Key, Region, Cloud) missing in .env. App cannot function."); st.stop()

try:
    genai.configure(api_key=google_api_key)
    print("Google GenAI Key configured globally.")
except Exception as e:
    st.error(f"🔴 Error configuring Google API Key: {e}. Check key & restart."); st.stop()
# ------------------------------------

# --- Page Config and Title ---
st.set_page_config(layout="wide", page_title="Agentic Multimodal RAG")
st.title("📄 Agentic Multimodal RAG Assistant")
st.write("Upload PDF, ask questions, get cited answers with visual context.")
if not CAMELOT_AVAILABLE: st.warning("Camelot library not found. Table extraction disabled.")
if not PILLOW_AVAILABLE: st.warning("Pillow library not found. Citation highlighting disabled.")
# ---------------------------

# --- Initialize Session State ---
# Simplified state initialization
defaults = {
    "pdf_processed": False, "doc_hash": None, "pinecone_index": None,
    "uploaded_filename": None, "processing_log": [], "display_citation": None,
    "pdf_path_in_session": None, "current_query": "", "last_processed_query": None,
    "current_answer": "", "current_citation_map": {}
}
for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
# ---------------------------

# --- Helper Function for Logging ---
def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.processing_log.append(f"[{timestamp}] {message}")

# --- PDF Upload ---
uploaded_file = st.file_uploader("1. Upload your PDF document", type="pdf")
# ----------------

# --- Processing Log Display ---
# Define here so it's always available to be updated
log_expander = st.expander("Processing Log", expanded=False)
log_container = log_expander.empty()
# Function to update the log display area
def display_log():
    if log_container: # Check if container exists
       log_container.info("\n".join(st.session_state.processing_log))
# -------------------------

# --- Main Processing Logic ---
if uploaded_file is not None:
    # Store PDF temporarily
    temp_pdf_path = os.path.join(UPLOAD_DIR, f"{hashlib.sha1(uploaded_file.name.encode()).hexdigest()}_{uploaded_file.name}")
    try:
        with open(temp_pdf_path, "wb") as f: f.write(uploaded_file.getbuffer())
        st.session_state.pdf_path_in_session = temp_pdf_path
    except Exception as file_e:
        st.error(f"Error saving uploaded file: {file_e}")
        st.stop()

    # --- Process only if file is new ---
    if st.session_state.uploaded_filename != uploaded_file.name:
        print(f"\n--- New PDF Uploaded: {uploaded_file.name} ---")
        # Clear states for new file
        st.session_state.processing_log = []
        st.session_state.pdf_processed = False
        st.session_state.doc_hash = None
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.display_citation = None
        st.session_state.current_query = ""
        st.session_state.last_processed_query = None
        st.session_state.current_answer = ""
        st.session_state.current_citation_map = {}
        # Also reset pinecone index state? Or assume it persists if config is same? Let's reset for safety.
        st.session_state.pinecone_index = None

        log_message(f"Starting processing for: {uploaded_file.name}")

        # --- Use st.status for showing progress ---
        with st.status("Processing PDF...", expanded=True) as status:
            try:
                # 1. Initialize Pinecone
                status.update(label="Initializing Vector Store...")
                st.session_state.pinecone_index = init_pinecone(
                    api_key=pinecone_api_key, index_name=PINECONE_INDEX_NAME,
                    dimension=embedding_dim, region=pinecone_region, cloud=pinecone_cloud
                )
                if st.session_state.pinecone_index is None:
                    raise ValueError("Pinecone initialization failed. Check terminal logs.")
                log_message("✅ Vector Store initialized.")

                # 2. Extract Data from PDF
                status.update(label="Extracting text, images, layout...")
                extracted_data, doc_hash = extract_pdf_data(temp_pdf_path, IMAGE_DIR)
                st.session_state.doc_hash = doc_hash
                if not doc_hash or not extracted_data:
                    raise ValueError("PDF processing failed or no data extracted.")
                log_message(f"✅ Extracted {len(extracted_data)} items. Hash: {doc_hash[:10]}...")

                # 3. Check if data exists (Skip analysis/upsert if yes)
                status.update(label="Checking existing data in vector store...")
                try:
                    existing_check = query_pinecone(st.session_state.pinecone_index, "ping", top_k=1, doc_hash=doc_hash)
                except Exception as query_e:
                     log_message(f"Warning: Error checking existing data: {query_e}. Assuming new data needed.")
                     existing_check = None

                if existing_check:
                    log_message(f"ℹ️ Data for hash {doc_hash[:10]} exists. Skipping analysis/upsert.")
                    st.session_state.pdf_processed = True
                    status.update(label="PDF already processed!", state="complete", expanded=False)

                else: # Data doesn't exist or check failed, proceed
                    log_message("ℹ️ Analyzing images and preparing data...")

                    # 4. Image Analysis
                    status.update(label="Analyzing images with Gemini Vision...")
                    processed_data_for_pinecone = []
                    items_to_analyze = [item for item in extracted_data if item['type'] == 'image_ref']
                    num_images = len(items_to_analyze)
                    for i, item in enumerate(items_to_analyze):
                         img_path=item['image_path']; page_num=item['page_number']; img_idx=item['img_index']
                         status.update(label=f"Analyzing image {i+1}/{num_images} (Page {page_num})...")
                         # No internal try/except here, let the main one catch it
                         description = analyze_image_with_gemini(img_path, google_api_key) # Uses globally configured key
                         if description == "ERROR: Invalid Google API Key": raise ValueError("Invalid Google API Key during image analysis.")
                         if description:
                             processed_data_for_pinecone.append({
                                 "type": "image_description", "description": description,
                                 "image_path": item.get('image_path'), "page_number": page_num, "img_index": img_idx,
                                 "bounding_box": item.get('bounding_box'), "page_screenshot_path": item.get('page_screenshot_path'),
                                 "linked_items": item.get('linked_items', [])
                             })
                         else: log_message(f"⚠️ Analysis skipped/failed for image: {os.path.basename(img_path)}")
                    log_message("✅ Image analysis complete.")

                    # Add non-image items
                    for item in extracted_data:
                        if item['type'] in ['text', 'table']: processed_data_for_pinecone.append(item)

                    # 5. Upsert to Pinecone
                    if processed_data_for_pinecone:
                        log_message(f"Step 3: Embedding/storing {len(processed_data_for_pinecone)} items...")
                        # Pass status update callback
                        def upsert_status_update(msg):
                            status.update(label=f"Storing data... {msg}")
                        upsert_data_to_pinecone(
                            st.session_state.pinecone_index, processed_data_for_pinecone,
                            st.session_state.doc_hash, uploaded_file.name,
                            status_callback=upsert_status_update # Pass callback
                        )
                        log_message("✅ Data stored successfully.")
                        st.session_state.pdf_processed = True # Mark ready
                    else:
                         log_message("⚠️ No data processed for upserting.")
                         st.warning("No content suitable for storing was found.")
                         st.session_state.pdf_processed = True # Still mark processed

                    # Final status update for this branch
                    status.update(label="PDF Processing Complete!", state="complete", expanded=False)

            except Exception as e:
                 # Catch any error during the status block
                 log_message(f"🔴 Processing Error: {e}\n{traceback.format_exc()}")
                 st.error(f"An error occurred during processing: {e}")
                 status.update(label="Processing Failed!", state="error", expanded=True)
                 st.session_state.pdf_processed = False # Ensure not marked as processed on error

        # Final success message outside the status block
        if st.session_state.pdf_processed:
            st.success(f"✅ PDF '{uploaded_file.name}' is ready!")

    # Update log display always after potential processing
    display_log()
    # --------------------------------------------------

# --- Query Interface ---
if st.session_state.get("pdf_processed"):
    st.header("2. Ask a Question")
    query = st.text_input("Enter question:", key="query_input", value=st.session_state.current_query)
    st.session_state.current_query = query

    col1, col2 = st.columns([2, 1])

    # Check if query processing is needed
    if query and (query != st.session_state.last_processed_query or not st.session_state.current_answer):
        if not st.session_state.pinecone_index or not st.session_state.doc_hash:
            st.error("Session state lost. Please re-upload PDF."); st.stop()

        print(f"\n--- Processing New Query: {query[:100]}... ---")
        st.session_state.display_citation = None # Reset citation display

        with st.spinner("Thinking..."): # Simple spinner for query phase ok
            # RAG
            rag_results = query_pinecone(st.session_state.pinecone_index, query, top_k=5, doc_hash=st.session_state.doc_hash)
            log_message(f"Retrieved {len(rag_results)} chunks via RAG.")

            # Conditional Web Search
            web_results = []; MIN_RAG_RESULTS = 1
            if len(rag_results) < MIN_RAG_RESULTS:
                log_message(f"⚠️ RAG results insufficient. Triggering web search."); web_results = perform_web_search(query, max_results=3)
                log_message(f"Retrieved {len(web_results)} web results.")
            else: log_message("ℹ️ Sufficient RAG results. Skipping web search.")

            # Generate Answer
            log_message("Generating final answer...")
            answer, citation_map = generate_answer_with_citations(query, rag_results, web_results, google_api_key) # Uses globally config key

            # Cache results
            st.session_state.current_answer = answer
            st.session_state.current_citation_map = citation_map
            st.session_state.last_processed_query = query
            log_message("✅ Answer generated and cached.")

    # Display Area (Uses cached results)
    with col1: # Answer Column
        st.subheader("Answer")
        answer_placeholder = st.empty()
        citation_buttons_placeholder = st.empty()
        answer_placeholder.markdown(st.session_state.get("current_answer", "*Enter a query to get an answer.*"))

        citations_in_answer = extract_citations_from_answer(st.session_state.get("current_answer", ""))
        if citations_in_answer:
            cols = citation_buttons_placeholder.columns(min(len(citations_in_answer), 6))
            col_idx = 0
            for i, cit_label in enumerate(citations_in_answer):
                button_key = f"cite_btn_{cit_label}_{i}"
                if cols[col_idx].button(cit_label, key=button_key):
                    st.session_state.display_citation = cit_label
                    st.rerun() # Rerun to display citation
                col_idx = (col_idx + 1) % len(cols)

    with col2: # Citation Column
        st.subheader("Cited Source")
        citation_display_placeholder = st.empty()

        if st.session_state.display_citation:
            citation_key = st.session_state.display_citation
            metadata = st.session_state.current_citation_map.get(citation_key)

            if metadata:
                with citation_display_placeholder.container():
                    st.markdown(f"**Displaying Source for:** `{citation_key}`")
                    source_type = metadata.get('source_type')
                    page_screenshot_path = metadata.get('page_screenshot_path')
                    bbox_strings = metadata.get('bounding_box') # Now List[str] or None

                    if source_type == 'pdf':
                        page_num = metadata.get('page_number', 'N/A')
                        content_type = metadata.get('content_type', 'N/A')
                        detail_label = f"PDF Page {page_num} - {content_type.capitalize() if content_type else 'Unknown'}"
                        st.markdown(f"**Source Details:** `{detail_label}`")

                        if page_screenshot_path and os.path.exists(page_screenshot_path):
                            if PILLOW_AVAILABLE:
                                try:
                                    img = Image.open(page_screenshot_path)
                                    draw = ImageDraw.Draw(img); img_width, img_height = img.size
                                    bbox_floats = None
                                    if isinstance(bbox_strings, list) and len(bbox_strings) == 4:
                                        try: bbox_floats = [float(coord_str) for coord_str in bbox_strings]
                                        except: pass # Ignore conversion errors, bbox_floats remains None

                                    if bbox_floats: # Draw if conversion successful
                                        draw_bbox = [max(0, bbox_floats[0]), max(0, bbox_floats[1]), min(img_width, bbox_floats[2]), min(img_height, bbox_floats[3])]
                                        draw.rectangle(draw_bbox, outline="red", width=4)
                                        st.image(img, caption=f"Page {page_num} - Cited Area Highlighted", use_container_width=True)
                                    else: # No valid bbox, show full screenshot
                                        # st.warning("Bounding box data missing/invalid.") # Less verbose
                                        st.image(img, caption=f"Page {page_num} - Screenshot", use_container_width=True)

                                    # Display text/desc/table below image
                                    if content_type == 'text': st.text_area("Cited Text:", value=metadata.get('text', 'N/A'), height=100, disabled=True, key=f"text_{citation_key}")
                                    elif content_type == 'image': st.write(f"**Image Description:** {metadata.get('image_description', 'N/A')}")
                                    elif content_type == 'table': st.text_area("Cited Table (Markdown):", value=metadata.get('table_markdown', 'N/A'), height=150, disabled=True, key=f"table_{citation_key}")

                                except Exception as img_e: st.error(f"Error displaying screenshot: {img_e}")
                            else: st.warning("Pillow needed for highlighting."); st.image(page_screenshot_path, caption=f"Page {page_num}", use_container_width=True)
                        else: st.warning(f"Screenshot not found: {page_screenshot_path}") # Fallback text display?
                    elif source_type == 'web': # Web display
                         title=metadata.get('title','N/A'); url=metadata.get('url','N/A'); snippet=metadata.get('snippet','N/A')
                         st.markdown(f"**Source Details:** `[Web: {url}]`"); st.write(f"**Title:** {title}"); st.write(f"**URL:** {url}"); st.markdown(f"**Snippet:**\n> {snippet}")
                    else: st.warning(f"Unknown source type: {source_type}")
            else:
                 st.warning(f"Could not retrieve details for citation `{citation_key}`.")

else: # No PDF processed
    st.info("Please upload a PDF document to begin.")

# --- Footer ---
st.divider()
st.caption("Multimodal RAG")
# --------------------------
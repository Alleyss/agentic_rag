# app.py
import streamlit as st
import os
import time
import re
import google.generativeai as genai
# Import functions from other modules
from utils import get_config, extract_citations_from_answer
from pdf_processor import extract_pdf_data
from image_analyzer import analyze_image_with_gemini
from vector_store import (
    init_pinecone,
    upsert_data_to_pinecone,
    query_pinecone,
    embedding_dim # Import embedding dimensio# Import if needed directly, though used within upsert
)
from web_search import perform_web_search
from answer_generator import generate_answer_with_citations

# --- Configuration ---
CONFIG = get_config() # Load updated config including region/cloud
PINECONE_INDEX_NAME = "agentic-multimodal-rag" # Choose your index name
IMAGE_DIR = "images"# Directory to store extracted images

st.set_page_config(layout="wide", page_title="Agentic Multimodal RAG")

st.title("📄 Agentic Multimodal RAG Assistant")
st.write("Upload a PDF with text, images, and graphs. Ask questions, and get answers grounded in the PDF and web search results, with citations.")

# --- Initialize Session State ---
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "doc_hash" not in st.session_state:
    st.session_state.doc_hash = None
if "pinecone_index" not in st.session_state:
    st.session_state.pinecone_index = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
if "processing_log" not in st.session_state:
    st.session_state.processing_log = []
if "citation_map" not in st.session_state:
    st.session_state.citation_map = {}
if "display_citation" not in st.session_state:
    st.session_state.display_citation = None # Stores the citation key to display
if "extracted_image_paths" not in st.session_state:
     st.session_state.extracted_image_paths = {} # Maps page_num -> list of image paths

# --- Helper Function for Logging ---
def log_message(message):
    st.session_state.processing_log.append(message)
    st.info(message) # Display message immediately


# --- PDF Upload and Processing ---
uploaded_file = st.file_uploader("1. Upload your PDF document", type="pdf")

if uploaded_file is not None:
    # Display processing log area
    log_expander = st.expander("Processing Log", expanded=True)
    log_container = log_expander.empty()

    # Use a temporary file to save the upload
    temp_pdf_path = os.path.join(".", uploaded_file.name)
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Process only if the file is new or hasn't been processed yet in this session
    if st.session_state.uploaded_filename != uploaded_file.name:
        st.session_state.processing_log = [] # Clear log for new file
        st.session_state.pdf_processed = False
        st.session_state.doc_hash = None
        st.session_state.uploaded_filename = uploaded_file.name
        st.session_state.citation_map = {}
        st.session_state.display_citation = None
        st.session_state.extracted_image_paths = {}

        log_message(f"Uploaded file: {uploaded_file.name}")

        # Initialize Pinecone
        with st.spinner("Initializing Vector Store..."):
            # --- Fetch serverless config ---
            pinecone_api_key=CONFIG.get("pinecone_api_key")
            pinecone_region=CONFIG.get("pinecone_region")
            pinecone_cloud=CONFIG.get("pinecone_cloud")
            if not all([pinecone_api_key, pinecone_region, pinecone_cloud]):
                 # Error message handled by get_config and init_pinecone
                 st.stop()
            st.session_state.pinecone_index = init_pinecone(
                api_key=pinecone_api_key,
                index_name=PINECONE_INDEX_NAME,
                dimension=embedding_dim, # Ensure this is 768
                region=pinecone_region,
                cloud=pinecone_cloud
                # metric='cosine' # Uses default cosine, or specify if needed
            )
            if st.session_state.pinecone_index is None:
                st.error("Failed to initialize Pinecone. Check logs or API keys.")
                st.stop()
            log_message("Vector Store initialized.")


        # --- Start PDF Processing ---
        with st.spinner(f"Processing PDF: {uploaded_file.name}... This may take a while for large documents."):
            log_message("Step 1: Extracting text and images from PDF...")
            extracted_data, doc_hash = extract_pdf_data(temp_pdf_path, IMAGE_DIR)
            st.session_state.doc_hash = doc_hash

            if not doc_hash:
                st.error("Failed to process PDF. Could not generate document hash.")
                st.stop()
            if not extracted_data:
                st.warning("No text or images could be extracted from the PDF.")
                # Even if nothing extracted, mark as processed to avoid re-running
                st.session_state.pdf_processed = True
                st.stop() # Stop if nothing to process further

            log_message(f"Extracted {len(extracted_data)} items. Document Hash: {doc_hash[:10]}...")

            # Store image paths by page for later display
            st.session_state.extracted_image_paths = {}
            for item in extracted_data:
                if item['type'] == 'image_ref':
                    page = item['page_number']
                    if page not in st.session_state.extracted_image_paths:
                        st.session_state.extracted_image_paths[page] = []
                    st.session_state.extracted_image_paths[page].append(item['image_path'])


            # Check if data for this document already exists in Pinecone
            # Query with a dummy vector and filter by doc_hash
            # This is a basic check; more robust checks might involve checking item count
            existing_check = query_pinecone(st.session_state.pinecone_index, "test query", top_k=1, doc_hash=doc_hash)
            if existing_check:
                 log_message(f"Data for this document (hash: {doc_hash[:10]}...) likely exists in Pinecone. Skipping analysis and upsert.")
                 st.session_state.pdf_processed = True # Mark as processed
            else:
                log_message("No existing data found in Pinecone for this document hash.")
                # --- Image Analysis ---
                log_message("Step 2: Analyzing extracted images with Gemini Vision...")
                processed_data_for_pinecone = []
                image_refs = [item for item in extracted_data if item['type'] == 'image_ref']
                num_images = len(image_refs)
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, item in enumerate(extracted_data):
                    if item['type'] == 'image_ref':
                        status_text.text(f"Analyzing image {i+1-len(processed_data_for_pinecone)}/{num_images} (Page {item['page_number']})...")
                        description = analyze_image_with_gemini(item['image_path'], CONFIG["google_api_key"])

                        if description == "ERROR: Invalid Google API Key":
                             st.error("Google API Key is invalid. Please check your configuration.")
                             st.stop()
                        if description:
                            processed_data_for_pinecone.append({
                                "type": "image_description",
                                "description": description,
                                "image_path": item['image_path'],
                                "page_number": item['page_number'],
                                "img_index": item['img_index']
                            })
                            log_message(f"Analyzed image: {os.path.basename(item['image_path'])}")
                        else:
                            log_message(f"Skipped analysis or failed for image: {os.path.basename(item['image_path'])}")
                        progress_bar.progress((i + 1 - len(processed_data_for_pinecone)) / num_images)
                    elif item['type'] == 'text':
                        processed_data_for_pinecone.append(item) # Keep text items

                status_text.text("Image analysis complete.")
                progress_bar.empty() # Remove progress bar

                # --- Upsert to Pinecone ---
                log_message("Step 3: Embedding and storing data in Vector Store...")
                upsert_data_to_pinecone(
                    st.session_state.pinecone_index,
                    processed_data_for_pinecone,
                    st.session_state.doc_hash,
                    uploaded_file.name # Pass original filename
                )
                log_message("Data stored successfully in Pinecone.")
                st.session_state.pdf_processed = True

        # Cleanup temporary PDF file
        os.remove(temp_pdf_path)
        log_message(f"Processing complete for {uploaded_file.name}.")
        st.success(f"✅ PDF '{uploaded_file.name}' processed and ready for questions!")

    # Update log display
    log_container.info("\n".join(st.session_state.processing_log))


# --- Query Interface ---
if st.session_state.get("pdf_processed"): # Check using .get for safety
    st.header("2. Ask a Question")
    query = st.text_input("Enter your question about the document:", key="query_input")

    if query:
        if not st.session_state.pinecone_index:
            st.error("Pinecone connection lost or not initialized. Please re-upload the PDF.")
            st.stop()
        if not st.session_state.doc_hash:
             st.error("Document context lost (no hash). Please re-upload the PDF.")
             st.stop()

        # Layout for answer and citation display
        col1, col2 = st.columns([2, 1]) # Answer column wider than citation

        with col1:
            st.subheader("Answer")
            answer_placeholder = st.empty()
            citation_buttons_placeholder = st.empty()

        with col2:
            st.subheader("Cited Source")
            citation_display_placeholder = st.empty()


        with st.spinner("Thinking... (Performing RAG and Web Search)"):
            # 1. RAG - Retrieve from Pinecone (filtered by current document)
            rag_results = query_pinecone(
                st.session_state.pinecone_index,
                query,
                top_k=5, # Retrieve more results for context
                doc_hash=st.session_state.doc_hash # Filter by current document
            )
            log_message(f"Retrieved {len(rag_results)} relevant chunks from PDF.")

            # 2. Web Search Agent
            web_results = perform_web_search(query, max_results=3)
            log_message(f"Retrieved {len(web_results)} relevant results from Web Search.")


            # 3. Generate Answer
            answer, citation_map = generate_answer_with_citations(
                query,
                rag_results,
                web_results,
                CONFIG["google_api_key"]
            )
            st.session_state.citation_map = citation_map # Store map for button clicks
            log_message("Generated final answer.")

        # 4. Display Answer
        answer_placeholder.markdown(answer)

        # 5. Display Citations as Buttons
        citations_in_answer = extract_citations_from_answer(answer)
        if citations_in_answer:
            # Use columns for horizontal button layout if many citations
            cols = citation_buttons_placeholder.columns(len(citations_in_answer))
            for i, cit_label in enumerate(citations_in_answer):
                # Create a unique key for each button based on the citation label
                button_key = f"cite_btn_{cit_label}"
                if cols[i].button(cit_label, key=button_key):
                     st.session_state.display_citation = cit_label # Set state to display this citation
                     # Rerun to update the citation display area
                     st.rerun()

        # 6. Display Clicked Citation Content
        if st.session_state.display_citation:
            citation_key = st.session_state.display_citation
            metadata = st.session_state.citation_map.get(citation_key)

            if metadata:
                with citation_display_placeholder.container():
                    st.markdown(f"**Source:** `{citation_key}`")
                    source_type = metadata.get('source_type')

                    if source_type == 'pdf':
                        page_num = metadata.get('page_number', 'N/A')
                        content_type = metadata.get('content_type', 'N/A')
                        st.write(f"**Source:** PDF Document")
                        st.write(f"**Page:** {page_num}")

                        if content_type == 'image':
                            img_path = metadata.get('image_path')
                            desc = metadata.get('image_description', 'No description available.')
                            st.write("**Type:** Image")
                            st.write(f"**Description:** {desc}")
                            if img_path and os.path.exists(img_path):
                                st.image(img_path, caption=f"Image from Page {page_num}", use_column_width=True)
                            else:
                                # Fallback: If image path broken, try finding *any* image from that page
                                page_images = st.session_state.extracted_image_paths.get(page_num, [])
                                if page_images and os.path.exists(page_images[0]):
                                    st.image(page_images[0], caption=f"Representative Image from Page {page_num}", use_column_width=True)
                                    st.warning(f"Could not find exact cited image '{os.path.basename(img_path)}'. Displaying first image from page {page_num}.")
                                else:
                                    st.warning(f"Could not display image (Path: {img_path}). File may be missing or inaccessible.")

                        elif content_type == 'text':
                            text_content = metadata.get('text', 'No text content available.')
                            st.write("**Type:** Text")
                            st.text_area("Cited Text:", value=text_content, height=150, disabled=True)
                             # Also show *an* image from that page for context, if available
                            page_images = st.session_state.extracted_image_paths.get(page_num, [])
                            if page_images and os.path.exists(page_images[0]):
                                st.image(page_images[0], caption=f"Representative Image from Page {page_num}", use_column_width=True)

                    elif source_type == 'web':
                        title = metadata.get('title', 'N/A')
                        url = metadata.get('url', 'N/A')
                        snippet = metadata.get('snippet', 'N/A')
                        st.write(f"**Source:** Web Search Result")
                        st.write(f"**Title:** {title}")
                        st.write(f"**URL:** {url}")
                        st.write(f"**Snippet:**")
                        st.markdown(f"> {snippet}") # Blockquote the snippet

            else:
                with citation_display_placeholder.container():
                    st.warning(f"Could not find metadata for citation: `{citation_key}`")


else:
    st.info("Please upload a PDF document to begin.")



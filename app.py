# app.py
import streamlit as st
import os
import time
import re
import google.generativeai as genai
import hashlib
import traceback # Keep for critical error logging if needed
import inspect
import uuid # For session IDs
import json # For saving/loading history

# --- Pillow Import ---
try:
    from PIL import Image, ImageDraw
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False
# ------------------------------------

# --- Module Imports ---
from utils import get_config, extract_citations_from_answer
from pdf_processor import extract_pdf_data, CAMELOT_AVAILABLE
from image_analyzer import analyze_image_with_gemini
from vector_store import (
    init_pinecone,
    upsert_data_to_pinecone,
    query_pinecone,
    embedding_dim # Make sure this is correctly reflecting the dimension used
)
from web_search import perform_web_search
from answer_generator import generate_answer_with_citations
# --------------------

# --- Page Config (MUST BE FIRST STREAMLIT COMMAND) ---
st.set_page_config(layout="wide", page_title="Agentic RAG") # Updated Title slightly

# --- Configuration ---
CONFIG = get_config()
# Ensure PINECONE_INDEX_NAME matches the index created with the correct embedding_dim
PINECONE_INDEX_NAME = "agentic-multimodal-rag-chat"
IMAGE_DIR = "images"
UPLOAD_DIR = "uploads"
CHAT_HISTORY_DIR = "chat_history"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
# --------------------

# --- Global Configs (API Keys etc.) ---
google_api_key = CONFIG.get("google_api_key")
pinecone_api_key = CONFIG.get("pinecone_api_key")
pinecone_region = CONFIG.get("pinecone_region")
pinecone_cloud = CONFIG.get("pinecone_cloud")

# --- Global Pinecone Index Initialization ---
@st.cache_resource # Cache the Pinecone index connection
def get_pinecone_index():
    # This function should only print logs, not use st.error/st.warning
    print("Attempting to initialize Pinecone...")
    if not all([pinecone_api_key, pinecone_region, pinecone_cloud]):
        print("🔴 Error: Pinecone config missing. Cannot initialize index.")
        return None
    # Assuming init_pinecone handles its own internal logging/errors
    index = init_pinecone(
        api_key=pinecone_api_key, index_name=PINECONE_INDEX_NAME,
        dimension=embedding_dim, region=pinecone_region, cloud=pinecone_cloud
    )
    if index:
        print(f"✅ Pinecone Index '{PINECONE_INDEX_NAME}' Initialized/Cached (Dim: {embedding_dim}).")
    else:
        print(f"🔴 Pinecone Index '{PINECONE_INDEX_NAME}' Initialization Failed.")
    return index

pinecone_index_global = get_pinecone_index()
# -------------------------------------------------------------------

# --- Initial Checks (AFTER set_page_config) ---
# Display Title and Caption first
st.title("📄 Agentic Multimodal RAG Assistant")
st.caption("Chat with your PDF, powered by RAG, Web Search, and Gemini.")

# Now perform checks that might use st.error/st.warning
if not google_api_key:
    st.error("🔴 Error: GOOGLE_API_KEY not found in config. App cannot function."); st.stop()

if pinecone_index_global is None:
    st.error(f"🔴 Error: Failed to initialize Pinecone index '{PINECONE_INDEX_NAME}'. Check console logs and config. App cannot function without vector store.")
    st.stop()

try:
    genai.configure(api_key=google_api_key)
    print("Google GenAI Key configured globally.")
except Exception as e:
    st.error(f"🔴 Error configuring Google API Key: {e}. Check key & restart."); st.stop()

# Optional library warnings
if not CAMELOT_AVAILABLE: st.warning("Camelot library not found. Table extraction disabled.")
if not PILLOW_AVAILABLE: st.warning("Pillow library not found. Citation highlighting disabled.")
# ------------------------------------

# --- Session State Initialization ---
# Initialize all keys used later in the app
defaults = {
    "session_id": str(uuid.uuid4()),
    "chats": {},
    "active_chat_id": None,
    "processing_log": [],
    "display_citation": None,
    "citation_message_index": None
}
for key, default_value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value
# --------------------------------------------

# --- Helper Function for Logging ---
def log_message(message):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    # Limit log size in session state to avoid excessive growth
    max_log_entries = 100
    st.session_state.processing_log = st.session_state.processing_log[-(max_log_entries-1):] + [log_entry]
    print(log_entry) # Keep console logs for debugging

# --- Chat History Saving/Loading ---
def save_chat_history(chat_id, chat_data):
    if not chat_id or not chat_data: return
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    try:
        with open(filepath, 'w') as f:
            # Use default=str to handle potential non-serializable types like timestamps if added later
            json.dump(chat_data, f, indent=4, default=str)
    except Exception as e:
        log_message(f"🔴 Error saving chat '{chat_id}': {e}")

def load_chat_history(chat_id):
    filepath = os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            log_message(f"🔴 Error loading chat '{chat_id}': {e}")
            return None
    return None

def load_all_chat_sessions():
    loaded_chats = {}
    if not os.path.exists(CHAT_HISTORY_DIR): return {}
    for filename in os.listdir(CHAT_HISTORY_DIR):
        if filename.endswith(".json"):
            # Use filename without extension as chat_id
            chat_id = os.path.splitext(filename)[0]
            chat_data = load_chat_history(chat_id)
            if chat_data:
                loaded_chats[chat_id] = chat_data
    return loaded_chats

# --- Load existing chats ONCE at the start if state is empty ---
if not st.session_state.chats:
    st.session_state.chats = load_all_chat_sessions()
    log_message(f"Loaded {len(st.session_state.chats)} existing chat sessions.")

# --- Sidebar ---
with st.sidebar:
    st.header("Chat Sessions")

    chat_options = list(st.session_state.chats.keys())
    # Sort options, perhaps numerically if possible, otherwise alphabetically
    try:
        # Attempt numeric sort first
        chat_options.sort(key=int)
    except ValueError:
        # Fallback to string sort if IDs are not purely numeric
        chat_options.sort()

    chat_display_names = {
        chat_id: st.session_state.chats[chat_id].get("chat_name", f"Chat {chat_id}") # Simpler default display
        for chat_id in chat_options
    }

    def switch_active_chat(new_chat_id):
        if st.session_state.active_chat_id and st.session_state.active_chat_id in st.session_state.chats:
            save_chat_history(st.session_state.active_chat_id, st.session_state.chats[st.session_state.active_chat_id])
        st.session_state.active_chat_id = new_chat_id
        log_message(f"Switched to chat: {new_chat_id}")
        st.session_state.display_citation = None
        st.session_state.citation_message_index = None

    if chat_options:
        try:
            # Ensure active_chat_id is a string if options are strings
            active_chat_id_str = str(st.session_state.active_chat_id) if st.session_state.active_chat_id is not None else None
            active_index = chat_options.index(active_chat_id_str) if active_chat_id_str in chat_options else 0
        except ValueError: active_index = 0

        selected_chat_id = st.selectbox(
            "Select Chat", options=chat_options,
            format_func=lambda x: chat_display_names.get(x, x), # Use the display name mapping
            index=active_index, key="chat_selector",
        )
        # Check if selection changed
        if selected_chat_id != st.session_state.active_chat_id:
             switch_active_chat(selected_chat_id)
             st.rerun()
    else:
        st.write("No chats yet. Start a new one!")

    # --- Helper for Next Chat ID ---
    def get_next_chat_id():
        existing_ids = st.session_state.chats.keys()
        max_id = 100
        for chat_id in existing_ids:
            try:
                numeric_id = int(chat_id)
                if numeric_id > max_id: max_id = numeric_id
            except (ValueError, TypeError): continue
        return max_id + 1

    if st.button("➕ New Chat", key="new_chat_button"):
        if st.session_state.active_chat_id and st.session_state.active_chat_id in st.session_state.chats:
            save_chat_history(st.session_state.active_chat_id, st.session_state.chats[st.session_state.active_chat_id])

        next_id_num = get_next_chat_id()
        new_chat_id = str(next_id_num) # Use string for key
        default_chat_name = f"Chat {new_chat_id}"

        st.session_state.chats[new_chat_id] = {
            "messages": [{"role": "assistant", "content": "Hi! Upload a PDF and ask me questions about it."}],
            "pdf_hash": None, "pdf_filename": None, "pdf_processed": False,
            "chat_name": default_chat_name, "citation_map": {},
            "last_answer": None, "last_query": None
        }
        # Switch to the new chat
        switch_active_chat(new_chat_id) # Use the function to handle state update
        log_message(f"Created and switched to new chat: {new_chat_id} ({default_chat_name})")
        st.rerun()

    st.divider()
    st.header("Document Context")

    # PDF Upload Section (Keep as before)
    if st.session_state.active_chat_id:
        active_chat_data = st.session_state.chats[st.session_state.active_chat_id]
        # ... (Display PDF status logic) ...
        if active_chat_data.get("pdf_filename"):
             st.info(f"Doc: **{active_chat_data['pdf_filename']}**")
             if active_chat_data.get("pdf_processed"): st.success("✅ Ready")
             else: st.warning("⏳ Processing...")
        else:
             st.info("No PDF uploaded for this chat.")

        uploaded_file = st.file_uploader(
            "Upload PDF for this chat", type="pdf",
            key=f"pdf_uploader_{st.session_state.active_chat_id}"
        )

        if uploaded_file is not None:
            current_pdf_filename = active_chat_data.get("pdf_filename")
            if uploaded_file.name != current_pdf_filename:
                log_message(f"New PDF upload triggered for chat '{st.session_state.active_chat_id}': {uploaded_file.name}")
                # Reset chat state for new PDF
                active_chat_data["pdf_processed"] = False
                active_chat_data["messages"] = [{"role": "assistant", "content": f"Processing {uploaded_file.name}..."}]
                active_chat_data["citation_map"] = {}
                active_chat_data["last_answer"] = None
                st.session_state.display_citation = None # Reset citation display
                st.session_state.citation_message_index = None

                # --- Save and Process PDF ---
                temp_pdf_path = os.path.join(UPLOAD_DIR, f"{hashlib.sha1(uploaded_file.name.encode() + str(time.time()).encode()).hexdigest()}_{uploaded_file.name}")
                try:
                    with open(temp_pdf_path, "wb") as f: f.write(uploaded_file.getbuffer())
                    active_chat_data["pdf_path_in_session"] = temp_pdf_path # May not be needed long term
                    log_message(f"Saved uploaded file to: {temp_pdf_path}")

                    # --- Start PDF Processing Status ---
                    with st.status(f"Processing {uploaded_file.name}...", expanded=True) as status:
                        try:
                            status.update(label="Extracting data...")
                            # Call pdf_processor
                            extracted_data, doc_hash = extract_pdf_data(temp_pdf_path, IMAGE_DIR)
                            if not doc_hash or not extracted_data: raise ValueError("PDF processing failed or no data extracted.")
                            active_chat_data["pdf_hash"] = doc_hash
                            active_chat_data["pdf_filename"] = uploaded_file.name
                            log_message(f"✅ Extracted {len(extracted_data)} items. Hash: {doc_hash[:10]}...")
                            status.update(label=f"Extracted {len(extracted_data)} items.")

                            # Check vector store
                            status.update(label="Checking vector store...")
                            data_exists = False
                            try:
                                if pinecone_index_global: # Check if index is valid
                                     existing_check = query_pinecone(pinecone_index_global, "ping", top_k=1, doc_hash=doc_hash)
                                     data_exists = bool(existing_check)
                            except Exception as query_e:
                                log_message(f"⚠️ Warning checking existing data: {query_e}. Assuming new data needed.")

                            if data_exists:
                                log_message(f"ℹ️ Data for hash {doc_hash[:10]} exists. Skipping analysis/upsert.")
                                active_chat_data["pdf_processed"] = True
                                status.update(label="Document already processed!", state="complete", expanded=False)
                            else:
                                log_message("ℹ️ Analyzing images and preparing data for vector store...")
                                status.update(label="Analyzing images...")
                                # --- Image Analysis and Prep ---
                                processed_data_for_pinecone = []
                                # (Keep image analysis loop as before)
                                items_to_analyze = [item for item in extracted_data if item['type'] == 'image_ref']
                                num_images = len(items_to_analyze)
                                for i, item in enumerate(items_to_analyze):
                                     img_path=item['image_path']; page_num=item['page_number']; img_idx=item['img_index']
                                     status.update(label=f"Analyzing image {i+1}/{num_images} (Page {page_num})...")
                                     description = analyze_image_with_gemini(img_path, google_api_key)
                                     if description == "ERROR: Invalid Google API Key": raise ValueError("Invalid Google API Key.")
                                     if description:
                                          processed_data_for_pinecone.append({
                                              "type": "image_description", "description": description,
                                              "image_path": item.get('image_path'), "page_number": page_num,
                                              "img_index": img_idx, "bounding_box": item.get('bounding_box'),
                                              "page_screenshot_path": item.get('page_screenshot_path'),
                                              "linked_items": item.get('linked_items', [])
                                          })
                                     else: log_message(f"⚠️ Analysis skipped/failed for image: {os.path.basename(img_path)}")
                                log_message("✅ Image analysis complete.")
                                # Add Text/Table Items
                                for item in extracted_data:
                                    if item['type'] in ['text', 'table']: processed_data_for_pinecone.append(item)

                                # --- Upsert to Pinecone ---
                                if processed_data_for_pinecone:
                                    status.update(label=f"Storing {len(processed_data_for_pinecone)} items...")
                                    def upsert_status_update(msg): status.update(label=f"Storing data... {msg}")
                                    if pinecone_index_global: # Check index again before upsert
                                        upsert_data_to_pinecone(
                                            pinecone_index_global, processed_data_for_pinecone,
                                            doc_hash, uploaded_file.name, status_callback=upsert_status_update
                                        )
                                        log_message("✅ Data stored successfully.")
                                        active_chat_data["pdf_processed"] = True
                                    else:
                                        log_message("🔴 Cannot upsert data, Pinecone index not available.")
                                        st.error("Pinecone index connection lost. Cannot store document data.")
                                        raise ConnectionError("Pinecone index not available for upsert")
                                else:
                                     log_message("⚠️ No data processed for upserting.")
                                     st.warning("No content suitable for storing was found.")
                                     active_chat_data["pdf_processed"] = True # Mark processed even if empty

                                status.update(label="Processing Complete!", state="complete", expanded=False)

                        except Exception as e:
                             # Catch processing errors
                             log_message(f"🔴 Processing Error for {uploaded_file.name}: {e}\n{traceback.format_exc()}")
                             st.error(f"An error occurred during PDF processing: {e}")
                             status.update(label="Processing Failed!", state="error", expanded=True)
                             # Reset PDF state in chat on failure
                             active_chat_data["pdf_processed"] = False
                             active_chat_data["pdf_filename"] = None
                             active_chat_data["pdf_hash"] = None
                except Exception as file_e:
                     # Catch file saving errors
                     st.error(f"Error saving uploaded file: {file_e}"); log_message(f"🔴 Error saving PDF: {file_e}")
                finally:
                     # Save chat state regardless of processing outcome
                     save_chat_history(st.session_state.active_chat_id, active_chat_data)
                     st.rerun() # Rerun to reflect the changes
    else:
        st.info("Start or select a chat to upload a PDF.")

    st.divider()
    st.header("Processing Log")
    log_expander = st.expander("Show Log", expanded=False)
    with log_expander:
        # Display logs in reverse chronological order
        st.info("\n".join(st.session_state.processing_log[::-1]))


# --- Main Chat Area ---
if st.session_state.active_chat_id and st.session_state.active_chat_id in st.session_state.chats:
    active_chat = st.session_state.chats[st.session_state.active_chat_id]

    # --- Display Chat Messages ---
    chat_container = st.container()
    with chat_container:
        # Iterate through messages with index
        for message_index, message in enumerate(active_chat.get("messages", [])):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # --- Citation Buttons ---
                # Check if the message is from assistant AND has citations
                if message.get("role") == "assistant" and message.get("citations_in_answer"):
                    citations_in_answer = message["citations_in_answer"]
                    # Create columns for buttons
                    cols = st.columns(min(len(citations_in_answer), 8)) # Max 8 buttons per row
                    col_idx = 0
                    for i, cit_label in enumerate(citations_in_answer):
                        # Unique key for each button
                        button_key = f"cite_btn_{st.session_state.active_chat_id}_{message_index}_{cit_label}"
                        # Check if the button is clicked
                        if cols[col_idx].button(cit_label, key=button_key, help=f"Show source {cit_label}"):
                            # Store which citation and message index was clicked
                            st.session_state.display_citation = cit_label
                            st.session_state.citation_message_index = message_index
                            st.rerun() # Rerun to show the expander
                        col_idx = (col_idx + 1) % len(cols) # Move to next column, wrap around

            # --- !!! CITATION EXPANDER DISPLAY LOGIC !!! ---
            # Check if a citation needs to be displayed for THIS message index
            if st.session_state.get("display_citation") and st.session_state.get("citation_message_index") == message_index:
                citation_key = st.session_state.display_citation
                # Get the citation map associated with the current active chat
                metadata = active_chat.get("citation_map", {}).get(citation_key)

                if metadata:
                    expander_label = f"Source Details for {citation_key}"
                    with st.expander(expander_label, expanded=True): # Keep expanded when first shown
                        source_type = metadata.get('source_type')

                        # --- Display PDF Source ---
                        if source_type == 'pdf':
                            page_num = metadata.get('page_number', 'N/A')
                            content_type = metadata.get('content_type', 'N/A') # text, image, table
                            detail_label = f"PDF Page {page_num} - {content_type.capitalize() if content_type else 'Unknown'}"
                            st.markdown(f"**Source:** `{detail_label}`")

                            page_screenshot_path = metadata.get('page_screenshot_path')
                            bbox_strings = metadata.get('bounding_box') # List[str] or None

                            # Display screenshot with highlight if available
                            if page_screenshot_path and os.path.exists(page_screenshot_path) and PILLOW_AVAILABLE:
                                try:
                                    img = Image.open(page_screenshot_path)
                                    draw = ImageDraw.Draw(img); img_width, img_height = img.size
                                    bbox_floats = None
                                    # Safely convert bbox strings to floats
                                    if isinstance(bbox_strings, list) and len(bbox_strings) == 4:
                                        try: bbox_floats = [float(coord_str) for coord_str in bbox_strings]
                                        except (ValueError, TypeError): pass

                                    # Draw rectangle if bbox is valid
                                    if bbox_floats:
                                        draw_bbox = [ max(0, bbox_floats[0]), max(0, bbox_floats[1]), min(img_width, bbox_floats[2]), min(img_height, bbox_floats[3]) ]
                                        draw.rectangle(draw_bbox, outline="red", width=3)
                                        st.image(img, caption=f"Page {page_num} - Cited Area Highlighted", use_container_width=True)
                                    else: # Show full screenshot if no valid bbox
                                        st.image(img, caption=f"Page {page_num} - Full Screenshot", use_container_width=True)
                                except Exception as img_e:
                                    st.error(f"Error displaying screenshot: {img_e}")
                                    log_message(f"🔴 Error displaying image {page_screenshot_path}: {img_e}")
                            elif page_screenshot_path and os.path.exists(page_screenshot_path):
                                # Fallback if Pillow is missing
                                st.image(page_screenshot_path, caption=f"Page {page_num}", use_container_width=True)
                            # else: st.warning(f"Screenshot not found.") # Optional warning

                            # Display text/desc/table chunk using a unique key for the widget
                            # Combine citation key and message index for uniqueness
                            widget_key_base = f"{citation_key}_{message_index}"
                            chunk_content_display = None
                            if content_type == 'text':
                                 chunk_content_display = metadata.get('chunk_content', metadata.get('text', 'N/A'))
                                 st.text_area("Cited Text Chunk:", value=chunk_content_display, height=150, disabled=True, key=f"text_{widget_key_base}")
                            elif content_type == 'image':
                                 chunk_content_display = metadata.get('chunk_content', metadata.get('image_description', 'N/A'))
                                 st.text_area("Cited Image Desc. Chunk:", value=chunk_content_display, height=150, disabled=True, key=f"img_{widget_key_base}")
                            elif content_type == 'table':
                                 chunk_content_display = metadata.get('chunk_content', metadata.get('table_markdown', 'N/A'))
                                 st.text_area("Cited Table Chunk (MD):", value=chunk_content_display, height=150, disabled=True, key=f"table_{widget_key_base}")

                        # --- Display Web Source ---
                        elif source_type == 'web':
                            title=metadata.get('title','N/A'); url=metadata.get('url','N/A'); snippet=metadata.get('snippet','N/A')
                            st.markdown(f"**Source:** `[Web]`")
                            if title != 'N/A': st.write(f"**Title:** {title}")
                            if url != 'N/A': st.markdown(f"**URL:** [{url}]({url})") # Make URL clickable
                            if snippet != 'N/A': st.text_area("Snippet:", value=snippet, height=150, disabled=True, key=f"web_{citation_key}_{message_index}")
                        else:
                            st.warning(f"Unknown source type in citation metadata: {source_type}")

                else:
                     # Handle case where metadata key is somehow invalid
                     with st.expander(f"Source Details for {citation_key}", expanded=True):
                        st.warning(f"Could not retrieve details for citation `{citation_key}`.")
            # --- !!! END CITATION EXPANDER LOGIC !!! ---


    # --- Chat Input Logic ---
    if prompt := st.chat_input("Ask a question about the PDF...", key=f"chat_input_{st.session_state.active_chat_id}", disabled=not active_chat.get("pdf_processed")):

        # Reset citation display state when user asks a new question
        st.session_state.display_citation = None
        st.session_state.citation_message_index = None

        # Append user message to chat history
        active_chat["messages"].append({"role": "user", "content": prompt, "timestamp": time.time()})
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- Generate and Display Assistant Response ---
        # Check if PDF is ready for the active chat
        if not active_chat.get("pdf_hash"):
            with st.chat_message("assistant"):
                st.markdown("I need a processed PDF document to answer questions about it.")
            active_chat["messages"].append({"role": "assistant", "content": "I need a processed PDF document to answer questions about it.", "timestamp": time.time()})
        else:
            # Display thinking message and spinner while processing
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking... 🤔")
                rag_results = []
                web_results = []
                final_answer = None
                citation_map = {}
                citations_in_answer = [] # Initialize here

                # Step 1 & 2: Perform RAG and conditional Web Search
                with st.spinner("Searching document and web..."):
                    try:
                        log_message(f"Querying Pinecone for: '{prompt[:50]}...'")
                        if pinecone_index_global: # Ensure index is valid
                             rag_results = query_pinecone(pinecone_index_global, prompt, top_k=5, doc_hash=active_chat["pdf_hash"])
                             log_message(f"Retrieved {len(rag_results)} chunks via RAG.")
                        else: log_message("🔴 Pinecone index not available for query.")
                    except Exception as rag_e:
                        log_message(f"🔴 Error during RAG search: {rag_e}")
                        st.toast("Error searching document.", icon="⚠️")

                    MIN_RAG_FOR_WEB_SKIP = 10 # Only search web if RAG finds nothing
                    if len(rag_results) < MIN_RAG_FOR_WEB_SKIP:
                        try:
                            log_message(f"ℹ️ RAG results insufficient ({len(rag_results)}). Triggering web search.")
                            web_results = perform_web_search(prompt, max_results=3)
                            log_message(f"Retrieved {len(web_results)} web results.")
                            log_message(f"Web results content: {web_results}")
                        except Exception as web_e:
                            log_message(f"🔴 Error during web search: {web_e}")
                            st.toast("Error during web search.", icon="⚠️")
                    else:
                        log_message(f"ℹ️ Sufficient RAG results ({len(rag_results)}). Skipping web search.")

                # Step 3: Decide whether to call LLM or use apology
                if not rag_results and not web_results:
                    log_message("🔴 No context found from RAG or Web Search.")
                    final_answer = "I apologize, but I couldn't find relevant information in the uploaded document or via web search to answer your question."
                    # citation_map remains empty, citations_in_answer remains empty
                else:
                    # Context found, call LLM
                    log_message("Context found. Generating final answer with LLM...")
                    with st.spinner("Generating answer..."):
                        try:
                            answer_llm, citation_map_llm = generate_answer_with_citations(
                                prompt, rag_results, web_results, google_api_key
                            )
                            final_answer = answer_llm
                            citation_map = citation_map_llm # Use the map from the LLM
                            citations_in_answer = extract_citations_from_answer(final_answer)
                            log_message("✅ Answer generated by LLM.")
                        except Exception as gen_e:
                            log_message(f"🔴 Error during answer generation: {gen_e}")
                            st.toast("Error generating final answer.", icon="⚠️")
                            final_answer = "Sorry, I encountered an error while trying to generate the answer."
                            # Keep citation_map/citations_in_answer empty on error

                # Step 4: Display final answer (apology or LLM result)
                message_placeholder.markdown(final_answer if final_answer else "An unexpected error occurred.")

                # Step 5: Append assistant response to chat history
                active_chat["messages"].append({
                    "role": "assistant",
                    "content": final_answer if final_answer else "Error retrieving answer.",
                    "citations_in_answer": citations_in_answer, # Store citations found
                    "timestamp": time.time() # Add timestamp for potential sorting later
                })
                # Update chat state with latest query/answer/map
                active_chat["last_query"] = prompt
                active_chat["last_answer"] = final_answer
                active_chat["citation_map"] = citation_map

        # Always save history after an interaction cycle
        save_chat_history(st.session_state.active_chat_id, active_chat)
        # Rerun to display the new message and update button states/expanders
        st.rerun()

# Display message if no chat is active
elif not st.session_state.active_chat_id:
    st.info("👈 Start a new chat or select an existing one from the sidebar.")

# Optional Footer
# st.divider()
# st.caption("Application Footer")
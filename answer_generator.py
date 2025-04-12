# answer_generator.py
import google.generativeai as genai
import os

# --- CHANGED: Format context with new types and headers ---
def format_context(rag_results, web_results):
    """Formats retrieved context for the LLM prompt with prioritization headers."""
    context_str = "Context:\n"
    citation_map = {} # To link simple citations in text to full metadata

    context_str += "--- PDF Context (Primary Source) ---\n" # Prioritization Header
    if not rag_results:
        context_str += "No relevant content found in the PDF.\n"
    for match in rag_results:
        meta = match.get('metadata', {})
        content_type = meta.get('content_type', 'unknown')
        page_num = meta.get('page_number', 'N/A')
        content = ""
        citation_label = ""

        # Use a simple index for citation labels for now
        citation_index = len(citation_map) + 1
        base_label = f"[{citation_index}]" # Simple numeric label

        if content_type == 'text':
            content = meta.get('text', '')
            # Detail label for mapping, simple label for text insertion
            detail_label = f"[PDF Page {page_num} - Text]"
            citation_label = base_label
            citation_map[citation_label] = meta # Store full metadata under simple label
            context_str += f"{citation_label} (Source: {detail_label})\n{content}\n\n"
        elif content_type == 'image':
            content = meta.get('image_description', '')
            detail_label = f"[PDF Page {page_num} - Image]"
            citation_label = base_label
            citation_map[citation_label] = meta
            context_str += f"{citation_label} (Source: {detail_label})\nImage Description: {content}\n\n"
        # --- ADDED: Handle table type ---
        elif content_type == 'table':
            content = meta.get('table_markdown', '')
            detail_label = f"[PDF Page {page_num} - Table]"
            citation_label = base_label
            citation_map[citation_label] = meta
            context_str += f"{citation_label} (Source: {detail_label})\nTable Content (Markdown):\n{content}\n\n"
        # --------------------------------

    context_str += "--- Web Search Results (Supplementary) ---\n" # Prioritization Header
    if not web_results:
        context_str += "No relevant content found via web search.\n"
    for result in web_results:
        citation_index = len(citation_map) + 1 # Continue numbering
        base_label = f"[{citation_index}]"
        title = result.get('title', 'N/A')
        url = result.get('url', 'N/A')
        snippet = result.get('snippet', 'N/A')
        detail_label = f"[Web: {url}]" # Detail label for mapping
        citation_label = base_label
        citation_map[citation_label] = result # Store full metadata under simple label
        context_str += f"{citation_label} (Source: {detail_label})\nTitle: {title}\nSnippet: {snippet}\n\n"

    return context_str.strip(), citation_map
# -----------------------------------------------------------

# --- CHANGED: Updated prompt with prioritization instructions ---
def generate_answer_with_citations(query, rag_results, web_results, api_key):
    """Generates an answer using Gemini Flash, based on context, with citations."""
    if not api_key:
        print("Error: GOOGLE_API_KEY not configured.")
        return "Error: Google API Key not configured.", {}

    # Configure GenAI (consider doing this globally in app.py instead)
    try:
        genai.configure(api_key=api_key)
    except Exception as e:
        return f"Error configuring Google API: {e}", {}

    # Use a capable model, flash might be too weak for complex instructions
    # model = genai.GenerativeModel('gemini-1.5-flash') # Or gemini-1.0-pro
    model_name = 'gemini-2.0-flash' # Start with Pro for better instruction following
    print(f"Using generative model: {model_name}")
    model = genai.GenerativeModel(model_name)


    context_string, citation_map = format_context(rag_results, web_results)

    # Strengthened Prompt
    prompt = f"""You are an AI assistant answering questions based *only* on the provided context sections below.

Query: {query}

{context_string}

**Instructions:**
1.  Answer the query based *strictly* on the information provided in the context sections.
2.  **Prioritize information from the '--- PDF Context (Primary Source) ---' section.**
3.  Use '--- Web Search Results (Supplementary) ---' ONLY if the PDF section does not provide the answer OR to add directly related, non-contradictory details that enhance the PDF information.
4.  If the Web Search Results seem irrelevant to the PDF context or the query's focus on the document, state that the web results are potentially unrelated and primarily base the answer on the PDF context. Do NOT use irrelevant web results.
5.  Cite your sources within the answer using the simple numeric citation labels provided (e.g., "[1]", "[2]"). Do NOT use the detail labels like "[PDF Page X]".
6.  If the context (including both PDF and relevant Web sections) does not contain the answer, explicitly state that the information is not available in the provided sources.
7.  Be concise and directly answer the query.

Answer:
"""

    print("\n--- Sending Prompt to LLM ---")
    # Limit printing very long prompts/context
    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
    print("--- End Prompt ---")

    try:
        # Increase safety thresholds slightly if needed, but start with defaults
        # safety_settings = { ... }
        response = model.generate_content(prompt) # Add safety_settings=safety_settings if defined

        # --- Robust check for response content ---
        try:
            generated_answer = response.text
            print("Answer generated successfully.")
        except ValueError: # Handle case where response.text might raise error (e.g., blocked)
             print(f"Warning: Could not access response.text. Block Reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'N/A'}")
             # Try to get safety feedback
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
             safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else 'N/A'
             error_message = f"Could not generate answer. Response potentially blocked (Reason: {block_reason}). Safety Ratings: {safety_ratings}"
             return error_message, citation_map
        except AttributeError: # Handle case where response structure is unexpected
             print(f"Warning: Unexpected response structure. Response: {response}")
             return "Could not generate answer due to unexpected response format.", citation_map

        if not generated_answer: # Check for empty text even if .text didn't raise ValueError
             block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
             safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else 'N/A'
             print(f"Warning: LLM response text is empty. Reason: {block_reason}, Safety: {safety_ratings}")
             error_message = f"Could not generate answer. LLM returned empty text (Reason: {block_reason})."
             return error_message, citation_map
        # -----------------------------------------

        # Return the generated answer and the map (linking simple labels to full metadata)
        return generated_answer, citation_map

    except Exception as e:
        print(f"Error during Gemini API call ({model_name}): {e}")
        # Check for specific API errors
        if "API key not valid" in str(e):
             return "Authentication Error: Invalid Google API Key.", {}
        elif "quota" in str(e).lower() or "resource has been exhausted" in str(e).lower():
             return "API Limit Error: Quota exceeded. Please try again later.", {}
        else:
             return f"An error occurred while generating the answer: {e}", {}
# -----------------------------------------------------------
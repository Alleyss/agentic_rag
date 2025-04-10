# answer_generator.py
import google.generativeai as genai
import os

def format_context(rag_results, web_results):
    """Formats retrieved context for the LLM prompt."""
    context_str = "Context:\n"
    citation_map = {} # To link simple citations in text to full metadata
    citation_index = 1

    context_str += "--- PDF Content ---\n"
    if not rag_results:
        context_str += "No relevant content found in the PDF.\n"
    for match in rag_results:
        meta = match.get('metadata', {})
        content_type = meta.get('content_type', 'unknown')
        page_num = meta.get('page_number', 'N/A')
        content = ""
        citation_label = ""

        if content_type == 'text':
            content = meta.get('text', '')
            citation_label = f"[PDF Page {page_num}]"
            citation_map[citation_label] = meta # Store full metadata
            context_str += f"{citation_label}\n{content}\n\n"
        elif content_type == 'image':
            content = meta.get('image_description', '')
            citation_label = f"[Image Page {page_num}]"
            citation_map[citation_label] = meta # Store full metadata
            context_str += f"{citation_label}\nImage Description: {content}\n\n"
        citation_index += 1 # Increment even if content wasn't added, to keep map consistent? Maybe not needed.

    context_str += "--- Web Search Results ---\n"
    if not web_results:
        context_str += "No relevant content found via web search.\n"
    for result in web_results:
        title = result.get('title', 'N/A')
        url = result.get('url', 'N/A')
        snippet = result.get('snippet', 'N/A')
        citation_label = f"[Web: {url}]"
        citation_map[citation_label] = result # Store full metadata
        context_str += f"{citation_label}\nTitle: {title}\nSnippet: {snippet}\n\n"
        citation_index += 1


    return context_str.strip(), citation_map


def generate_answer_with_citations(query, rag_results, web_results, api_key):
    """Generates an answer using Gemini Pro, based on context, with citations."""
    if not api_key:
        print("Error: GOOGLE_API_KEY not configured.")
        return "Error: Google API Key not configured.", {}

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')

    context_string, citation_map = format_context(rag_results, web_results)

    prompt = f"""You are an AI assistant answering questions based *only* on the provided context from a PDF document and web search results.

Query: {query}

{context_string}

Answer the query based *strictly* on the information provided above.
Cite your sources within the answer using the exact citation labels provided (e.g., "[PDF Page X]", "[Image Page Y]", "[Web: URL]").
If the context does not contain the answer, state that the information is not available in the provided sources.
Be concise and directly answer the question.

Answer:
"""

    # print("\n--- Sending Prompt to LLM ---")
    # print(prompt) # For debugging
    # print("--- End Prompt ---")


    try:
        response = model.generate_content(prompt)
        # print("\n--- LLM Raw Response ---")
        # print(response) # For debugging
        # print("--- End Raw Response ---")

        if not response.parts:
            # Handle blocked responses or empty parts
            block_reason = response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'
            safety_ratings = response.prompt_feedback.safety_ratings if response.prompt_feedback else 'N/A'
            print(f"Warning: LLM response blocked or empty. Reason: {block_reason}, Safety: {safety_ratings}")
            error_message = f"Could not generate answer. The response was blocked (Reason: {block_reason})."
            # Check if content was filtered due to safety
            if block_reason and block_reason != 'BLOCK_REASON_UNSPECIFIED':
                 error_message = f"Could not generate answer. The response was blocked due to safety concerns ({block_reason}). Please try rephrasing your query or check the input document."
            elif not response.text:
                 error_message = "Could not generate answer. The LLM returned an empty response."

            return error_message, citation_map # Return map anyway for potential debugging

        generated_answer = response.text
        print("Answer generated successfully.")
        # The citation_map created during context formatting is returned
        # It links the labels potentially used in the answer back to the full metadata
        return generated_answer, citation_map

    except Exception as e:
        print(f"Error during Gemini Pro API call: {e}")
        return f"An error occurred while generating the answer: {e}", {}
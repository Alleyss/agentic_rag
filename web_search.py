# web_search.py
from duckduckgo_search import DDGS
import time

def perform_web_search(query, max_results=3):
    """Performs a web search using DuckDuckGo Search."""
    results_list = []
    try:
        print(f"Performing web search for: {query}")
        with DDGS() as ddgs:
            search_results = ddgs.text(query, max_results=max_results)
            if not search_results:
                 print("No web results found.")
                 return []

            for i, result in enumerate(search_results):
                print(f"  Web Result {i+1}: {result.get('title')} - {result.get('href')}")
                results_list.append({
                    'source_type': 'web', # Consistent key
                    'title': result.get('title', 'N/A'),
                    'url': result.get('href', 'N/A'),
                    'snippet': result.get('body', 'N/A'), # 'body' is the snippet
                    'content_type': 'web_result' # Consistent key for type within source
                })

        print(f"Web search complete. Found {len(results_list)} results.")
        return results_list

    except Exception as e:
        print(f"Error during web search for '{query}': {e}")
        return [] # Return empty list on error
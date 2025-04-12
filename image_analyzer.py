# image_analyzer.py
import google.generativeai as genai
from PIL import Image
import os
import time

def analyze_image_with_gemini(image_path, api_key):
    """
    Analyzes an image using Gemini Vision Pro.

    Returns:
        str: Description of the image, or None if analysis fails.
    """
    if not api_key:
        print("Error: GOOGLE_API_KEY not configured.")
        return None
    genai.configure(api_key=api_key)

    # Check model availability if needed (optional)
    # for m in genai.list_models():
    #   if 'generateContent' in m.supported_generation_methods:
    #     print(m.name)

    model = genai.GenerativeModel('gemini-2.0-flash')
    print(f"Analyzing image: {image_path}")
    if not os.path.exists(image_path):
         print(f"Error: Image path does not exist: {image_path}")
         return None
    try:
        img = Image.open(image_path)
        # Gemini Vision API works better with slightly more specific prompts
        prompt = ("Describe this image in detail. If it's a graph or chart, "
                  "explain what it shows, including trends, key data points, "
                  "and labels if possible. If it's a diagram, explain its components "
                  "and relationships. If it's a general image, describe the scene "
                  "and objects.")

        # Add retries for potential API flakiness
        for attempt in range(3):
            try:
                response = model.generate_content([prompt, img], stream=False)
                response.resolve() # Ensure response is fully generated
                # Check for safety ratings or empty content
                if not response.parts:
                     print(f"Warning: No content parts in response for {image_path}. Safety: {response.prompt_feedback}")
                     # You might want specific handling based on response.prompt_feedback.block_reason
                     # For simplicity, we'll return None here if blocked or empty
                     return None

                description = response.text
                print(f"Analysis complete for: {image_path}")
                return description
            except Exception as e:
                print(f"Error during Gemini Vision API call (Attempt {attempt+1}/3) for {image_path}: {e}")
                if "API key not valid" in str(e):
                    return "ERROR: Invalid Google API Key" # Specific error feedback
                if "Resource has been exhausted" in str(e) or "429" in str(e):
                    print("Rate limit likely hit, sleeping...")
                    time.sleep(5 + attempt * 5) # Exponential backoff
                elif attempt == 2: # Last attempt failed
                     print(f"Failed to analyze {image_path} after multiple retries.")
                     return None # Or raise the exception: raise e
                else:
                    time.sleep(2) # Short sleep for other transient errors

    except Exception as e:
        print(f"Error opening or processing image for analysis {image_path}: {e}")
        return None

    return None # Should not be reached if logic is correct
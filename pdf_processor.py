# pdf_processor.py
import pymupdf # <--- CHANGE: Import pymupdf instead of fitz
import os
from PIL import Image
import io
import hashlib

def get_doc_hash(filepath):
    """Generates a SHA256 hash for the document."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def extract_pdf_data(pdf_path, image_output_dir="images"):
    """
    Extracts text chunks and images from a PDF, saving images locally.

    Returns:
        list: A list of dictionaries, each containing 'type', 'content',
              'page_number', and potentially 'image_path'.
        str: A unique hash representing the document content.
    """
    extracted_data = []
    doc_hash = get_doc_hash(pdf_path)
    image_save_dir = os.path.join(image_output_dir, doc_hash) # Unique dir per doc
    os.makedirs(image_save_dir, exist_ok=True)

    try:
        # --- CHANGE: Use pymupdf.open() instead of fitz.open() ---
        doc = pymupdf.open(pdf_path)
        # ---------------------------------------------------------
        print(f"Processing PDF: {pdf_path}, Pages: {len(doc)}") # len(doc) still works

        for page_num in range(len(doc)):
            page = doc.load_page(page_num) # .load_page() method remains the same
            page_number = page_num + 1 # 1-based index

            # Extract Text
            text = page.get_text("text") # .get_text() method remains the same
            if text.strip():
                # Basic chunking (could be more sophisticated)
                chunks = text.split('\n\n') # Split by double newline
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                         extracted_data.append({
                            "type": "text",
                            "content": chunk.strip(),
                            "page_number": page_number,
                            "chunk_index": i, # Optional index within page
                        })

            # Extract Images
            image_list = page.get_images(full=True) # .get_images() method remains the same
            print(f"Page {page_number}: Found {len(image_list)} images.")

            for img_index, img in enumerate(image_list):
                xref = img[0]
                # .extract_image() method remains the same
                base_image = doc.extract_image(xref)
                # Handle potential dictionary changes if any (unlikely for basic extraction)
                if not base_image:
                    print(f"Warning: Could not extract image data for xref {xref} on page {page_number}.")
                    continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Save image
                image_filename = f"page_{page_number}_img_{img_index}.{image_ext}"
                image_path = os.path.join(image_save_dir, image_filename)

                try:
                    # Validate and save image
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    img_pil.verify() # Basic validation
                    img_pil = Image.open(io.BytesIO(image_bytes)) # Re-open after verify

                    # Ensure image has content
                    if img_pil.width > 1 and img_pil.height > 1: # Avoid tiny/invalid images
                       img_pil.save(image_path)
                       print(f"Saved image: {image_path}")
                       extracted_data.append({
                           "type": "image_ref", # Placeholder type before analysis
                           "image_path": image_path,
                           "page_number": page_number,
                           "img_index": img_index,
                       })
                    else:
                        print(f"Skipped saving potentially invalid image (xref {xref}) on page {page_number}.")

                except Exception as e:
                    print(f"Error processing or saving image (xref {xref}) on page {page_number}: {e}")


        doc.close() # .close() method remains the same
        print(f"Finished processing PDF. Extracted {len(extracted_data)} items.")
        return extracted_data, doc_hash

    except Exception as e:
        print(f"Error opening or processing PDF {pdf_path}: {e}")
        return [], None
# pdf_processor.py
import pymupdf
import os
from PIL import Image
import io
import hashlib
import time

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    print("Warning: camelot-py not found. Table extraction will be skipped.")
    CAMELOT_AVAILABLE = False

def get_doc_hash(filepath):
    # ... (no changes) ...
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as file:
        while chunk := file.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


# --- ADDED: Helper function for safe coordinate handling ---
def safe_round_coords(coords, page_num, item_type, item_idx):
    """Safely converts coordinates to float and rounds them."""
    rounded_coords = []
    try:
        for coord in coords:
            rounded_coords.append(round(float(coord), 2))
        if len(rounded_coords) == 4:
             return rounded_coords
        else:
             print(f"Warning: Unexpected number of coordinates ({len(rounded_coords)}) for {item_type} {item_idx} on page {page_num}. Expected 4.")
             return None
    except (ValueError, TypeError) as e:
        print(f"Warning: Invalid coordinate format for {item_type} {item_idx} on page {page_num}. Coords: {coords}. Error: {e}. Skipping bbox.")
        return None
# ---------------------------------------------------------


def extract_pdf_data(pdf_path, image_output_dir="images"):
    """Extracts text, images, tables, screenshots, bounding boxes, and links."""
    extracted_data = []
    doc_hash = get_doc_hash(pdf_path)
    image_save_dir = os.path.join(image_output_dir, doc_hash)
    os.makedirs(image_save_dir, exist_ok=True)

    doc = None # Initialize doc to None for finally block
    try:
        doc = pymupdf.open(pdf_path)
        print(f"Processing PDF: {pdf_path}, Pages: {len(doc)}")

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_number = page_num + 1
            page_items = []

            # --- Generate page screenshot ---
            screenshot_path = None
            try:
                pix = page.get_pixmap(dpi=150)
                screenshot_filename = f"page_{page_number}_screenshot.png"
                screenshot_path = os.path.join(image_save_dir, screenshot_filename)
                pix.save(screenshot_path)
                # print(f"Saved page screenshot: {screenshot_path}") # Less verbose
            except Exception as screen_e:
                 print(f"Error generating screenshot for page {page_number}: {screen_e}")
                 screenshot_path = None
            # ---------------------------------

            # --- Extract Text Blocks ---
            text_blocks = page.get_text("blocks")
            for i, tb in enumerate(text_blocks):
                x0, y0, x1, y1, block_text, _, block_type = tb
                if block_text.strip() and block_type == 0:
                    # --- Use safe rounding ---
                    bbox = safe_round_coords([x0, y0, x1, y1], page_number, "text", i)
                    # -------------------------
                    page_items.append({
                        "type": "text", "content": block_text.strip(),
                        "page_number": page_number, "chunk_index": i,
                        "bounding_box": bbox, # Will be None if rounding failed
                        "page_screenshot_path": screenshot_path
                    })
            # -------------------------

            # --- Extract Images ---
            image_list = page.get_images(full=True)
            # print(f"Page {page_number}: Found {len(image_list)} images.") # Less verbose
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                if not base_image: continue

                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                img_bbox_page = img[2:6]
                # --- Use safe rounding ---
                img_bbox = safe_round_coords(img_bbox_page, page_number, "image", img_index)
                # -------------------------

                image_filename = f"page_{page_number}_img_{img_index}.{image_ext}"
                image_path = os.path.join(image_save_dir, image_filename)

                try:
                    # (Save image logic - no changes needed here)
                    img_pil = Image.open(io.BytesIO(image_bytes)); img_pil.verify()
                    img_pil = Image.open(io.BytesIO(image_bytes))
                    if img_pil.width > 1 and img_pil.height > 1:
                       img_pil.save(image_path)
                       # print(f"Saved image: {image_path}") # Less verbose
                       page_items.append({
                           "type": "image_ref", "image_path": image_path,
                           "page_number": page_number, "img_index": img_index,
                           "bounding_box": img_bbox, # Will be None if rounding failed
                           "page_screenshot_path": screenshot_path
                       })
                    # else: print(f"Skipped invalid image {xref} page {page_number}.")
                except Exception as e: print(f"Error saving image {xref} page {page_number}: {e}")
            # ----------------------

            # --- Table Extraction ---
            if CAMELOT_AVAILABLE:
                try:
                    # (Camelot logic - no changes needed here)
                    tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='stream', suppress_stdout=True)
                    if tables.n == 0: tables = camelot.read_pdf(pdf_path, pages=str(page_number), flavor='lattice', suppress_stdout=True)
                    # print(f"Page {page_number}: Found {tables.n} tables.") # Less verbose
                    for table_index, table in enumerate(tables):
                        table_markdown = table.df.to_markdown(index=False)
                        page_items.append({
                            "type": "table", "content": table_markdown,
                            "page_number": page_number, "chunk_index": f"t{table_index}",
                            "bounding_box": None, # Still skipping table bbox for now
                            "page_screenshot_path": screenshot_path
                        })
                except Exception as table_e:
                    print(f"Warning: Camelot failed on page {page_number}: {table_e}")
            # -----------------------

            # --- Text-Image Linking ---
            page_texts = [item for item in page_items if item['type'] == 'text']
            page_images = [item for item in page_items if item['type'] == 'image_ref']
            PROXIMITY_THRESHOLD_Y = 75

            for text_item in page_texts:
                text_bbox = text_item.get('bounding_box')
                if not text_bbox: continue # Skip if text bbox failed
                text_center_y = (text_bbox[1] + text_bbox[3]) / 2
                text_item['linked_items'] = []

                for img_item in page_images:
                    img_bbox = img_item.get('bounding_box')
                    if not img_bbox: continue # Skip if image bbox failed
                    img_center_y = (img_bbox[1] + img_bbox[3]) / 2
                    link_found = False
                    if abs(text_center_y - img_center_y) < PROXIMITY_THRESHOLD_Y: link_found = True
                    elif (abs(text_bbox[3] - img_bbox[1]) < PROXIMITY_THRESHOLD_Y / 2) or \
                         (abs(text_bbox[1] - img_bbox[3]) < PROXIMITY_THRESHOLD_Y / 2): link_found = True

                    if link_found:
                        text_id = f"text_p{text_item['page_number']}_c{text_item['chunk_index']}"
                        img_id = f"img_p{img_item['page_number']}_i{img_item['img_index']}"
                        text_item['linked_items'].append(img_id)
                        if 'linked_items' not in img_item: img_item['linked_items'] = []
                        img_item['linked_items'].append(text_id)
                        # print(f"Linked {text_id} and {img_id}") # Less verbose
            # -----------------------

            extracted_data.extend(page_items)

        print(f"Finished processing PDF. Extracted {len(extracted_data)} items.")
        return extracted_data, doc_hash

    except Exception as e:
        # This is the catch block that likely printed the error you saw
        print(f"Error opening or processing PDF {pdf_path}: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for better debugging
        return [], None # Return empty list and None hash on failure
    finally:
        # --- Ensure the document is closed ---
        if doc:
            doc.close()
            print(f"Closed PDF document: {pdf_path}")
        # -------------------------------------
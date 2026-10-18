import os
import fitz
import json
def normalize_pdf_to_structured_content(pdf_path, doc_id, filename=None):
    print(f"[+] Initializing Stage B processing for: {pdf_path}")
    if not os.path.exists(pdf_path): return None
    os.makedirs("documents", exist_ok=True)
    os.makedirs("ocr_cache", exist_ok=True)
    try: doc = fitz.open(pdf_path)
    except Exception as e: print(e); return None
    structured_content = {
        "doc_id": doc_id,
        "filename": filename or os.path.basename(pdf_path),
        "total_pages": len(doc),
        "pages": {},
    }
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1
        image_target_path = f"documents/{doc_id}_page_{page_num}.png"
        pix = page.get_pixmap(dpi=150)
        pix.save(image_target_path)
        raw_blocks = page.get_text("blocks")
        normalized_blocks = []
        for b in raw_blocks:
            if len(b) > 4:
                normalized_blocks.append({
                    "bbox": [round(b[0], 2), round(b[1], 2), round(b[2], 2), round(b[3], 2)],
                    "text_lines": [l.strip() for l in str(b[4]).split("\n") if l.strip()]
                })
        structured_content["pages"][f"page_{page_num}"] = {
            "image_path": image_target_path,
            "dimensions": [page.rect.width, page.rect.height],
            "spatial_blocks": normalized_blocks,
            "raw_text_stream": page.get_text("text").strip()
        }
    output_cache_file = f"ocr_cache/{doc_id}_structured.json"
    with open(output_cache_file, "w", encoding="utf-8") as f:
        json.dump(structured_content, f, ensure_ascii=False, indent=2)
    print(f"[✓] Stage B Extraction Complete: Caching completed at {output_cache_file}")
    return structured_content
if __name__ == "__main__":
    print("[+] Creating small synthetic local test document...")
    syn_doc = fitz.open()
    page = syn_doc.new_page(width=612, height=792)
    page.insert_text((50, 100), "Sovereign AI Framework Core Verification Document", fontsize=14)
    page.insert_text((50, 150), "Total Balance Due: $1,450.00", fontsize=12)
    page.insert_text((50, 200), "Verification Token: PASS-STAGE-B", fontsize=12)
    syn_doc.save("sample_invoice.pdf")
    syn_doc.close()
    normalize_pdf_to_structured_content("sample_invoice.pdf", "doc_test_01")

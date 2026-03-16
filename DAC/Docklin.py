from pathlib import Path
import json
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter

# 1. Setup OCR for Indian Languages (e.g., Hindi and English)
# pipeline_options = PdfPipelineOptions()
# pipeline_options.ocr_options = EasyOcrOptions(lang=["hi", "en"]) 

# 2. Convert the document
converter = DocumentConverter()
result = converter.convert(r"C:\Users\ICT\Desktop\Adjustment of SETs 2024.pdf")

# 3. Export to Markdown
print(result.document.export_to_markdown())

# doc_dict = result.document.export_to_dict()
# output_path = Path(r"C:\voucher_ocr_system\output3.json")

# with open(output_path, "w", encoding="utf-8") as f:
#     json.dump(doc_dict, f, indent=4, ensure_ascii=False)

# print("JSON exported successfully.")

import streamlit as st
import tempfile
from pathlib import Path
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter

st.set_page_config(page_title="PDF OCR Extractor", layout="wide")

st.title("📄 PDF OCR to Markdown Converter")

# Tabs
tab1, tab2 = st.tabs(["Upload PDF Files", "About"])

# ---------------- TAB 1 ----------------
with tab1:

    st.header("Upload PDF Files")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        # Setup OCR
        pipeline_options = PdfPipelineOptions()
        pipeline_options.ocr_options = EasyOcrOptions(lang=["hi", "en"])

        converter = DocumentConverter(pipeline_options=pipeline_options)

        results = []

        for file in uploaded_files:

            with st.spinner(f"Processing {file.name}..."):

                # Save file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(file.read())
                    tmp_path = tmp.name

                # Convert
                result = converter.convert(tmp_path)

                markdown_text = result.document.export_to_markdown()

                results.append((file.name, markdown_text))

        st.success("Processing Complete!")

        st.subheader("Download Results")

        for filename, markdown_text in results:

            output_name = Path(filename).stem + ".md"

            st.download_button(
                label=f"Download {output_name}",
                data=markdown_text,
                file_name=output_name,
                mime="text/markdown"
            )

# ---------------- TAB 2 ----------------
with tab2:

    st.header("About")
    st.write(
        """
        This app extracts text from scanned PDFs using **Docling + EasyOCR**.

        Features:
        - Supports **multiple PDF uploads**
        - OCR for **Hindi + English**
        - Converts document into **Markdown format**
        - Each PDF produces a **separate downloadable output**
        """
    )
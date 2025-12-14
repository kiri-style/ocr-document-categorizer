import streamlit as st
import numpy as np
from PIL import Image
import easyocr
from pdf2image import convert_from_bytes

from utils import detect_document, categorize_text

st.set_page_config(page_title="OCR Document Categorizer", layout="centered")

st.title("📄 OCR Document Categorizer")

st.markdown("""
This application automatically:
1. Detects and crops a document inside an image or PDF  
2. Extracts editable text using OCR  
3. Categorizes the extracted content into **user-defined headings**
""")

# ======================
# Upload
# ======================
uploaded_file = st.file_uploader(
    "Upload a document (Image or PDF)",
    type=["jpg", "png", "jpeg", "pdf"]
)

if uploaded_file:
    # ======================
    # Load image
    # ======================
    if uploaded_file.type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read())
        image = pages[0]  # first page only
    else:
        image = Image.open(uploaded_file).convert("RGB")

    image_np = np.array(image)

    st.subheader("📷 Original Image")
    st.image(image, use_column_width=True)

    # ======================
    # Category input
    # ======================
    st.subheader("🗂 Categorization Settings")

    default_categories = [
        "Header / Title",
        "Dates",
        "Names / Entities",
        "Numbers / Amounts",
        "Main Content"
    ]

    categories_input = st.text_area(
        "Define categories (one per line)",
        "\n".join(default_categories),
        height=120
    )

    user_categories = [
        c.strip() for c in categories_input.split("\n") if c.strip()
    ]

    # ======================
    # Process
    # ======================
    if st.button("🚀 Process Document"):
        with st.spinner("Detecting document..."):
            cropped = detect_document(image_np)

        st.subheader("📄 Detected Document")
        st.image(cropped, use_column_width=True)

        # OCR
        with st.spinner("Running OCR..."):
            reader = easyocr.Reader(['en'], gpu=False)
            results = reader.readtext(cropped)

        extracted_text = "\n".join([res[1] for res in results])

        st.subheader("📝 Extracted Text (Editable)")
        st.text_area(
            "OCR Result",
            extracted_text,
            height=250
        )

        # Categorization
        with st.spinner("Categorizing text..."):
            categorized = categorize_text(extracted_text, user_categories)

        st.subheader("📌 Categorized Content")

        for category, content in categorized.items():
            st.markdown(f"### {category}")
            if content:
                for line in content:
                    st.markdown(f"- {line}")
            else:
                st.markdown("_No content detected_")
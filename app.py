import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from utils import detect_document
from pdf2image import convert_from_bytes

st.set_page_config(page_title="OCR Document Categorizer")

st.title("📄 OCR Document Categorizer")

st.markdown("""
This application detects a document inside an image, crops it automatically,
extracts editable text using OCR, and categorizes the content into sections.
""")

uploaded_file = st.file_uploader(
    "Upload a document (Image or PDF)",
    type=["jpg", "png", "jpeg", "pdf"]
)
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        pages = convert_from_bytes(uploaded_file.read())
        image = pages[0]  # première page
    else:
        image = Image.open(uploaded_file)

    image_np = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    if st.button("Process Document"):
        cropped = detect_document(image_np)

        st.subheader("Detected Document")
        st.image(cropped, use_column_width=True)

        reader = easyocr.Reader(['en'], gpu=False)
        results = reader.readtext(cropped)

        text = " ".join([res[1] for res in results])

        st.subheader("Extracted Text")
        st.text_area("Editable text", text, height=200)

        st.subheader("Categorized Content")
        lines = text.split(".")
        categorized = {
            "Title": lines[0] if lines else "",
            "Content": ". ".join(lines[1:4]),
            "Summary": ". ".join(lines[-2:])
        }

        for k, v in categorized.items():
            st.markdown(f"**{k}:** {v}")
